"""HADO Smart Court — 통합 실행 (Level 1 + Level 2).

파이프라인
---------
Level 1: 카메라 → YOLOv8 감지 → IoU 트래커 → Homography 변환 → Bird-eye view
Level 2: + 전술 분석 → 추천 위치 화살표 + (선택) 음성 안내

키 단축키
---------
ESC : 종료
s   : 현재 합성 프레임 스냅샷 (data/snapshots/)
r   : CSV 좌표 로깅 시작/중지 (data/position_logs/)
c   : 모든 트랙 궤적 초기화
v   : 음성 가이드 토글 (Level 2)
t   : 팀 배정 초기화 (Level 2)

실행 예시
---------
    python -m src.main                          # Level 1 (기본)
    python -m src.main --level 2                # Level 2 (가이드 포함)
    python -m src.main --level 2 --voice        # + 음성 안내
    python -m src.main --headless --record demo.mp4  # SSH 환경: 영상으로 저장
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.analyzer import analyze_match
from src.camera import Camera
from src.detector import PersonDetector
from src.guide import VoiceGuide, draw_guide_on_birdeye, draw_guide_text_on_frame
from src.homography import Calibration, pixel_to_court
from src.movement_model import MovementModel
from src.recorder import MatchRecorder
from src.tactic_engine import PlayerState, TacticEngine
from src.tracker import IoUTracker
from src.upload import upload_match
from src.visualizer import (
    combine_views,
    draw_detections_on_frame,
    draw_hud,
    draw_players_on_birdeye,
    render_court_birdeye,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_csv_writer():
    """녹화 시작 시 새 CSV 파일 생성."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "data" / "position_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"positions_{ts}.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "frame_idx", "track_id", "team",
        "x_m", "y_m",
        "target_x_m", "target_y_m", "urgency", "rule",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence",
    ])
    print(f"[Recording] ● 시작 — {path}")
    return f, writer, path


def tracks_to_player_states(tracks, calib: Calibration) -> list[PlayerState]:
    """트래커 출력 → 전술 엔진 입력 변환."""
    states = []
    if not tracks:
        return states
    foot_px = np.array([t.foot_point for t in tracks], dtype=np.float32)
    foot_m = pixel_to_court(foot_px, calib)
    for t, m in zip(tracks, foot_m):
        states.append(PlayerState(
            track_id=t.track_id,
            court_x=float(m[0]),
            court_y=float(m[1]),
            confidence=t.confidence,
        ))
    return states


def run(args):
    config = load_config(Path(args.config))
    print(f"[Main] 설정 로드: {args.config}")

    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"[!] 캘리브레이션 없음: {calib_path}")
        print(f"    먼저 `python -m src.calibrate` 실행하세요.")
        return 1
    calib = Calibration.from_json(calib_path)
    print(f"[Main] 캘리브레이션: {calib.court_width_m}m × {calib.court_height_m}m")
    print(f"[Main] Level: {args.level}")

    # 모듈 초기화
    detector = PersonDetector(
        model_path=config["detector"]["model_path"],
        imgsz=config["detector"]["imgsz"],
        conf_threshold=config["detector"]["conf_threshold"],
        iou_threshold=config["detector"]["iou_threshold"],
        target_class=config["detector"]["target_class"],
    )
    tracker = IoUTracker(
        iou_threshold=config["tracker"]["iou_threshold"],
        max_lost_frames=config["tracker"]["max_lost_frames"],
    )
    if args.level >= 2:
        mv_csv = PROJECT_ROOT / "data" / "movement_data.csv"
        movement_model = MovementModel(mv_csv) if mv_csv.exists() else None
        if movement_model:
            print(f"[Main] MovementModel 로드: {movement_model.pattern_count}개 패턴")
        tactic_engine = TacticEngine.from_config(config, movement_model=movement_model)
    else:
        tactic_engine = None
    voice_guide = VoiceGuide(enabled=args.voice) if (args.level >= 2 and args.voice) else None

    px_per_m = config["court"]["render_scale_px_per_m"]
    bg_color = tuple(config["court"]["bg_color"])
    line_color = tuple(config["court"]["line_color"])
    court_template = render_court_birdeye(calib, px_per_m=px_per_m,
                                          bg_color=bg_color, line_color=line_color)

    # 카메라 또는 비디오 입력
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    cam = Camera(
        source=source,
        width=config["camera"]["width"],
        height=config["camera"]["height"],
        fps=config["camera"]["fps"],
    )
    cam.open()

    # MP4 출력 (옵션: --record)
    video_writer = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 첫 프레임으로 해상도 추정
        ok, first_frame = cam.read()
        if not ok:
            print("[!] 첫 프레임 읽기 실패")
            cam.close()
            return 1
        # 합성 화면 크기 = 카메라 너비 + 4(구분선) + scaled bird-eye 너비
        h_cam = first_frame.shape[0]
        court_w_scaled = int(court_template.shape[1] * (h_cam / court_template.shape[0]))
        out_w = first_frame.shape[1] + 4 + court_w_scaled
        out_h = h_cam
        video_writer = cv2.VideoWriter(str(rec_path), fourcc, 20.0, (out_w, out_h))
        print(f"[Record] {rec_path} ({out_w}x{out_h} @20fps)")

    # 경기 녹화 모드 (--match): 타임스탬프 폴더 + CSV + 사후 분석
    match_recorder: MatchRecorder | None = None
    if args.match:
        matches_dir = PROJECT_ROOT / "data" / "matches"
        match_recorder = MatchRecorder(matches_dir, calib)

    # 상태 변수
    fps = 0.0
    fps_alpha = 0.9
    last_t = time.time()
    frame_idx = 0
    recording = False
    csv_file = None
    csv_writer = None
    csv_path: Path | None = None

    snap_dir = PROJECT_ROOT / "data" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    if not args.headless:
        cv2.namedWindow("HADO Smart Court", cv2.WINDOW_NORMAL)
        print("[Main] 실행 — ESC 종료 / s 스냅 / r 녹화 / c 궤적 / v 음성 / t 팀 초기화")
    else:
        print("[Main] Headless 모드 — Ctrl+C 또는 --max-frames 도달 시 종료")

    try:
        while True:
            if args.record and frame_idx == 0:
                frame = first_frame
            else:
                ok, frame = cam.read()
                if not ok:
                    print("[!] 프레임 실패")
                    break

            # ===== Level 1 =====
            dets = detector.detect(frame)
            tracks = tracker.update(dets)

            # ===== Level 2 =====
            advices = []
            if tactic_engine is not None and tracks:
                player_states = tracks_to_player_states(tracks, calib)
                advices = tactic_engine.analyze(player_states)
                if voice_guide:
                    voice_guide.speak_advices(advices)

            # ===== CSV 로깅 =====
            if recording and tracks and csv_writer is not None:
                ts = time.time()
                advice_by_id = {a.track_id: a for a in advices}
                for t in tracks:
                    foot_px = np.array([t.foot_point], dtype=np.float32)
                    foot_m = pixel_to_court(foot_px, calib)[0]
                    a = advice_by_id.get(t.track_id)
                    csv_writer.writerow([
                        f"{ts:.3f}", frame_idx, t.track_id,
                        a.team if a else "",
                        f"{foot_m[0]:.3f}", f"{foot_m[1]:.3f}",
                        f"{a.target_pos[0]:.3f}" if a else "",
                        f"{a.target_pos[1]:.3f}" if a else "",
                        a.urgency if a else "",
                        a.rule if a else "",
                        f"{t.bbox[0]:.1f}", f"{t.bbox[1]:.1f}",
                        f"{t.bbox[2]:.1f}", f"{t.bbox[3]:.1f}",
                        f"{t.confidence:.3f}",
                    ])

            # ===== 시각화 =====
            cam_view = draw_detections_on_frame(frame.copy(), tracks)
            birdeye = draw_players_on_birdeye(
                court_template, tracks, calib, px_per_m=px_per_m,
                radius=config["visualization"]["player_radius_px"],
                show_trajectory=True,
                trajectory_length=config["visualization"]["trajectory_length"],
            )
            if advices:
                birdeye = draw_guide_on_birdeye(birdeye, advices, px_per_m=px_per_m)
                cam_view = draw_guide_text_on_frame(cam_view, advices)

            combined = combine_views(cam_view, birdeye)
            combined = draw_hud(combined, fps, len(tracks), recording)

            if video_writer is not None:
                video_writer.write(combined)

            if match_recorder is not None:
                if frame_idx == 0:
                    match_recorder.start(combined.shape)
                match_recorder.write(combined, tracks, birdeye)

            if not args.headless:
                cv2.imshow("HADO Smart Court", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):
                    snap_path = snap_dir / f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(snap_path), combined)
                    print(f"[Snapshot] {snap_path}")
                elif key == ord('r'):
                    if not recording:
                        csv_file, csv_writer, csv_path = make_csv_writer()
                        recording = True
                    else:
                        if csv_file:
                            csv_file.close()
                        print(f"[Recording] ■ 종료 — {csv_path}")
                        csv_file = None
                        csv_writer = None
                        recording = False
                elif key == ord('c'):
                    for t in tracker.tracks:
                        t.history.clear()
                    print("[Main] 궤적 초기화")
                elif key == ord('v') and voice_guide:
                    voice_guide.enabled = not voice_guide.enabled
                    print(f"[Voice] {'활성화' if voice_guide.enabled else '비활성화'}")
                elif key == ord('t') and tactic_engine:
                    tactic_engine.reset_teams()
                    print("[Tactic] 팀 배정 초기화")

            # FPS 추정
            now = time.time()
            inst_fps = 1.0 / max(0.001, now - last_t)
            fps = fps_alpha * fps + (1 - fps_alpha) * inst_fps if fps > 0 else inst_fps
            last_t = now
            frame_idx += 1

            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[Main] max_frames({args.max_frames}) 도달")
                break

    except KeyboardInterrupt:
        print("\n[Main] 사용자 중단")
    finally:
        if csv_file:
            csv_file.close()
        if video_writer:
            video_writer.release()
            print(f"[Record] 저장 완료: {args.record}")
        if match_recorder is not None:
            match_dir = match_recorder.stop()
            analyze_match(match_dir, calib)
            if args.upload:
                upload_match(match_dir)
        if voice_guide:
            voice_guide.close()
        cam.close()
        cv2.destroyAllWindows()

    print(f"[Main] 종료 — 평균 FPS {fps:.2f}, 총 {frame_idx} 프레임")
    return 0


def main():
    parser = argparse.ArgumentParser(description="HADO Smart Court Level 1/2")
    parser.add_argument("--source", default=0, help="카메라 소스 또는 비디오 파일")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "court_config.yaml"))
    parser.add_argument("--calibration", default=str(PROJECT_ROOT / "config" / "calibration.json"))
    parser.add_argument("--level", type=int, default=1, choices=[1, 2],
                        help="1: 위치 추적만, 2: 전술 가이드 포함")
    parser.add_argument("--voice", action="store_true", help="음성 가이드 활성화 (Level 2)")
    parser.add_argument("--headless", action="store_true", help="GUI 창 없이 실행 (SSH 환경용)")
    parser.add_argument("--record", default=None, help="출력 영상 mp4 경로")
    parser.add_argument("--max-frames", type=int, default=0, help="이만큼 처리 후 종료 (0=무제한)")
    parser.add_argument("--match", action="store_true",
                        help="경기 녹화 모드: data/matches/<timestamp>/ 에 MP4+CSV+통계 저장")
    parser.add_argument("--upload", action="store_true",
                        help="경기 종료 후 구글 드라이브 자동 업로드 (rclone 필요)")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
