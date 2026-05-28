"""카메라/캘리브레이션 없이 버드아이뷰 파이프라인을 시연하는 데모.

calibration.json, YOLOv8, 실제 카메라 없이도 동작.
가짜 캘리브레이션 + 시뮬레이션 선수 궤적으로 combine_views 전체 화면을 확인.

실행:
    python -m src.demo                  # 창 표시 + data/demo.mp4 저장
    python -m src.demo --headless       # 창 없이 video만 저장
    python -m src.demo --frames 60      # 빠른 미리보기
    ./run.sh demo
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from src.detector import Detection
from src.guide import draw_guide_on_birdeye
from src.homography import compute_homography, court_to_pixel
from src.movement_model import MovementModel
from src.tactic_engine import PlayerState, TacticEngine
from src.tracker import IoUTracker
from src.visualizer import (
    combine_views,
    draw_detections_on_frame,
    draw_hud,
    draw_players_on_birdeye,
    render_court_birdeye,
)

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COURT_W = 10.0
_COURT_H = 6.0
_FRAME_W, _FRAME_H = 1280, 720

_CONFIG_PATH = PROJECT_ROOT / "config" / "court_config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_calib():
    """1280×720 가상 카메라: 코트를 위에서 비스듬히 내려다보는 원근감."""
    return compute_homography(
        corners_pixel=np.array([
            [180, 140],   # 좌상
            [1100, 140],  # 우상
            [1160, 590],  # 우하
            [120, 590],   # 좌하
        ], dtype=np.float32),
        court_width_m=_COURT_W,
        court_height_m=_COURT_H,
        image_size=(_FRAME_W, _FRAME_H),
    )


# (base_x, base_y, amp_x, amp_y, freq_x, freq_y, phase)
_PLAYER_PARAMS = [
    (2.0, 1.5, 1.2, 1.0, 0.80, 1.10, 0.0),   # #1 Team A technician
    (1.5, 4.0, 1.0, 1.2, 1.20, 0.70, 1.5),   # #2 Team A defender
    (3.5, 3.0, 1.0, 1.5, 0.90, 1.30, 0.8),   # #3 Team A main_attacker
    (8.0, 1.5, 1.2, 1.0, 0.80, 1.10, 1.6),   # #4 Team B main_attacker
    (8.5, 4.0, 1.0, 1.2, 1.20, 0.70, 2.8),   # #5 Team B defender
    (6.5, 3.0, 1.0, 1.5, 0.90, 1.30, 3.5),   # #6 Team B technician
]
_PLAYER_ROLES = {
    1: "technician", 2: "defender",    3: "main_attacker",
    4: "main_attacker", 5: "defender", 6: "technician",
}


def _player_pos(frame_idx: int, pid: int) -> tuple[float, float]:
    """선수 ID(1~6)별 시뮬레이션 위치 (코트 좌표 m). 진영 불가침 적용."""
    t = frame_idx / 30.0
    bx, by, ax, ay, wx, wy, ph = _PLAYER_PARAMS[pid - 1]
    x = bx + ax * math.sin(wx * t + ph)
    y = by + ay * math.cos(wy * t + ph * 1.3)
    # 진영 불가침: 1~3=팀A(x<5), 4~6=팀B(x>5)
    if pid <= 3:
        x = max(0.05, min(4.95, x))
    else:
        x = max(5.05, min(_COURT_W - 0.05, x))
    return x, max(0.05, min(_COURT_H - 0.05, y))


def _build_camera_bg(calib) -> np.ndarray:
    """코트 코너 좌표로 원근 배경 이미지를 만든다 (재사용 목적으로 1회만 생성)."""
    bg = np.full((_FRAME_H, _FRAME_W, 3), (30, 30, 30), dtype=np.uint8)
    pts = calib.corners_pixel.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(bg, [pts], (45, 55, 40))
    cv2.polylines(bg, [pts], True, (200, 200, 200), 2)

    # 중앙선
    mid_top = court_to_pixel(np.array([[5.0, 0.0]]), calib)[0].astype(int)
    mid_bot = court_to_pixel(np.array([[5.0, _COURT_H]]), calib)[0].astype(int)
    cv2.line(bg, tuple(mid_top), tuple(mid_bot), (150, 150, 60), 1, cv2.LINE_AA)

    # 1m 그리드
    for xi in range(1, int(_COURT_W)):
        p0 = court_to_pixel(np.array([[float(xi), 0.0]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[float(xi), _COURT_H]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), (55, 65, 50), 1, cv2.LINE_AA)
    for yi in range(1, int(_COURT_H) + 1):
        p0 = court_to_pixel(np.array([[0.0, float(yi)]]), calib)[0].astype(int)
        p1 = court_to_pixel(np.array([[_COURT_W, float(yi)]]), calib)[0].astype(int)
        cv2.line(bg, tuple(p0), tuple(p1), (55, 65, 50), 1, cv2.LINE_AA)

    return bg


def _court_to_bbox(xm: float, ym: float, calib) -> np.ndarray:
    """코트 좌표 → 카메라 픽셀 바운딩 박스. 원근감(depth)으로 크기를 추정한다."""
    foot_px = court_to_pixel(np.array([[xm, ym]]), calib)[0]
    fx, fy = float(foot_px[0]), float(foot_px[1])

    cam_top = float(calib.corners_pixel[:, 1].min())
    cam_bot = float(calib.corners_pixel[:, 1].max())
    depth = max(0.0, min(1.0, (fy - cam_top) / max(1.0, cam_bot - cam_top)))

    h_box = int(70 + 110 * depth)
    w_box = int(h_box * 0.42)
    return np.array([fx - w_box / 2, fy - h_box, fx + w_box / 2, fy], dtype=np.float32)


def run(args) -> int:
    config = _load_config()
    calib = _make_calib()
    bg = _build_camera_bg(calib)
    px_per_m = 100
    court_tmpl = render_court_birdeye(calib, px_per_m=px_per_m)
    tracker = IoUTracker(iou_threshold=0.3, max_lost_frames=10)

    # Level 2: TacticEngine + MovementModel
    mv_csv = PROJECT_ROOT / "data" / "movement_data.csv"
    movement_model = MovementModel(mv_csv) if mv_csv.exists() else None
    tactic_engine = TacticEngine.from_config(config, movement_model=movement_model)
    if movement_model:
        print(f"[Demo] MovementModel: {movement_model.pattern_count}패턴 로드")
    # 팀 배정 고정 (track_id 1~3=A, 4~6=B)
    for tid in range(1, 4):
        tactic_engine._team_assignment[tid] = "A"
    for tid in range(4, 7):
        tactic_engine._team_assignment[tid] = "B"

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    vid_path = out_dir / "demo.mp4"

    # 출력 크기: combine_views 결과 기준으로 결정
    _sample = combine_views(np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8), court_tmpl)
    out_h, out_w = _sample.shape[:2]
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (out_w, out_h))
    print(f"[Demo] 저장: {vid_path}  ({out_w}×{out_h} @30fps, {args.frames}프레임)")

    if not args.headless:
        cv2.namedWindow("HADO Demo", cv2.WINDOW_NORMAL)

    preview_saved = False
    for fi in range(args.frames):
        dets = [
            Detection(
                x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]),
                confidence=0.90,
            )
            for pid in range(1, 7)
            for bbox in [_court_to_bbox(*_player_pos(fi, pid), calib)]
        ]

        tracks = tracker.update(dets)

        # Level 2: 전술 분석 (데모에서는 시뮬레이션 좌표를 그대로 사용)
        player_states = [
            PlayerState(
                track_id=t.track_id,
                court_x=_player_pos(fi, t.track_id)[0],
                court_y=_player_pos(fi, t.track_id)[1],
                role=_PLAYER_ROLES.get(t.track_id, ""),
            )
            for t in tracks
        ]
        advices = tactic_engine.analyze(player_states)

        cam_view = bg.copy()
        draw_detections_on_frame(cam_view, tracks)

        birdeye = draw_players_on_birdeye(
            court_tmpl, tracks, calib,
            px_per_m=px_per_m,
            show_trajectory=True,
            trajectory_length=45,
        )
        birdeye = draw_guide_on_birdeye(birdeye, advices, px_per_m=px_per_m)

        combined = combine_views(cam_view, birdeye)
        draw_hud(combined, fps=30.0, n_players=len(tracks))
        writer.write(combined)

        # 중간 시점에 프리뷰 PNG 1장 저장
        if not preview_saved and fi >= args.frames // 2:
            preview_path = out_dir / "demo_preview.png"
            cv2.imwrite(str(preview_path), combined)
            print(f"[Demo] 프리뷰: {preview_path}")
            preview_saved = True

        if not args.headless:
            cv2.imshow("HADO Demo", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if fi % 90 == 0:
            print(f"[Demo] {fi}/{args.frames} 프레임...")

    writer.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"[Demo] 완료 — {vid_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="HADO 버드아이뷰 데모 (카메라 불필요)")
    parser.add_argument("--headless", action="store_true", help="창 없이 실행")
    parser.add_argument("--frames", type=int, default=300, help="총 프레임 수 (300=10초)")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
