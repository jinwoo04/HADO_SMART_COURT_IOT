"""YOLOv8-pose 파이프라인 데모.

실제 카메라 영상(AR 이펙트 없음)에서 선수 감지·추적·자세 분류·
이동 예측까지 수행. bird-eye view와 함께 좌우 분할 화면 출력.

실행:
    python -m src.demo_pose --video data/1.mp4
    python -m src.demo_pose --video data/1.mp4 --headless
    python -m src.demo_pose --video data/1.mp4 --out data/pose_out.mp4
    ./run.sh demo_pose --video data/1.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.detector import Detection, PersonDetector
from src.guide import draw_guide_on_birdeye
from src.homography import (
    Intrinsics, build_undistort_maps, compute_homography,
    pixel_to_court, undistort_frame,
)
from src.movement_model import MovementModel, MovementPrediction
from src.pose import (
    POSTURE_COLOR, POSTURE_KO,
    analyze_pose, draw_movement_arrow, draw_posture_label, draw_skeleton,
)
from src.tactic_engine import PlayerState, TacticEngine
from src.tracker import IoUTracker, Track
from src.visualizer import draw_hud, render_court_birdeye

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COURT_W, _COURT_H = 10.0, 6.0


# ---------- 캘리브레이션 ----------
def _make_calib(frame_w: int, frame_h: int):
    """영상 해상도에 맞춰 기본 캘리브레이션 생성.

    실제 코트 촬영이 아닌 영상에서는 근사값이므로
    bird-eye 정확도보다 파이프라인 동작 확인이 목적.
    """
    m = 0.10   # 여백 비율
    w, h = frame_w, frame_h
    corners = np.array([
        [w * m,       h * m      ],   # 좌상
        [w * (1 - m), h * m      ],   # 우상
        [w * (1 - m), h * (1 - m)],   # 우하
        [w * m,       h * (1 - m)],   # 좌하
    ], dtype=np.float32)
    return compute_homography(
        corners_pixel=corners,
        court_width_m=_COURT_W,
        court_height_m=_COURT_H,
        image_size=(frame_w, frame_h),
    )


# ---------- 팀 자동 배정 ----------
_team_cache: dict[int, str] = {}

def _assign_team(track_id: int, foot_x_px: float, frame_w: int) -> str:
    if track_id not in _team_cache:
        _team_cache[track_id] = "A" if foot_x_px < frame_w / 2 else "B"
    return _team_cache[track_id]


# ---------- 역할 추정 (코트 좌표 기반) ----------
def _infer_roles(
    tracks: list[Track],
    court_pts: np.ndarray,
) -> dict[int, str]:
    """코트 내 위치로 각 선수 역할 추정.

    중앙선(x=5m)에 가장 가까운 선수 → main_attacker
    나머지: 극단 레인(y<2 or y>4) → technician, 그 외 → defender
    """
    roles: dict[int, str] = {}
    team_players: dict[str, list[tuple[int, tuple[float, float]]]] = {"A": [], "B": []}

    for i, t in enumerate(tracks):
        if i >= len(court_pts):
            continue
        team = _team_cache.get(t.track_id, "A")
        team_players[team].append((t.track_id, (float(court_pts[i][0]), float(court_pts[i][1]))))

    for team, players in team_players.items():
        if not players:
            continue
        # 중앙선까지 거리 기준 정렬 → 가장 가까운 = main_attacker
        by_dist = sorted(players, key=lambda p: abs(p[1][0] - 5.0))
        roles[by_dist[0][0]] = "main_attacker"
        for tid, (cx, cy) in by_dist[1:]:
            roles[tid] = "technician" if (cy < 2.0 or cy > 4.0) else "defender"

    return roles


# ---------- 게임 상황 추정 (팀별 평균 x 위치) ----------
def _infer_contexts(
    tracks: list[Track],
    court_pts: np.ndarray,
) -> dict[str, str]:
    """팀별 평균 전진도로 attack/defend/transition 구분.

    팀A: 평균 x > 2.5m → attack  /  팀B: 평균 x < 7.5m → attack
    양 팀 모두 attack 국면이면 → transition
    """
    xs: dict[str, list[float]] = {"A": [], "B": []}
    for i, t in enumerate(tracks):
        if i >= len(court_pts):
            continue
        team = _team_cache.get(t.track_id, "A")
        xs[team].append(float(court_pts[i][0]))

    def _ctx(team: str) -> str:
        if not xs[team]:
            return "attack"
        avg = sum(xs[team]) / len(xs[team])
        return "attack" if (team == "A" and avg > 2.5) or (team == "B" and avg < 7.5) else "defend"

    ctx_a, ctx_b = _ctx("A"), _ctx("B")
    if ctx_a == "attack" and ctx_b == "attack":
        return {"A": "transition", "B": "transition"}
    return {"A": ctx_a, "B": ctx_b}


# ---------- 이동 예측 오버레이 ----------
def _draw_prediction_overlay(
    frame: np.ndarray,
    tracks: list[Track],
) -> None:
    """카메라 뷰: 속도 벡터 기반 단기 예측 화살표 (1~12프레임 앞)."""
    for t in tracks:
        if len(t.history) < 4:
            continue
        color = (0, 220, 255) if _team_cache.get(t.track_id) == "A" else (255, 180, 60)
        draw_movement_arrow(frame, list(t.history), color=color)


def _draw_model_predictions(
    birdeye: np.ndarray,
    preds: dict[int, tuple[MovementPrediction | None, str]],
    px_per_m: int,
) -> None:
    """버드아이뷰: k-NN 모델 예측 경로 표시.

    실선 경로  : 매칭된 패턴의 남은 이동 경로
    다이아몬드 : 목표 도달 위치
    숫자       : 신뢰도 (%)
    """
    for track_id, (pred, team) in preds.items():
        if pred is None or pred.confidence < 0.25:
            continue

        color = (80, 220, 130) if team == "A" else (80, 160, 255)

        # 전체 예측 경로
        path = pred.full_path
        for i in range(len(path) - 1):
            p1 = (int(path[i][0] * px_per_m), int(path[i][1] * px_per_m))
            p2 = (int(path[i + 1][0] * px_per_m), int(path[i + 1][1] * px_per_m))
            cv2.line(birdeye, p1, p2, color, 1, cv2.LINE_AA)

        # 목표 위치 마커
        tx = int(pred.target_pos[0] * px_per_m)
        ty = int(pred.target_pos[1] * px_per_m)
        cv2.drawMarker(birdeye, (tx, ty), color,
                       markerType=cv2.MARKER_DIAMOND, markerSize=14, thickness=2)

        # 신뢰도
        cv2.putText(birdeye, f"{pred.confidence:.0%}",
                    (tx + 7, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)


# ---------- 자세 통계 오버레이 ----------
def _draw_posture_stats(
    img: np.ndarray,
    posture_counts: dict[str, int],
) -> None:
    """우상단: 자세 분류 누적 통계."""
    try:
        from src.annotate import put_text_kr
        h, w = img.shape[:2]
        x0, y0 = w - 160, 8
        cv2.rectangle(img, (x0 - 4, y0 - 4), (w - 4, y0 + 58), (20, 20, 20), -1)
        put_text_kr(img, "자세 분류", (x0, y0), 12, (200, 200, 200))
        for i, (p, cnt) in enumerate(posture_counts.items()):
            color = POSTURE_COLOR.get(p, (180, 180, 180))
            label = POSTURE_KO.get(p, p)
            put_text_kr(img, f"{label}: {cnt}", (x0, y0 + 16 + i * 15), 11, color)
    except Exception:
        pass


# ---------- 메인 루프 ----------
def run(args) -> int:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[PoseDemo] 영상 열기 실패: {args.video}")
        return 1

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[PoseDemo] {args.video}  ({frame_w}×{frame_h} @{src_fps:.0f}fps, {total}프레임)")

    # 렌즈 왜곡 보정 맵 로드 (--intrinsic 지정 시)
    undistort_maps: tuple[np.ndarray, np.ndarray] | None = None
    intrinsic_path = Path(f"config/cam{args.cam_id}_intrinsics.json")
    if args.intrinsic:
        if intrinsic_path.exists():
            intr = Intrinsics.from_json(intrinsic_path)
            undistort_maps = build_undistort_maps(intr)
            print(f"[PoseDemo] 렌즈 보정 로드: {intrinsic_path}  "
                  f"(RMS={intr.reprojection_error:.3f}px)")
        else:
            print(f"[PoseDemo] ⚠ intrinsics 파일 없음: {intrinsic_path}")
            print(f"           먼저: ./run.sh calibrate --intrinsic --cam-id {args.cam_id}")

    # 모델 / 캘리브레이션 / 엔진 초기화
    detector = PersonDetector(model_path=args.model,
                               imgsz=args.imgsz, conf_threshold=args.conf)
    tracker       = IoUTracker(iou_threshold=0.3, max_lost_frames=12)
    px_per_m      = 80
    tactic_engine = TacticEngine()
    movement_model = MovementModel()
    print(f"[PoseDemo] MovementModel: {movement_model.pattern_count}개 패턴 로드")

    # ArUco 또는 근사 캘리브레이션
    aruco_mode = args.aruco
    if aruco_mode:
        from src.aruco_calibrate import calibrate_from_aruco, detect_markers, draw_aruco_overlay
        print("[PoseDemo] ArUco 모드: 첫 프레임에서 마커 자동 감지...")
        calib = _make_calib(frame_w, frame_h)   # 감지 전까지 근사값 사용
        _aruco_last_update = 0                   # 마지막 ArUco 업데이트 프레임
    else:
        calib = _make_calib(frame_w, frame_h)

    court_tmpl = render_court_birdeye(calib, px_per_m=px_per_m)

    # 출력 비디오
    out_path = Path(args.out) if args.out else None
    writer = None

    if not args.headless:
        cv2.namedWindow("HADO Pose Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HADO Pose Demo", 1400, 500)

    posture_counts: dict[str, int] = {"attack": 0, "shield": 0, "neutral": 0}
    t0 = time.time()
    fi = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1

        # ── ArUco 자동 캘리브레이션 (30프레임마다 갱신) ──────
        if aruco_mode and fi - _aruco_last_update >= 30:
            new_calib = calibrate_from_aruco(frame)
            if new_calib is not None:
                calib = new_calib
                court_tmpl = render_court_birdeye(calib, px_per_m=px_per_m)
                _aruco_last_update = fi
                if fi <= 31:
                    print(f"[PoseDemo] ArUco 캘리브레이션 완료 (frame {fi})")

        # ── 렌즈 왜곡 보정 (intrinsics 로드된 경우) ──────────
        if undistort_maps is not None:
            frame = undistort_frame(frame, *undistort_maps)

        # ── 감지 + 추적 ──────────────────────────────────────
        dets   = detector.detect(frame)
        tracks = tracker.update(dets)

        # ── 발 위치 → 코트 좌표 변환 ─────────────────────────
        foot_pts = np.array([[t.foot_point[0], t.foot_point[1]]
                              for t in tracks], dtype=np.float32) if tracks else np.empty((0, 2))
        if len(foot_pts) > 0:
            court_pts = pixel_to_court(foot_pts, calib)    # (N, 2) meter
        else:
            court_pts = np.empty((0, 2))

        # ── 팀 배정 & PlayerState 구성 ───────────────────────
        player_states: list[PlayerState] = []
        for i, t in enumerate(tracks):
            _assign_team(t.track_id, t.foot_point[0], frame_w)
            if i < len(court_pts):
                cx, cy = float(court_pts[i][0]), float(court_pts[i][1])
                player_states.append(PlayerState(
                    track_id=t.track_id,
                    court_x=max(0.1, min(9.9, cx)),
                    court_y=max(0.1, min(5.9, cy)),
                ))

        advices = tactic_engine.analyze(player_states) if player_states else []

        # ── 역할·상황 추정 → k-NN 이동 예측 ─────────────────────────
        roles    = _infer_roles(tracks, court_pts)
        contexts = _infer_contexts(tracks, court_pts)
        mv_preds: dict[int, tuple[MovementPrediction | None, str]] = {}
        for i, t in enumerate(tracks):
            if i >= len(court_pts):
                continue
            team = _team_cache.get(t.track_id, "A")
            pos  = (float(court_pts[i][0]), float(court_pts[i][1]))
            role = roles.get(t.track_id, "main_attacker")
            ctx  = contexts.get(team, "attack")
            pred = movement_model.predict(pos, role, ctx, team)
            mv_preds[t.track_id] = (pred, team)

        # ── 자세 분석 ─────────────────────────────────────────
        # track_id → Detection 매핑 (IoU로 연결 안 됐으면 가장 가까운 det)
        det_map: dict[int, Detection] = {}
        for t in tracks:
            # tracker history의 최신 bbox를 dets에서 찾아 매핑
            best = min(
                dets,
                key=lambda d: abs(d.foot_point[0] - t.foot_point[0])
                              + abs(d.foot_point[1] - t.foot_point[1]),
                default=None,
            )
            if best:
                det_map[t.track_id] = best

        # ── 카메라 뷰 렌더링 ──────────────────────────────────
        cam_view = frame.copy()
        for t in tracks:
            det = det_map.get(t.track_id)
            if det is None:
                continue

            team  = _team_cache.get(t.track_id, "A")
            color = (100, 200, 100) if team == "A" else (100, 150, 255)

            # 바운딩 박스
            cv2.rectangle(cam_view,
                          (int(det.x1), int(det.y1)),
                          (int(det.x2), int(det.y2)),
                          color, 2)
            # 선수 번호
            cv2.putText(cam_view, f"#{t.track_id}",
                        (int(det.x1) + 2, int(det.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 스켈레톤
            draw_skeleton(cam_view, det, base_color=color)

            # 자세 라벨
            feat = analyze_pose(det)
            if feat:
                draw_posture_label(cam_view, det, feat)
                posture_counts[feat.posture] = posture_counts.get(feat.posture, 0) + 1

        # ArUco 마커 오버레이
        if aruco_mode:
            detected = detect_markers(cam_view)
            cam_view = draw_aruco_overlay(cam_view, detected, calib)

        # 이동 예측 화살표
        _draw_prediction_overlay(cam_view, tracks)

        # ── Bird-eye view 렌더링 ───────────────────────────────
        birdeye = court_tmpl.copy()

        # 선수 점 + ID
        for i, t in enumerate(tracks):
            if i >= len(court_pts):
                continue
            cx = int(court_pts[i][0] * px_per_m)
            cy = int(court_pts[i][1] * px_per_m)
            team  = _team_cache.get(t.track_id, "A")
            color = (100, 200, 100) if team == "A" else (100, 150, 255)
            cv2.circle(birdeye, (cx, cy), 8, color, -1, cv2.LINE_AA)
            cv2.putText(birdeye, str(t.track_id), (cx - 4, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 자세별 테두리
            det = det_map.get(t.track_id)
            if det:
                feat = analyze_pose(det)
                if feat and feat.posture != "neutral":
                    pc = POSTURE_COLOR[feat.posture]
                    cv2.circle(birdeye, (cx, cy), 13, pc, 2, cv2.LINE_AA)

        birdeye = draw_guide_on_birdeye(birdeye, advices, px_per_m=px_per_m)
        _draw_model_predictions(birdeye, mv_preds, px_per_m)

        # ── 자세 통계 & FPS ───────────────────────────────────
        _draw_posture_stats(cam_view, posture_counts)
        elapsed = time.time() - t0
        fps_now = fi / max(0.01, elapsed)
        draw_hud(cam_view, fps=fps_now, n_players=len(tracks))

        # ── 좌우 결합 ─────────────────────────────────────────
        bh, bw = birdeye.shape[:2]
        ch, cw = cam_view.shape[:2]
        # 카메라 뷰를 birdeye 높이에 맞게 리사이즈
        scale = bh / ch
        cam_rs = cv2.resize(cam_view, (int(cw * scale), bh))
        combined = np.hstack([cam_rs, birdeye])

        if writer is None and out_path:
            out_path.parent.mkdir(exist_ok=True)
            h_out, w_out = combined.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                min(src_fps, 30.0), (w_out, h_out),
            )
            print(f"[PoseDemo] 저장: {out_path}  ({w_out}×{h_out})")

        if writer:
            writer.write(combined)

        if not args.headless:
            cv2.imshow("HADO Pose Demo", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if fi % 60 == 0:
            print(f"[PoseDemo] {fi}/{total}  FPS={fps_now:.1f}  "
                  f"자세: {posture_counts}")

    cap.release()
    if writer:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()

    print(f"\n[PoseDemo] 완료 — {fi}프레임  자세통계: {posture_counts}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="HADO YOLOv8-pose 파이프라인 데모")
    parser.add_argument("--video",    required=True,      help="입력 영상 경로")
    parser.add_argument("--model",    default="yolov8n-pose.pt", help="YOLO 모델 경로")
    parser.add_argument("--out",      default="",         help="출력 mp4 경로 (미지정 시 저장 안 함)")
    parser.add_argument("--imgsz",    type=int,   default=320)
    parser.add_argument("--conf",     type=float, default=0.35)
    parser.add_argument("--aruco",     action="store_true",
                        help="ArUco 마커 자동 캘리브레이션")
    parser.add_argument("--intrinsic", action="store_true",
                        help="렌즈 왜곡 보정 적용 (config/cam{id}_intrinsics.json 필요)")
    parser.add_argument("--cam-id",    type=int, default=0,
                        help="카메라 ID (intrinsics 파일 선택용, 기본값=0)")
    parser.add_argument("--headless",  action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
