"""Bird-eye view 코트 시각화.

- 카메라 원본 프레임에 바운딩 박스/ID/발 위치 오버레이
- 별도 패널에 코트 탑뷰 + 선수 위치 점 + 궤적
- 두 패널을 가로로 합쳐 단일 창에 표시
"""
from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np

from src.homography import Calibration, court_to_pixel, pixel_to_court
from src.tracker import Track


# 선수 ID별 고정 색상 (BGR)
PLAYER_COLORS = [
    (0, 100, 255),    # 주황
    (255, 100, 0),    # 파랑
    (0, 200, 100),    # 녹색
    (200, 0, 255),    # 자주
    (0, 255, 255),    # 노랑
    (255, 0, 100),    # 분홍
]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return PLAYER_COLORS[(track_id - 1) % len(PLAYER_COLORS)]


def render_court_birdeye(
    calib: Calibration,
    px_per_m: int = 100,
    bg_color: tuple[int, int, int] = (40, 50, 35),
    line_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """빈 코트 bird-eye view 이미지를 만든다.

    (court_width_m * px_per_m, court_height_m * px_per_m) 크기.
    """
    w = int(calib.court_width_m * px_per_m)
    h = int(calib.court_height_m * px_per_m)
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # 코트 외곽 라인
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), line_color, 2)

    # 중앙선
    cv2.line(img, (w // 2, 0), (w // 2, h), (180, 180, 60), 1, cv2.LINE_AA)

    # 1m 그리드 (옅게)
    for x_m in range(1, int(calib.court_width_m)):
        x_px = int(x_m * px_per_m)
        cv2.line(img, (x_px, 0), (x_px, h), (60, 70, 55), 1, cv2.LINE_AA)
    for y_m in range(1, int(calib.court_height_m)):
        y_px = int(y_m * px_per_m)
        cv2.line(img, (0, y_px), (w, y_px), (60, 70, 55), 1, cv2.LINE_AA)

    # HADO 전술 구역선 (x축): 팀A 3선(1.5m)/2선(3.0m), 팀B 미러(7.0m/8.5m)
    zone_x_color = (70, 160, 70)
    for x_zone in [1.5, 3.0, 7.0, 8.5]:
        xp = int(x_zone * px_per_m)
        if 0 < xp < w:
            cv2.line(img, (xp, 0), (xp, h), zone_x_color, 1, cv2.LINE_AA)

    # HADO 레인 경계선 (y축): 3개 레인 분리 (2.0m / 4.0m)
    lane_y_color = (55, 130, 55)
    for y_lane in [2.0, 4.0]:
        yp = int(y_lane * px_per_m)
        if 0 < yp < h:
            cv2.line(img, (0, yp), (w, yp), lane_y_color, 1, cv2.LINE_AA)

    # 모서리 좌표 라벨
    cv2.putText(img, "(0,0)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)
    cv2.putText(img, f"({calib.court_width_m:.1f},{calib.court_height_m:.2f})",
                (w - 90, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)

    return img


def draw_detections_on_frame(
    frame: np.ndarray,
    tracks: List[Track],
    show_id: bool = True,
    show_foot: bool = True,
) -> np.ndarray:
    """원본 카메라 프레임에 트랙 정보 오버레이 (in-place)."""
    for t in tracks:
        color = color_for_id(t.track_id)
        x1, y1, x2, y2 = t.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show_id:
            label = f"#{t.track_id}  {t.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if show_foot:
            fx, fy = t.foot_point
            cv2.circle(frame, (int(fx), int(fy)), 6, color, -1)
            cv2.circle(frame, (int(fx), int(fy)), 7, (255, 255, 255), 1)

    return frame


def draw_players_on_birdeye(
    court_img: np.ndarray,
    tracks: List[Track],
    calib: Calibration,
    px_per_m: int = 100,
    radius: int = 12,
    show_trajectory: bool = True,
    trajectory_length: int = 30,
) -> np.ndarray:
    """Bird-eye view 위에 트랙 선수 위치 점 + 궤적을 그린다."""
    out = court_img.copy()

    for t in tracks:
        color = color_for_id(t.track_id)

        # 궤적
        if show_trajectory and len(t.history) >= 2:
            recent = t.history[-trajectory_length:]
            recent_m = pixel_to_court(np.array(recent, dtype=np.float32), calib)
            for i in range(1, len(recent_m)):
                p1 = (int(recent_m[i-1, 0] * px_per_m), int(recent_m[i-1, 1] * px_per_m))
                p2 = (int(recent_m[i, 0] * px_per_m), int(recent_m[i, 1] * px_per_m))
                alpha = i / len(recent_m)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(out, p1, p2, faded, 2, cv2.LINE_AA)

        # 현재 위치
        foot_px = np.array([t.foot_point], dtype=np.float32)
        foot_m = pixel_to_court(foot_px, calib)[0]
        cx, cy = int(foot_m[0] * px_per_m), int(foot_m[1] * px_per_m)

        # 코트 밖이면 그리지 않음
        h, w = out.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        cv2.circle(out, (cx, cy), radius + 2, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), radius, color, -1)
        cv2.putText(out, str(t.track_id),
                    (cx - 6 if t.track_id < 10 else cx - 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def combine_views(camera_view: np.ndarray, court_view: np.ndarray) -> np.ndarray:
    """카메라 뷰 + bird-eye 뷰를 한 창에 가로로 합친다.

    카메라 뷰는 그대로, bird-eye view는 카메라 뷰의 높이에 맞춰 리사이즈.
    """
    h_cam = camera_view.shape[0]
    h_court, w_court = court_view.shape[:2]
    scale = h_cam / h_court
    new_w = int(w_court * scale)
    resized_court = cv2.resize(court_view, (new_w, h_cam), interpolation=cv2.INTER_LINEAR)

    # 구분선
    separator = np.full((h_cam, 4, 3), 80, dtype=np.uint8)
    combined = np.hstack([camera_view, separator, resized_court])
    return combined


def draw_hud(frame: np.ndarray, fps: float, n_players: int, recording: bool = False) -> np.ndarray:
    """프레임 상단에 HUD 표시 (FPS, 선수 수, 녹화 상태)."""
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

    info = f"FPS: {fps:5.1f}  |  Players: {n_players}"
    if recording:
        info += "  |  ● REC"
    cv2.putText(frame, info, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)

    keys = "ESC: exit | s: snap | r: rec | c: clear"
    (tw, _), _ = cv2.getTextSize(keys, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, keys, (w - tw - 12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def main():
    """단독 테스트: 가짜 캘리브레이션으로 빈 코트 + 더미 선수 렌더링."""
    from src.homography import compute_homography

    calib = compute_homography(
        corners_pixel=np.array([[100, 100], [1180, 100], [1180, 600], [100, 600]], dtype=np.float32),
        court_width_m=10.0,
        court_height_m=6.0,
        image_size=(1280, 720),
    )

    court = render_court_birdeye(calib, px_per_m=80)
    print(f"코트 이미지 크기: {court.shape}")

    # 더미 트랙 2명
    fake_tracks = [
        Track(track_id=1, bbox=np.array([400, 200, 480, 480], dtype=np.float32),
              confidence=0.92, age=10,
              history=[(440, 480), (445, 475), (450, 470), (455, 480)]),
        Track(track_id=2, bbox=np.array([800, 250, 880, 500], dtype=np.float32),
              confidence=0.88, age=10,
              history=[(840, 500), (835, 495), (830, 490)]),
    ]

    rendered = draw_players_on_birdeye(court, fake_tracks, calib, px_per_m=100)
    out_path = "/tmp/birdeye_test.png"
    cv2.imwrite(out_path, rendered)
    print(f"테스트 이미지 저장: {out_path}")


if __name__ == "__main__":
    main()
