"""조끼 색상 기반 선수 포지션 감지.

상체 영역의 HSV 색상을 분석해 포지션을 판별한다.
  빨간색 → main_attacker
  파란색 → technician
  하얀색 → defender

실행:
    python -m src.vest --source 0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.detector import Detection


# ── HSV 색상 범위 ────────────────────────────────────────────────
# (lower, upper) — cv2.inRange 기준 (H: 0-179, S: 0-255, V: 0-255)
_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "main_attacker": [   # 빨간색 (Hue가 0~10 또는 170~179으로 두 구간)
        (np.array([0,   120, 80],  np.uint8), np.array([10,  255, 255], np.uint8)),
        (np.array([170, 120, 80],  np.uint8), np.array([179, 255, 255], np.uint8)),
    ],
    "technician": [      # 파란색
        (np.array([100, 80, 60],   np.uint8), np.array([130, 255, 255], np.uint8)),
    ],
    "defender": [        # 하얀색 (채도 낮고 명도 높음)
        (np.array([0,   0,  180],  np.uint8), np.array([179, 50,  255], np.uint8)),
    ],
}

# 포지션별 시각화 색상 (BGR)
VEST_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "main_attacker": (0,   60,  220),   # 빨강
    "technician":    (220, 120,  0  ),  # 파랑
    "defender":      (220, 220, 220),   # 흰색
    "unknown":       (100, 100, 100),   # 회색
}

ROLE_KO: dict[str, str] = {
    "main_attacker": "어태커",
    "technician":    "테크니션",
    "defender":      "디펜더",
    "unknown":       "미확인",
}

# 상체 영역 비율 (bbox 기준)
_TORSO_TOP    = 0.20   # 머리 제외
_TORSO_BOTTOM = 0.65   # 하체 제외
_TORSO_LEFT   = 0.15   # 팔 제외
_TORSO_RIGHT  = 0.85


@dataclass
class VestResult:
    role:       str    # "main_attacker" | "technician" | "defender" | "unknown"
    confidence: float  # 0.0~1.0
    pixel_ratio: float # 감지 색상 픽셀 비율 (디버깅용)


def detect_vest(
    frame: np.ndarray,
    det: Detection,
    min_confidence: float = 0.10,
) -> VestResult:
    """바운딩 박스 상체 영역에서 조끼 색상으로 포지션 판별.

    Parameters
    ----------
    frame          : BGR 카메라 프레임
    det            : 선수 Detection 객체
    min_confidence : 이 비율 미만이면 unknown 반환

    Returns
    -------
    VestResult
    """
    h_frame, w_frame = frame.shape[:2]

    # 상체 영역 좌표 계산
    bw = det.x2 - det.x1
    bh = det.y2 - det.y1
    x1 = int(max(0, det.x1 + bw * _TORSO_LEFT))
    x2 = int(min(w_frame, det.x1 + bw * _TORSO_RIGHT))
    y1 = int(max(0, det.y1 + bh * _TORSO_TOP))
    y2 = int(min(h_frame, det.y1 + bh * _TORSO_BOTTOM))

    if x2 <= x1 or y2 <= y1:
        return VestResult("unknown", 0.0, 0.0)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return VestResult("unknown", 0.0, 0.0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_px = roi.shape[0] * roi.shape[1]

    # 각 포지션별 픽셀 수 집계
    scores: dict[str, float] = {}
    for role, ranges in _RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        scores[role] = float(mask.sum() / 255) / total_px

    best_role = max(scores, key=lambda r: scores[r])
    best_score = scores[best_role]

    if best_score < min_confidence:
        return VestResult("unknown", best_score, best_score)

    # 신뢰도: 1위 점수 / 전체 합산 (다른 색과 얼마나 명확히 구분되는지)
    total_score = sum(scores.values()) or 1e-6
    confidence = round(best_score / total_score, 3)

    return VestResult(best_role, confidence, round(best_score, 3))


def draw_vest_label(
    frame: np.ndarray,
    det: Detection,
    result: VestResult,
) -> None:
    """바운딩 박스 위에 포지션 라벨 + 신뢰도 표시 (in-place)."""
    if result.role == "unknown":
        return
    color = VEST_COLOR_BGR.get(result.role, (150, 150, 150))
    label = f"{ROLE_KO[result.role]} {result.confidence:.0%}"
    cv2.putText(
        frame, label,
        (int(det.x1), max(0, int(det.y1) - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )

    # 상체 ROI 표시 (디버깅 시 시각 확인용)
    bw, bh = det.x2 - det.x1, det.y2 - det.y1
    rx1 = int(det.x1 + bw * _TORSO_LEFT)
    rx2 = int(det.x1 + bw * _TORSO_RIGHT)
    ry1 = int(det.y1 + bh * _TORSO_TOP)
    ry2 = int(det.y1 + bh * _TORSO_BOTTOM)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 1)


# ── 단독 테스트 ────────────────────────────────────────────────────
def main() -> None:
    """카메라/영상에서 실시간 조끼 색상 감지 테스트."""
    import argparse
    from src.detector import PersonDetector

    parser = argparse.ArgumentParser(description="조끼 색상 포지션 감지 테스트")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--imgsz",  type=int, default=320)
    args = parser.parse_args()

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    detector = PersonDetector(model_path=args.model, imgsz=args.imgsz)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Vest] 소스 열기 실패: {src}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print("[Vest] 실행 중 — ESC 종료")
    print("  빨강=어태커  파랑=테크니션  흰색=디펜더")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        for det in dets:
            result = detect_vest(frame, det)
            color = VEST_COLOR_BGR.get(result.role, (150, 150, 150))
            cv2.rectangle(frame,
                          (int(det.x1), int(det.y1)),
                          (int(det.x2), int(det.y2)), color, 2)
            draw_vest_label(frame, det, result)

        cv2.imshow("Vest Detection (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
