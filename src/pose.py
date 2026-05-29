"""YOLOv8-pose 키포인트 기반 자세 분석 모듈.

일반 카메라로 촬영된 선수 영상에서 자세(크라우치·쉴드준비·중립)를 분류하고
다음 동작을 보조 예측한다.

COCO 17 키포인트 (인덱스 기준)
-------------------------------
0:코   1:왼눈  2:오른눈  3:왼귀  4:오른귀
5:왼어깨  6:오른어깨  7:왼팔꿈치  8:오른팔꿈치
9:왼손목  10:오른손목  11:왼엉덩이  12:오른엉덩이
13:왼무릎  14:오른무릎  15:왼발목  16:오른발목
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.detector import Detection

# ── 키포인트 인덱스 ────────────────────────────────────────────
_N  = 0                           # 코
_LS, _RS = 5, 6                   # 어깨
_LE, _RE = 7, 8                   # 팔꿈치
_LW, _RW = 9, 10                  # 손목
_LH, _RH = 11, 12                 # 엉덩이
_LK, _RK = 13, 14                 # 무릎
_LA, _RA = 15, 16                 # 발목

# ── 스켈레톤 연결 ──────────────────────────────────────────────
SKELETON_PAIRS: list[tuple[int, int]] = [
    (_LS, _RS),                    # 어깨
    (_LS, _LE), (_LE, _LW),        # 왼팔
    (_RS, _RE), (_RE, _RW),        # 오른팔
    (_LS, _LH), (_RS, _RH),        # 상체 옆면
    (_LH, _RH),                    # 엉덩이
    (_LH, _LK), (_LK, _LA),        # 왼다리
    (_RH, _RK), (_RK, _RA),        # 오른다리
    (_N,  _LS), (_N,  _RS),        # 목 (코→어깨 근사)
]

# 몸통(흰색) / 팔(파랑) / 다리(초록) 색상 그룹
_LIMB_COLORS: dict[tuple[int, int], tuple[int, int, int]] = {
    (_LS, _RS):   (220, 220, 220),
    (_LS, _LH):   (220, 220, 220),
    (_RS, _RH):   (220, 220, 220),
    (_LH, _RH):   (220, 220, 220),
    (_N,  _LS):   (220, 220, 220),
    (_N,  _RS):   (220, 220, 220),
    (_LS, _LE):   (255, 140, 60),
    (_LE, _LW):   (255, 140, 60),
    (_RS, _RE):   (60,  140, 255),
    (_RE, _RW):   (60,  140, 255),
    (_LH, _LK):   (80,  220, 100),
    (_LK, _LA):   (80,  220, 100),
    (_RH, _RK):   (80,  200, 255),
    (_RK, _RA):   (80,  200, 255),
}

# ── 자세 레이블 ────────────────────────────────────────────────
POSTURE_KO = {
    "attack":  "공격 자세",
    "shield":  "쉴드 준비",
    "neutral": "중립",
}

POSTURE_COLOR: dict[str, tuple[int, int, int]] = {
    "attack":  (40,  160, 255),   # 주황
    "shield":  (60,  180, 100),   # 초록
    "neutral": (160, 160, 160),   # 회색
}

# 자세 → 예상 의도 매핑 (TacticEngine 보조 신호)
POSTURE_TO_INTENT = {
    "attack":  "direct_attack",
    "shield":  "shield_protect",
    "neutral": "",
}


# ── 데이터클래스 ───────────────────────────────────────────────
@dataclass
class PoseFeatures:
    """단일 선수의 자세 특징값."""
    keypoints:    np.ndarray       # (17, 3) — x, y, conf
    crouch_score: float            # 0.0~1.0 (높을수록 낮은 자세)
    arm_spread:   float            # 0.0~1.0 (팔을 크게 벌린 정도)
    posture:      str              # "attack" | "shield" | "neutral"
    intent_hint:  str = field(default="")  # POSTURE_TO_INTENT 매핑값


# ── 자세 분석 ──────────────────────────────────────────────────
_KP_CONF_MIN = 0.25   # 이 신뢰도 미만 키포인트는 무시


def _kp(kpts: np.ndarray, idx: int) -> Optional[tuple[float, float]]:
    """신뢰도 통과한 키포인트 좌표 반환."""
    if kpts[idx, 2] >= _KP_CONF_MIN:
        return float(kpts[idx, 0]), float(kpts[idx, 1])
    return None


def analyze_pose(det: Detection) -> Optional[PoseFeatures]:
    """Detection에서 자세 특징 추출. keypoints 없으면 None 반환."""
    if det.keypoints is None:
        return None

    kpts = det.keypoints           # (17, 3)
    bbox_h = max(1.0, det.y2 - det.y1)
    bbox_w = max(1.0, det.x2 - det.x1)

    # ── 크라우치 점수 ──────────────────────────────────────────
    # 어깨 중점 y ~ 발목 중점 y 의 수직 간격 / bbox 높이
    # 직립: ~0.65×bbox_h  /  크라우치: ~0.35×bbox_h
    sh_l, sh_r = _kp(kpts, _LS), _kp(kpts, _RS)
    an_l, an_r = _kp(kpts, _LA), _kp(kpts, _RA)

    crouch_score = 0.0
    if sh_l and sh_r and an_l and an_r:
        sh_y  = (sh_l[1] + sh_r[1]) / 2
        an_y  = (an_l[1] + an_r[1]) / 2
        span  = an_y - sh_y            # 양수 (발목이 아래)
        crouch_score = float(max(0.0, min(1.0, 1.0 - span / (0.60 * bbox_h))))
    elif sh_l and sh_r:
        # 발목 미감지: 무릎으로 대체
        kn_l, kn_r = _kp(kpts, _LK), _kp(kpts, _RK)
        if kn_l and kn_r:
            sh_y = (sh_l[1] + sh_r[1]) / 2
            kn_y = (kn_l[1] + kn_r[1]) / 2
            crouch_score = float(max(0.0, min(1.0, 1.0 - (kn_y - sh_y) / (0.40 * bbox_h))))

    # ── 팔 벌림 (쉴드 준비 지표) ───────────────────────────────
    # 손목과 어깨 사이 수평 거리 / bbox 폭
    arm_spread = 0.0
    lw = _kp(kpts, _LW)
    rw = _kp(kpts, _RW)
    if lw and sh_l:
        arm_spread = max(arm_spread, abs(lw[0] - sh_l[0]) / bbox_w)
    if rw and sh_r:
        arm_spread = max(arm_spread, abs(rw[0] - sh_r[0]) / bbox_w)

    # ── 자세 분류 ──────────────────────────────────────────────
    if arm_spread > 0.50 and crouch_score < 0.40:
        posture = "shield"
    elif crouch_score > 0.45:
        posture = "attack"
    else:
        posture = "neutral"

    return PoseFeatures(
        keypoints=kpts,
        crouch_score=round(crouch_score, 3),
        arm_spread=round(arm_spread, 3),
        posture=posture,
        intent_hint=POSTURE_TO_INTENT.get(posture, ""),
    )


# ── 시각화 ─────────────────────────────────────────────────────
def draw_skeleton(
    img: np.ndarray,
    det: Detection,
    base_color: tuple[int, int, int] = (200, 200, 200),
    use_limb_colors: bool = True,
) -> None:
    """카메라 뷰에 스켈레톤 오버레이 (in-place).

    Parameters
    ----------
    base_color      : use_limb_colors=False 일 때 단색
    use_limb_colors : True면 팔·다리별 색상 구분
    """
    if det.keypoints is None:
        return

    kpts = det.keypoints

    # 관절 연결선
    for a, b in SKELETON_PAIRS:
        pa, pb = _kp(kpts, a), _kp(kpts, b)
        if pa and pb:
            color = (_LIMB_COLORS.get((a, b)) or base_color) if use_limb_colors else base_color
            cv2.line(img,
                     (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])),
                     color, 2, cv2.LINE_AA)

    # 관절 점
    for i in range(17):
        p = _kp(kpts, i)
        if p:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_posture_label(
    img: np.ndarray,
    det: Detection,
    features: PoseFeatures,
) -> None:
    """바운딩 박스 상단에 자세 분류 라벨 표시 (한글 PIL)."""
    try:
        from src.annotate import put_text_kr
        color = POSTURE_COLOR.get(features.posture, (200, 200, 200))
        label = POSTURE_KO.get(features.posture, features.posture)
        score_txt = f"{label} ({features.crouch_score:.2f})"
        put_text_kr(img, score_txt, (int(det.x1), max(0, int(det.y1) - 18)), 13, color)
    except Exception:
        pass


def draw_movement_arrow(
    img: np.ndarray,
    track_history: list[tuple[float, float]],
    color: tuple[int, int, int] = (0, 220, 255),
    predict_frames: int = 12,
) -> None:
    """트랙 히스토리로 이동 벡터 → 예측 위치 화살표 표시."""
    if len(track_history) < 4:
        return
    # 최근 4 프레임 평균 속도
    recent = track_history[-4:]
    vx = (recent[-1][0] - recent[0][0]) / 3
    vy = (recent[-1][1] - recent[0][1]) / 3
    if abs(vx) < 1.0 and abs(vy) < 1.0:
        return   # 거의 정지

    cx, cy = int(recent[-1][0]), int(recent[-1][1])
    px = int(cx + vx * predict_frames)
    py = int(cy + vy * predict_frames)
    cv2.arrowedLine(img, (cx, cy), (px, py), color,
                    2, tipLength=0.35, line_type=cv2.LINE_AA)


# ── 단독 테스트 ────────────────────────────────────────────────
def main() -> None:
    """yolov8n-pose로 단일 이미지/영상 자세 분석."""
    import argparse
    from src.detector import PersonDetector

    parser = argparse.ArgumentParser(description="Pose 분석 테스트")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model",  default="yolov8n-pose.pt")
    parser.add_argument("--imgsz",  type=int, default=320)
    args = parser.parse_args()

    detector = PersonDetector(model_path=args.model, imgsz=args.imgsz)

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Pose] 소스 열기 실패: {src}")
        return

    print("[Pose] 실행 중 — ESC로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        for det in dets:
            cv2.rectangle(frame,
                          (int(det.x1), int(det.y1)),
                          (int(det.x2), int(det.y2)),
                          (80, 80, 80), 1)
            draw_skeleton(frame, det)
            feat = analyze_pose(det)
            if feat:
                draw_posture_label(frame, det, feat)

        cv2.imshow("Pose Analysis (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
