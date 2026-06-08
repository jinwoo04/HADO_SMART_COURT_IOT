"""src/pose.py — classify_hado_action() + sample_vest_hue() 단위 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detection
from src.pose import (
    ACTION_KO, ActionResult,
    classify_hado_action, sample_vest_hue,
)

_W, _H = 100.0, 200.0   # 가상 바운딩 박스 (왼쪽 팀 → frame_center_x=320 기준 팀A)


def _det(kpts_17x3: list[tuple[float, float, float]],
         x1: float = 0.0, x2: float = _W) -> Detection:
    arr = np.array(kpts_17x3, dtype=np.float32)
    return Detection(x1=x1, y1=0.0, x2=x2, y2=_H,
                     confidence=0.9, keypoints=arr)


def _zero_kpts() -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 0.0)] * 17


def _kpt(x: float, y: float, conf: float = 0.9) -> tuple[float, float, float]:
    return (x, y, conf)


# 키포인트 인덱스
_LS, _RS = 5, 6
_LE, _RE = 7, 8
_LW, _RW = 9, 10
_LH, _RH = 11, 12
_LK, _RK = 13, 14


def _make_kpts(patches: "dict[int, tuple[float, float]]") -> list[tuple[float, float, float]]:
    """기본값 (신뢰도=0) + 지정 키포인트만 활성화."""
    pts = _zero_kpts()
    for idx, (x, y) in patches.items():
        pts[idx] = _kpt(x, y)
    return pts


class TestClassifyHadoAction:

    def test_no_keypoints_returns_none(self):
        det = Detection(x1=0, y1=0, x2=_W, y2=_H, confidence=0.9, keypoints=None)
        assert classify_hado_action(det) is None

    def test_returns_action_result(self):
        kpts = _make_kpts({
            _LS: (40.0, 60.0), _RS: (60.0, 60.0),
            _LH: (45.0, 130.0), _RH: (55.0, 130.0),
        })
        result = classify_hado_action(_det(kpts))
        assert isinstance(result, ActionResult)
        assert result.action in ACTION_KO
        assert 0.0 <= result.confidence <= 1.0
        assert "shoot" in result.scores
        assert "charge" in result.scores

    def test_charge_arm_raised_vertical(self):
        """손목이 어깨보다 scale×0.55 이상 위, 수직 방향 → charge 판정."""
        # scale = max(shoulder_w=20, torso_h*0.6=30, 30) = 30
        # dy = (20 - 80)/30 = -2.0 < -0.55, |dx| = 10/30 = 0.33 < 0.70 → charge
        kpts = _make_kpts({
            _LS: (40.0, 80.0), _RS: (60.0, 80.0),
            _LW: (30.0, 20.0),   # 수직에 가깝게 들어올림
            _RW: (70.0, 80.0),
            _LH: (45.0, 130.0), _RH: (55.0, 130.0),
            _LK: (45.0, 155.0), _RK: (55.0, 155.0),
        })
        result = classify_hado_action(_det(kpts))
        assert result is not None
        assert result.action == "charge", f"expected charge, got {result.action} (scores={result.scores})"

    def test_shoot_arm_forward_team_a(self):
        """팀A(왼쪽) 기준 손목이 오른쪽(앞)으로 뻗임 → shoot 판정."""
        # scale = max(20, 30, 30) = 30
        # ls=(40,80), rw=(120,80) → dx=(120-60)/30=2.0, dy=0, forward=2.0*1=2.0 > 0.65 → shoot
        kpts = _make_kpts({
            _LS: (40.0, 80.0), _RS: (60.0, 80.0),
            _LW: (40.0, 80.0), _RW: (120.0, 80.0),   # 오른손 앞으로 뻗음
            _LH: (45.0, 130.0), _RH: (55.0, 130.0),
            _LK: (45.0, 155.0), _RK: (55.0, 155.0),
        })
        # 팀A: center_x=50 < frame_center_x=320 → facing=+1(오른쪽이 전방)
        result = classify_hado_action(_det(kpts), frame_center_x=320.0)
        assert result is not None
        assert result.action == "shoot", f"expected shoot, got {result.action} (scores={result.scores})"

    def test_shoot_not_fired_backward(self):
        """팀A 기준 손목이 뒤(왼쪽)로 향하면 shoot 미발화."""
        kpts = _make_kpts({
            _LS: (40.0, 80.0), _RS: (60.0, 80.0),
            _LW: (-20.0, 80.0), _RW: (10.0, 80.0),   # 뒤쪽
            _LH: (45.0, 130.0), _RH: (55.0, 130.0),
        })
        result = classify_hado_action(_det(kpts), frame_center_x=320.0)
        assert result is not None
        assert result.action != "shoot"

    def test_crouch_compressed_pose(self):
        """무릎이 어깨와 가까이 있음 → crouch 판정."""
        kpts = _make_kpts({
            _LS: (40.0, 80.0), _RS: (60.0, 80.0),
            _LK: (42.0, 110.0), _RK: (58.0, 110.0),
            _LH: (45.0, 100.0), _RH: (55.0, 100.0),
        })
        result = classify_hado_action(_det(kpts))
        assert result is not None
        assert result.action == "crouch", f"expected crouch, got {result.action} (scores={result.scores})"

    def test_dodge_left(self):
        """어깨 중점이 엉덩이 중점 왼쪽으로 치우침 → dodge_l."""
        kpts = _make_kpts({
            _LS: (10.0, 70.0), _RS: (30.0, 70.0),   # sh_cx = 20
            _LH: (45.0, 130.0), _RH: (55.0, 130.0),  # hi_cx = 50
        })
        result = classify_hado_action(_det(kpts))
        assert result is not None
        assert result.action == "dodge_l", f"expected dodge_l, got {result.action} (scores={result.scores})"

    def test_dodge_right(self):
        """어깨 중점이 엉덩이 중점 오른쪽으로 치우침 → dodge_r."""
        kpts = _make_kpts({
            _LS: (70.0, 70.0), _RS: (90.0, 70.0),   # sh_cx = 80
            _LH: (45.0, 130.0), _RH: (55.0, 130.0),  # hi_cx = 50
        })
        result = classify_hado_action(_det(kpts))
        assert result is not None
        assert result.action == "dodge_r", f"expected dodge_r, got {result.action} (scores={result.scores})"

    def test_ready_upright(self):
        """직립 자세, 팔 어깨 아래 → ready 판정."""
        kpts = _make_kpts({
            _LS: (40.0, 60.0), _RS: (60.0, 60.0),
            _LW: (35.0, 80.0), _RW: (65.0, 80.0),   # 손목 어깨 아래
            _LE: (37.0, 70.0), _RE: (63.0, 70.0),
            _LH: (42.0, 120.0), _RH: (58.0, 120.0),
            _LK: (42.0, 155.0), _RK: (58.0, 155.0),
        })
        result = classify_hado_action(_det(kpts))
        assert result is not None
        assert result.action == "ready", f"expected ready, got {result.action}"

    def test_low_conf_keypoints_ignored(self):
        """신뢰도 낮은 키포인트는 무시 → ready로 폴백."""
        pts = [(50.0, 10.0, 0.1)] * 17
        result = classify_hado_action(_det(pts))
        assert result is not None
        assert result.action == "ready"

    def test_crouch_priority_over_charge(self):
        """crouch와 charge 동시 성립 → crouch 우선."""
        # 무릎이 어깨 바로 아래(crouch) + 팔도 올라있음(charge 가능)
        kpts = _make_kpts({
            _LS: (40.0, 80.0), _RS: (60.0, 80.0),
            _LW: (35.0, 20.0),                       # 팔 위로
            _LK: (42.0, 110.0), _RK: (58.0, 110.0),
            _LH: (45.0, 100.0), _RH: (55.0, 100.0),
        })
        result = classify_hado_action(_det(kpts))
        assert result is not None
        assert result.action == "crouch", f"우선순위 실패: {result.action}"


class TestSampleVestHue:

    def _make_frame_with_vest(
        self, det: Detection, hue_bgr: tuple[int, int, int]
    ) -> np.ndarray:
        """토르소 영역을 특정 색으로 채운 640×480 더미 프레임."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if det.keypoints is not None:
            kpts = det.keypoints
            ls = (kpts[_LS, 0], kpts[_LS, 1])
            rs = (kpts[_RS, 0], kpts[_RS, 1])
            lh = (kpts[_LH, 0], kpts[_LH, 1])
            rh = (kpts[_RH, 0], kpts[_RH, 1])
            x1 = int(min(ls[0], rs[0], lh[0], rh[0]))
            y1 = int(min(ls[1], rs[1]))
            x2 = int(max(ls[0], rs[0], lh[0], rh[0]))
            y2 = int(max(lh[1], rh[1]))
            frame[y1:y2, x1:x2] = hue_bgr
        return frame

    def test_no_keypoints_returns_minus1(self):
        det = Detection(x1=0, y1=0, x2=100, y2=200, confidence=0.9, keypoints=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert sample_vest_hue(frame, det) == -1

    def test_low_conf_keypoints_returns_minus1(self):
        pts = [(50.0, 100.0, 0.1)] * 17   # 신뢰도 미달
        det = _det(pts, x1=0, x2=100)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert sample_vest_hue(frame, det) == -1

    def test_black_vest_returns_minus1(self):
        """채도 낮은 검은색 픽셀 → -1 반환."""
        import cv2
        kpts = _make_kpts({
            _LS: (200.0, 100.0), _RS: (260.0, 100.0),
            _LH: (210.0, 180.0), _RH: (250.0, 180.0),
        })
        det = _det(kpts, x1=150, x2=310)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)   # 검은 프레임
        assert sample_vest_hue(frame, det) == -1

    def test_red_vest_returns_low_hue(self):
        """빨간색 BGR=(0,0,200) → hue ≈ 0 (±15)."""
        import cv2
        kpts = _make_kpts({
            _LS: (200.0, 100.0), _RS: (260.0, 100.0),
            _LH: (210.0, 180.0), _RH: (250.0, 180.0),
        })
        det = _det(kpts, x1=150, x2=310)
        frame = self._make_frame_with_vest(det, hue_bgr=(0, 0, 200))
        result = sample_vest_hue(frame, det)
        assert result != -1, "빨간 조끼에서 hue 추출 실패"
        # red hue is near 0 (wraps 0/180), check it's in red range
        assert result <= 15 or result >= 165, f"예상 red hue(0~15 or 165~179), got {result}"
