"""src/pose.py — classify_hado_action() 단위 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detection
from src.pose import (
    ACTION_KO, ActionResult,
    classify_hado_action,
)

_W, _H = 100.0, 200.0   # 가상 바운딩 박스 크기


def _det(kpts_17x3: list[tuple[float, float, float]]) -> Detection:
    arr = np.array(kpts_17x3, dtype=np.float32)  # (17, 3)
    return Detection(x1=0.0, y1=0.0, x2=_W, y2=_H,
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
        assert set(result.scores.keys()) == {"shoot", "shield", "crouch", "dodge_l", "dodge_r", "ready"}

    def test_shoot_arm_raised(self):
        """손목이 어깨보다 bbox_h*0.3 위 → shoot 판정."""
        kpts = _make_kpts({
            _LS: (40.0, 80.0), _RS: (60.0, 80.0),
            _LW: (30.0, 20.0),   # 왼손목을 어깨보다 훨씬 위로
            _RW: (70.0, 80.0),
            _LH: (45.0, 130.0), _RH: (55.0, 130.0),
            _LK: (45.0, 155.0), _RK: (55.0, 155.0),
        })
        result = classify_hado_action(_det(kpts))
        assert result is not None
        assert result.action == "shoot", f"expected shoot, got {result.action} (scores={result.scores})"

    def test_crouch_compressed_pose(self):
        """무릎이 어깨와 가까이 있음 → crouch 판정."""
        kpts = _make_kpts({
            _LS: (40.0, 80.0), _RS: (60.0, 80.0),
            _LK: (42.0, 110.0), _RK: (58.0, 110.0),   # 무릎이 어깨 바로 아래
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
        """직립 자세 → ready 판정."""
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
        pts = [(50.0, 10.0, 0.1)] * 17   # 모두 신뢰도 0.1 (< 0.25 임계값)
        result = classify_hado_action(_det(pts))
        assert result is not None
        assert result.action == "ready"
