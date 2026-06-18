"""src/vest.py — detect_vest(), draw_vest_label() 단위 테스트."""
from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detection
from src.vest import VestResult, detect_vest, draw_vest_label, VEST_COLOR_BGR, ROLE_KO


def _det(x1=0.0, y1=0.0, x2=100.0, y2=200.0) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, keypoints=None)


def _solid_frame(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


class TestDetectVest:

    def test_returns_vest_result(self):
        det = _det()
        frame = _solid_frame(480, 640, (0, 0, 0))
        result = detect_vest(frame, det)
        assert isinstance(result, VestResult)

    def test_black_frame_returns_unknown(self):
        """채도 없는 검은 프레임 → unknown (신뢰도 미달)."""
        det = _det()
        frame = _solid_frame(480, 640, (0, 0, 0))
        result = detect_vest(frame, det)
        assert result.role == "unknown"

    def test_degenerate_bbox_returns_unknown(self):
        """면적 0인 바운딩 박스 → unknown."""
        det = _det(x1=50, y1=50, x2=50, y2=50)
        frame = _solid_frame(480, 640, (0, 0, 200))
        result = detect_vest(frame, det)
        assert result.role == "unknown"

    def test_red_vest_detected_as_main_attacker(self):
        """순수 빨간색(BGR 0,0,200) 프레임 → main_attacker."""
        det = _det(x1=0, y1=0, x2=200, y2=400)
        frame = _solid_frame(480, 640, (0, 0, 200))   # BGR red
        result = detect_vest(frame, det)
        assert result.role == "main_attacker", (
            f"expected main_attacker, got {result.role} (confidence={result.confidence})"
        )

    def test_blue_vest_detected_as_technician(self):
        """순수 파란색(BGR 200,0,0) 프레임 → technician."""
        det = _det(x1=0, y1=0, x2=200, y2=400)
        frame = _solid_frame(480, 640, (200, 0, 0))   # BGR blue
        result = detect_vest(frame, det)
        assert result.role == "technician", (
            f"expected technician, got {result.role}"
        )

    def test_white_vest_detected_as_defender(self):
        """흰색(BGR 240,240,240) 프레임 → defender."""
        det = _det(x1=0, y1=0, x2=200, y2=400)
        frame = _solid_frame(480, 640, (240, 240, 240))
        result = detect_vest(frame, det)
        assert result.role == "defender", (
            f"expected defender, got {result.role}"
        )

    def test_confidence_in_range(self):
        """신뢰도는 항상 0~1 사이."""
        det = _det(x1=0, y1=0, x2=200, y2=400)
        for bgr in [(0, 0, 200), (200, 0, 0), (240, 240, 240), (0, 0, 0)]:
            frame = _solid_frame(480, 640, bgr)
            result = detect_vest(frame, det)
            assert 0.0 <= result.confidence <= 1.0, f"confidence={result.confidence} 범위 초과"

    def test_pixel_ratio_in_range(self):
        """pixel_ratio는 항상 0~1 사이."""
        det = _det(x1=0, y1=0, x2=200, y2=400)
        frame = _solid_frame(480, 640, (0, 0, 200))
        result = detect_vest(frame, det)
        assert 0.0 <= result.pixel_ratio <= 1.0


class TestDrawVestLabel:

    def test_unknown_role_no_change(self):
        """role=unknown이면 프레임을 수정하지 않는다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        det = _det(x1=100, y1=50, x2=200, y2=300)
        result = VestResult(role="unknown", confidence=0.0, pixel_ratio=0.0)
        draw_vest_label(frame, det, result)
        assert np.array_equal(frame, original), "unknown role에서 프레임이 변경됨"

    def test_known_role_modifies_frame(self):
        """role이 알려진 포지션이면 라벨을 그려 프레임이 변경된다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = _det(x1=100, y1=50, x2=200, y2=300)
        result = VestResult(role="main_attacker", confidence=0.85, pixel_ratio=0.3)
        draw_vest_label(frame, det, result)
        assert frame.any(), "main_attacker 라벨이 그려지지 않았다"

    def test_draw_vest_label_no_crash_all_roles(self):
        """모든 역할에 대해 draw_vest_label이 예외 없이 동작해야 한다."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = _det(x1=50, y1=50, x2=150, y2=250)
        for role in ("main_attacker", "technician", "defender", "unknown"):
            f = frame.copy()
            draw_vest_label(f, det, VestResult(role=role, confidence=0.7, pixel_ratio=0.2))


class TestVestConstants:

    def test_vest_color_bgr_has_all_roles(self):
        for role in ("main_attacker", "technician", "defender", "unknown"):
            assert role in VEST_COLOR_BGR
            color = VEST_COLOR_BGR[role]
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_role_ko_has_all_roles(self):
        for role in ("main_attacker", "technician", "defender", "unknown"):
            assert role in ROLE_KO
            assert isinstance(ROLE_KO[role], str)
            assert len(ROLE_KO[role]) > 0
