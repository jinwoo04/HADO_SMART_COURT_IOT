"""src/benchmark.py — _ram_mb(), _cpu_temp_c(), benchmark() 단위 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestRamMb:

    def test_returns_float(self):
        from src.benchmark import _ram_mb
        result = _ram_mb()
        assert isinstance(result, float)

    def test_positive_or_zero(self):
        """psutil 있으면 양수, 없으면 0."""
        from src.benchmark import _ram_mb
        assert _ram_mb() >= 0.0

    def test_reasonable_range(self):
        """실행 중인 프로세스 RAM은 최소 10MB, 최대 8GB 이내."""
        from src.benchmark import _ram_mb, _PSUTIL
        if not _PSUTIL:
            pytest.skip("psutil 미설치")
        mb = _ram_mb()
        assert 10.0 < mb < 8192.0, f"비현실적인 RAM: {mb:.0f} MB"


class TestCpuTempC:

    def test_returns_float(self):
        from src.benchmark import _cpu_temp_c
        result = _cpu_temp_c()
        assert isinstance(result, float)

    def test_zero_on_non_pi(self):
        """/sys/class/thermal 없는 Mac에서는 0.0 반환."""
        import platform
        from src.benchmark import _cpu_temp_c
        if platform.system() == "Linux":
            pytest.skip("Pi4 환경에서는 실제 온도 반환될 수 있음")
        assert _cpu_temp_c() == 0.0

    def test_non_negative(self):
        from src.benchmark import _cpu_temp_c
        assert _cpu_temp_c() >= 0.0


# ── benchmark() 반환 dict 구조 검증 ────────────────────────────────────────────

_REQUIRED_KEYS = {
    "n_frames", "avg_latency_ms", "median_latency_ms", "p95_latency_ms",
    "inference_fps", "wall_fps", "peak_ram_mb", "peak_temp_c", "imgsz", "model",
}

_DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _make_mock_detector():
    det = MagicMock()
    det.detect.return_value = []
    return det


def _make_mock_camera():
    cam = MagicMock()
    cam.read.return_value = (True, _DUMMY_FRAME)
    return cam


class TestBenchmarkReturnDict:

    def _run_tiny(self, source=0, n_frames=3, warmup=1):
        from src.benchmark import benchmark
        with (
            patch("src.benchmark.PersonDetector", return_value=_make_mock_detector()),
            patch("src.benchmark.Camera", return_value=_make_mock_camera()),
        ):
            return benchmark(source=source, n_frames=n_frames, imgsz=320,
                             model_path="dummy.pt", warmup=warmup)

    def test_return_keys_complete(self):
        """반환 dict에 필수 키가 모두 있어야 한다."""
        result = self._run_tiny()
        missing = _REQUIRED_KEYS - set(result.keys())
        assert not missing, f"누락 키: {missing}"

    def test_inference_fps_consistent(self):
        """inference_fps == 1000 / avg_latency_ms."""
        result = self._run_tiny()
        expected = 1000 / result["avg_latency_ms"]
        assert abs(result["inference_fps"] - expected) < 1e-6

    def test_n_frames_matches_actual(self):
        """n_frames 키가 요청한 프레임 수와 일치해야 한다."""
        result = self._run_tiny(n_frames=5)
        assert result["n_frames"] == 5

    def test_peak_ram_non_negative(self):
        result = self._run_tiny()
        assert result["peak_ram_mb"] >= 0.0

    def test_model_path_stored(self):
        result = self._run_tiny()
        assert result["model"] == "dummy.pt"
