"""src/benchmark.py — _ram_mb(), _cpu_temp_c() 단위 테스트."""
from __future__ import annotations

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
