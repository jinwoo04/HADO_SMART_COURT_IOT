"""src/measure_w5.py — _find_models(), _format_report() 단위 테스트."""
from __future__ import annotations

import pytest


class TestFindModels:

    def test_returns_dict_with_three_keys(self):
        from src.measure_w5 import _find_models
        result = _find_models()
        assert set(result.keys()) == {"NCNN", "ONNX", "PT"}

    def test_values_are_str_or_none(self):
        from src.measure_w5 import _find_models
        for v in _find_models().values():
            assert v is None or isinstance(v, str)

    def test_none_when_model_absent(self, tmp_path, monkeypatch):
        """모델 파일 없는 디렉터리에서는 모두 None 반환."""
        import src.measure_w5 as m
        monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
        result = m._find_models()
        assert all(v is None for v in result.values())

    def test_str_when_model_present(self, tmp_path, monkeypatch):
        """ONNX 파일이 있으면 해당 경로 반환."""
        import src.measure_w5 as m
        onnx = tmp_path / "yolov8n-pose.onnx"
        onnx.write_text("dummy")
        monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
        result = m._find_models()
        assert result["ONNX"] == str(onnx)
        assert result["NCNN"] is None


class TestFormatReport:

    def _sample_result(self, label: str, fps: float) -> dict:
        return {
            "model_label": label,
            "inference_fps": fps,
            "wall_fps": fps * 0.9,
            "p95_latency_ms": 70.0,
            "peak_ram_mb": 350.0,
            "peak_temp_c": 65.0,
        }

    def test_returns_string(self):
        from src.measure_w5 import _format_report
        r = _format_report([self._sample_result("NCNN", 16.0)], 250.0, "2026-06-15 09:00")
        assert isinstance(r, str)

    def test_contains_model_label(self):
        from src.measure_w5 import _format_report
        r = _format_report([self._sample_result("NCNN", 16.0)], -1.0, "2026-06-15 09:00")
        assert "NCNN" in r

    def test_pass_flag_when_fps_above_15(self):
        from src.measure_w5 import _format_report
        r = _format_report([self._sample_result("NCNN", 16.0)], -1.0, "2026-06-15 09:00")
        assert "✅" in r

    def test_warn_flag_when_fps_10_to_15(self):
        from src.measure_w5 import _format_report
        r = _format_report([self._sample_result("ONNX", 12.0)], -1.0, "2026-06-15 09:00")
        assert "⚠" in r

    def test_fail_flag_when_fps_below_10(self):
        from src.measure_w5 import _format_report
        r = _format_report([self._sample_result("PT", 3.0)], -1.0, "2026-06-15 09:00")
        assert "❌" in r

    def test_tts_shown_when_positive(self):
        from src.measure_w5 import _format_report
        r = _format_report([], 320.0, "2026-06-15 09:00")
        assert "320" in r

    def test_tts_na_when_negative(self):
        from src.measure_w5 import _format_report
        r = _format_report([], -1.0, "2026-06-15 09:00")
        assert "pyttsx3 없음" in r

    def test_contains_position_error_table(self):
        from src.measure_w5 import _format_report
        r = _format_report([], -1.0, "2026-06-15 09:00")
        assert "P1" in r and "P10" in r

    def test_multiple_models(self):
        from src.measure_w5 import _format_report
        results = [
            self._sample_result("NCNN", 15.5),
            self._sample_result("ONNX", 6.2),
        ]
        r = _format_report(results, -1.0, "2026-06-15 09:00")
        assert "NCNN" in r and "ONNX" in r
