"""src/pi_preflight.py 단위 테스트."""
from __future__ import annotations

import json
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── check_python_deps ──────────────────────────────────────────────────────────

def test_deps_pass_when_all_importable(capsys):
    from src.pi_preflight import check_python_deps
    result = check_python_deps()
    out = capsys.readouterr().out
    assert "ultralytics" in out
    assert "cv2" in out or "opencv" in out


# ── check_models ───────────────────────────────────────────────────────────────

def test_models_detects_ncnn(tmp_path, monkeypatch):
    from src import pi_preflight as pf
    ncnn_dir = tmp_path / "yolov8n-pose_ncnn_model"
    ncnn_dir.mkdir()
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    assert pf.check_models() is True


def test_models_detects_onnx_fallback(tmp_path, monkeypatch):
    from src import pi_preflight as pf
    (tmp_path / "yolov8n-pose.onnx").touch()
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    assert pf.check_models() is True


def test_models_fails_when_nothing(tmp_path, monkeypatch):
    from src import pi_preflight as pf
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    assert pf.check_models() is False


# ── check_calibration ─────────────────────────────────────────────────────────

def test_calibration_missing(tmp_path, monkeypatch):
    from src import pi_preflight as pf
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    assert pf.check_calibration() is False


def test_calibration_valid(tmp_path, monkeypatch):
    from src import pi_preflight as pf
    from src.homography import compute_homography
    calib_dir = tmp_path / "config"
    calib_dir.mkdir()
    corners = np.array([[100, 100], [1180, 100], [1180, 600], [100, 600]],
                       dtype=np.float32)
    calib = compute_homography(corners, court_width_m=10.0, court_height_m=6.0,
                               image_size=(1280, 720))
    calib.to_json(calib_dir / "calibration.json")
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    assert pf.check_calibration() is True


def test_calibration_corrupt_json(tmp_path, monkeypatch):
    from src import pi_preflight as pf
    calib_dir = tmp_path / "config"
    calib_dir.mkdir()
    (calib_dir / "calibration.json").write_text("not valid json{{{")
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    assert pf.check_calibration() is False


# ── check_data_dirs ───────────────────────────────────────────────────────────

def test_data_dirs_creates_and_passes(tmp_path, monkeypatch):
    from src import pi_preflight as pf
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    assert pf.check_data_dirs() is True
    assert (tmp_path / "data" / "position_logs").exists()
    assert (tmp_path / "data" / "snapshots").exists()


# ── _resolve_model_path (main.py) ─────────────────────────────────────────────

def test_resolve_model_prefers_ncnn(tmp_path, monkeypatch):
    import src.main as m_mod
    ncnn = tmp_path / "yolov8n-pose_ncnn_model"
    ncnn.mkdir()
    monkeypatch.setattr(m_mod, "PROJECT_ROOT", tmp_path)
    result = m_mod._resolve_model_path("yolov8n.pt")
    assert "ncnn" in result.lower()


def test_resolve_model_falls_to_onnx(tmp_path, monkeypatch):
    import src.main as m_mod
    (tmp_path / "yolov8n-pose.onnx").touch()
    monkeypatch.setattr(m_mod, "PROJECT_ROOT", tmp_path)
    result = m_mod._resolve_model_path("yolov8n.pt")
    assert result.endswith(".onnx")


def test_resolve_model_uses_config_fallback(tmp_path, monkeypatch):
    import src.main as m_mod
    monkeypatch.setattr(m_mod, "PROJECT_ROOT", tmp_path)
    result = m_mod._resolve_model_path("yolov8n.pt")
    assert result == "yolov8n.pt"


# ── _check_display (main.py) ──────────────────────────────────────────────────

def test_check_display_headless_arg(monkeypatch):
    import src.main as m_mod
    args = MagicMock()
    args.headless = True
    assert m_mod._check_display(args) is True


def test_check_display_linux_no_display(monkeypatch):
    import src.main as m_mod
    args = MagicMock()
    args.headless = False
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    result = m_mod._check_display(args)
    assert result is True  # 자동 headless 전환


def test_check_display_mac_no_headless(monkeypatch):
    import src.main as m_mod
    args = MagicMock()
    args.headless = False
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    result = m_mod._check_display(args)
    assert result is False  # Mac → headless 강제 안 함
