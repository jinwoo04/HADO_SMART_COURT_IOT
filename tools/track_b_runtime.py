"""Shared runtime helpers for Track B scripts."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILENAME = "hado_player_v4_player_only_yolov8n_mps_e30_best.pt"


def default_model_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("TRACK_B_PLAYER_MODEL")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            PROJECT_ROOT / "models" / MODEL_FILENAME,
            PROJECT_ROOT / "outputs" / "track_b_roboflow_export_review" / MODEL_FILENAME,
            Path.home() / "Documents" / "Codex",
        ]
    )
    return candidates


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg:
        model_path = Path(model_arg).expanduser()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(model_path)

    for candidate in default_model_candidates():
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            matches = sorted(candidate.rglob(MODEL_FILENAME))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        "Track B model checkpoint was not found. Pass --model or set TRACK_B_PLAYER_MODEL."
    )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
