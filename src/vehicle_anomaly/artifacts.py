"""Saving and loading model artifacts (preprocessor, models, metadata)."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

PREPROCESSOR_FILE = "preprocessor.joblib"
METADATA_FILE = "metadata.json"


class ModelArtifactsError(FileNotFoundError):
    """Raised when model artifacts are missing or inconsistent."""


def model_file_name(model_name: str) -> str:
    return f"{model_name}.joblib"


def library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:
        versions["xgboost"] = "not installed"
    return versions


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_preprocessor(model_dir: Path, preprocessor: Any) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / PREPROCESSOR_FILE
    joblib.dump(preprocessor, path)
    return path


def save_model(model_dir: Path, model_name: str, model: Any) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / model_file_name(model_name)
    joblib.dump(model, path)
    return path


def save_metadata(model_dir: Path, metadata: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / METADATA_FILE
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path


def load_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / METADATA_FILE
    if not path.is_file():
        raise ModelArtifactsError(
            f"Model metadata not found at {path}. Train a model first: python scripts/train_model.py"
        )
    metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return metadata


def load_preprocessor(model_dir: Path) -> Any:
    path = model_dir / PREPROCESSOR_FILE
    if not path.is_file():
        raise ModelArtifactsError(f"Preprocessor not found at {path}")
    return joblib.load(path)


def load_model(model_dir: Path, model_name: str) -> Any:
    path = model_dir / model_file_name(model_name)
    if not path.is_file():
        raise ModelArtifactsError(f"Model '{model_name}' not found at {path}")
    return joblib.load(path)
