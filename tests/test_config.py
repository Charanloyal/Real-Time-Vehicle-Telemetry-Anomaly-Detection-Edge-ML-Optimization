from __future__ import annotations

from pathlib import Path

import pytest

from vehicle_anomaly.config import (
    INFERENCE_THREADS_ENV_VAR,
    MODEL_DIR_ENV_VAR,
    RUNTIME_ENV_VAR,
    ConfigError,
    load_config,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def test_repo_config_loads_with_absolute_paths() -> None:
    config = load_config(REPO_CONFIG)
    assert config.project_root == REPO_CONFIG.parent.parent
    assert config.paths.model_dir.is_absolute()
    assert config.paths.model_dir == config.project_root / "artifacts" / "models"
    assert config.training.models == (
        "random_forest",
        "xgboost",
        "random_forest_window",
        "xgboost_window",
        "isolation_forest",
    )
    assert config.api.runtime == "native" and config.streaming.feature_set == "window"


def test_runtime_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, "ONNX")
    assert load_config(REPO_CONFIG).api.runtime == "onnx"
    monkeypatch.setenv(RUNTIME_ENV_VAR, "tensorrt")
    with pytest.raises(ConfigError, match=r"api\.runtime"):
        load_config(REPO_CONFIG)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("feature_set: window", "feature_set: sequence", "feature_set"),
        ("warm_start_rate: 0.5", "warm_start_rate: 1.5", "warm_start_rate"),
        ("drift_episodes_per_1000_events: 0.3   #", "drift_episodes_per_1000_events: -1   #", "drift_episodes"),
    ],
)
def test_new_settings_are_validated(tmp_path: Path, original: str, replacement: str, message: str) -> None:
    text = REPO_CONFIG.read_text(encoding="utf-8")
    assert original in text
    path = tmp_path / "config.yaml"
    path.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_model_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(tmp_path))
    assert load_config(REPO_CONFIG).paths.model_dir == tmp_path.resolve()


def test_inference_threads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_config(REPO_CONFIG).api.inference_threads is None
    monkeypatch.setenv(INFERENCE_THREADS_ENV_VAR, "1")
    assert load_config(REPO_CONFIG).api.inference_threads == 1
    for bad in ("zero", "0"):
        monkeypatch.setenv(INFERENCE_THREADS_ENV_VAR, bad)
        with pytest.raises(ConfigError, match=r"inference_threads|VTAD_INFERENCE_THREADS"):
            load_config(REPO_CONFIG)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    text = REPO_CONFIG.read_text(encoding="utf-8").replace("api:\n", "api:\n  unexpected: 1\n")
    path = tmp_path / "configs" / "config.yaml"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="unexpected"):
        load_config(path)


def test_invalid_engine_rejected(tmp_path: Path) -> None:
    text = REPO_CONFIG.read_text(encoding="utf-8").replace("engine: auto", "engine: flink")
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="engine"):
        load_config(path)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
