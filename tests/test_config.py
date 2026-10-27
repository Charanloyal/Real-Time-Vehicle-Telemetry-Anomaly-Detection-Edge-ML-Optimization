from __future__ import annotations

from pathlib import Path

import pytest

from vehicle_anomaly.config import MODEL_DIR_ENV_VAR, ConfigError, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def test_repo_config_loads_with_absolute_paths() -> None:
    config = load_config(REPO_CONFIG)
    assert config.project_root == REPO_CONFIG.parent.parent
    assert config.paths.model_dir.is_absolute()
    assert config.paths.model_dir == config.project_root / "artifacts" / "models"
    assert config.training.models == ("random_forest", "xgboost")


def test_model_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(tmp_path))
    assert load_config(REPO_CONFIG).paths.model_dir == tmp_path.resolve()


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
