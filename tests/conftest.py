from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vehicle_anomaly.config import AppConfig, LoggingConfig, PathsConfig, load_config
from vehicle_anomaly.data_generator import generate_training_data
from vehicle_anomaly.predict import AnomalyPredictor
from vehicle_anomaly.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def make_paths(root: Path) -> PathsConfig:
    return PathsConfig(
        raw_data=root / "data" / "raw" / "telemetry.parquet",
        processed_dir=root / "data" / "processed",
        streaming_input=root / "data" / "streaming_input",
        streaming_ground_truth=root / "data" / "raw" / "streaming_ground_truth.parquet",
        streaming_output=root / "data" / "streaming_output",
        checkpoint_dir=root / "data" / "checkpoints",
        model_dir=root / "artifacts" / "models",
        reports_dir=root / "reports",
    )


def small_config(root: Path) -> AppConfig:
    base = load_config(REPO_CONFIG)
    return replace(
        base,
        project_root=root,
        paths=make_paths(root),
        data_generation=replace(
            base.data_generation, num_vehicles=20, events_per_vehicle=1500, drift_episodes_per_1000_events=1.5
        ),
        streaming_data=replace(
            base.streaming_data,
            num_vehicles=4,
            num_files=4,
            events_per_file=1000,
            malformed_rate=0.03,
            duplicate_rate=0.02,
            drift_episodes_per_1000_events=1.5,
        ),
        training=replace(
            base.training,
            random_forest={**base.training.random_forest, "n_estimators": 25, "n_jobs": 2},
            xgboost={**base.training.xgboost, "n_estimators": 60, "n_jobs": 2},
            isolation_forest={**base.training.isolation_forest, "n_estimators": 50, "n_jobs": 2},
        ),
        benchmark=replace(
            base.benchmark,
            single_iterations=20,
            warmup_iterations=3,
            batch_size=50,
            batch_iterations=2,
            load_iterations=1,
        ),
        logging=LoggingConfig(level="WARNING", json=False),
    )


@pytest.fixture(scope="session")
def trained_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    """Small synthetic dataset + trained models, built once per test session."""
    config = small_config(tmp_path_factory.mktemp("vtad"))
    generate_training_data(config)
    run_training(config, evaluate=True)
    return config


@pytest.fixture(scope="session")
def predictor(trained_config: AppConfig) -> AnomalyPredictor:
    return AnomalyPredictor.load(trained_config.paths.model_dir)


@pytest.fixture
def stream_config(trained_config: AppConfig, tmp_path: Path) -> AppConfig:
    """Trained models shared with the session, but isolated streaming input/output per test."""
    paths = replace(
        trained_config.paths,
        streaming_input=tmp_path / "streaming_input",
        streaming_ground_truth=tmp_path / "streaming_ground_truth.parquet",
        streaming_output=tmp_path / "streaming_output",
        checkpoint_dir=tmp_path / "checkpoints",
        reports_dir=tmp_path / "reports",
    )
    return replace(trained_config, paths=paths)
