from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vehicle_anomaly.artifacts import ModelArtifactsError, load_metadata
from vehicle_anomaly.benchmark import run_benchmark
from vehicle_anomaly.config import AppConfig
from vehicle_anomaly.predict import AnomalyPredictor
from vehicle_anomaly.schema import FEATURE_COLUMNS, REQUIRED_FIELDS
from vehicle_anomaly.train import TEST_SET_FILE, classification_metrics, tune_threshold, xgboost_status


@pytest.fixture(scope="module")
def test_events(trained_config: AppConfig) -> list[dict]:
    frame = pd.read_parquet(trained_config.paths.processed_dir / TEST_SET_FILE)
    return frame.loc[:, list(REQUIRED_FIELDS)].head(300).to_dict("records")


def _model_names(config: AppConfig) -> list[str]:
    return list(load_metadata(config.paths.model_dir)["models"])


def test_training_produces_artifacts_and_metadata(trained_config: AppConfig) -> None:
    paths = trained_config.paths
    metadata = load_metadata(paths.model_dir)
    expected = {"random_forest", "xgboost"} if xgboost_status()[0] else {"random_forest"}
    assert set(metadata["models"]) == expected
    assert metadata["selected_model"] in expected
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    for leaky in ("vehicle_id", "event_id", "event_timestamp", "is_anomaly", "anomaly_type"):
        assert leaky not in metadata["feature_columns"]
    for entry in metadata["models"].values():
        assert 0.0 < entry["threshold"] < 1.0
        assert "test_metrics" in entry and "confusion_matrix" in entry["test_metrics"]
    assert (paths.reports_dir / "model_evaluation.md").is_file()
    assert (paths.figures_dir / "confusion_matrices.png").is_file()


def test_splits_do_not_share_vehicles(trained_config: AppConfig) -> None:
    splits = json.loads((trained_config.paths.processed_dir / "splits.json").read_text(encoding="utf-8"))
    train, val, test = (set(splits[k]) for k in ("train_vehicles", "validation_vehicles", "test_vehicles"))
    assert not (train & val or train & test or val & test)


def test_models_learn_signal(trained_config: AppConfig) -> None:
    metadata = load_metadata(trained_config.paths.model_dir)
    for entry in metadata["models"].values():
        assert entry["test_metrics"]["roc_auc"] > 0.85


def test_optimized_scores_match_reference_pipeline(trained_config: AppConfig, test_events: list[dict]) -> None:
    for name in _model_names(trained_config):
        predictor = AnomalyPredictor.load(trained_config.paths.model_dir, name)
        reference = predictor.model.predict_proba(predictor.preprocessor.transform(pd.DataFrame(test_events)))[:, 1]
        batch = np.array([p.anomaly_score for p in predictor.predict_records(test_events)])
        single = np.array([predictor.predict_one(e).anomaly_score for e in test_events[:50]])
        np.testing.assert_allclose(batch, reference, rtol=0, atol=1e-6)
        np.testing.assert_allclose(single, reference[:50], rtol=0, atol=1e-6)


def test_thread_limit_does_not_change_scores(trained_config: AppConfig, test_events: list[dict]) -> None:
    events = test_events * 2  # > SMALL_BATCH_ROWS so the multi-threaded estimator path is used
    for name in _model_names(trained_config):
        default = AnomalyPredictor.load(trained_config.paths.model_dir, name)
        limited = AnomalyPredictor.load(trained_config.paths.model_dir, name, n_jobs=1)
        np.testing.assert_allclose(
            [p.anomaly_score for p in limited.predict_records(events)],
            [p.anomaly_score for p in default.predict_records(events)],
            rtol=0, atol=1e-9,
        )


def test_predict_one_returns_prediction(predictor: AnomalyPredictor, test_events: list[dict]) -> None:
    prediction = predictor.predict_one(test_events[0])
    assert prediction.event_id == test_events[0]["event_id"]
    assert 0.0 <= prediction.anomaly_score <= 1.0
    assert prediction.is_anomaly == (prediction.anomaly_score >= predictor.threshold)


def test_obvious_anomaly_scores_higher_than_normal(predictor: AnomalyPredictor, test_events: list[dict]) -> None:
    normal = {**test_events[0], "engine_temperature": 90.0, "battery_voltage": 14.2, "acceleration": 0.2,
              "brake_pressure": 0.5}
    overheating = {**normal, "engine_temperature": 126.0, "battery_voltage": 10.0}
    assert predictor.predict_one(overheating).anomaly_score > predictor.predict_one(normal).anomaly_score
    assert predictor.predict_one(overheating).is_anomaly


def test_missing_values_are_imputed(predictor: AnomalyPredictor, test_events: list[dict]) -> None:
    event = {**test_events[0], "fuel_level": None}
    event.pop("brake_pressure")
    assert 0.0 <= predictor.predict_one(event).anomaly_score <= 1.0


def test_predict_frame_adds_columns(predictor: AnomalyPredictor, test_events: list[dict]) -> None:
    frame = pd.DataFrame(test_events[:10])
    scored = predictor.predict_frame(frame)
    assert {"anomaly_score", "is_anomaly"} <= set(scored.columns)
    assert len(scored) == 10 and "anomaly_score" not in frame.columns
    assert predictor.predict_frame(frame.iloc[:0]).empty
    assert predictor.predict_records([]) == []


def test_load_missing_artifacts_raises(tmp_path) -> None:
    with pytest.raises(ModelArtifactsError, match="train_model.py"):
        AnomalyPredictor.load(tmp_path)


def test_load_unknown_model_raises(trained_config: AppConfig) -> None:
    with pytest.raises(ModelArtifactsError, match="not listed"):
        AnomalyPredictor.load(trained_config.paths.model_dir, "does_not_exist")


def test_threshold_tuning_and_metrics() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = tune_threshold(y, scores)
    assert 0.3 < threshold <= 0.8
    metrics = classification_metrics(y, scores, threshold)
    assert metrics["f1"] == 1.0 and metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 2]]
    assert classification_metrics(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3]), 0.5)["roc_auc"] is None


def test_benchmark_report(trained_config: AppConfig, tmp_path) -> None:
    config = replace(trained_config, paths=replace(trained_config.paths, reports_dir=tmp_path))
    report = run_benchmark(config)
    assert {"environment", "settings", "models", "measured_at"} <= set(report)
    assert report["environment"]["logical_cpus"]
    for result in report["models"]:
        assert set(result["single_event"]) == {"baseline", "numpy_features", "optimized"}
        assert set(result["batch"]) == {"baseline_per_event_loop", "baseline_vectorized", "optimized_batch"}
        assert result["single_event"]["optimized"]["calls"] == config.benchmark.single_iterations
        assert result["parity"]["max_abs_score_diff_single"] < 1e-6
        assert result["parity"]["prediction_agreement"] == 1.0
        assert "single_mean_latency_pct" in result["improvement"]
    assert (tmp_path / "benchmark_results.md").is_file()
    assert (tmp_path / "figures" / "benchmark_latency.png").is_file()
