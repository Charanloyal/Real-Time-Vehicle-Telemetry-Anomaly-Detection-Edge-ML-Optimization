from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vehicle_anomaly.config import AppConfig
from vehicle_anomaly.data_generator import (
    demo_fleet_events,
    generate_streaming_data,
    generate_telemetry,
    write_streaming_files,
)
from vehicle_anomaly.predict import AnomalyPredictor, WindowAnomalyPredictor
from vehicle_anomaly.schema import REQUIRED_FIELDS
from vehicle_anomaly.stream_common import BatchProcessor, load_batch_metrics, load_predictions
from vehicle_anomaly.streaming import (
    build_spark_session,
    parse_raw_lines,
    process_spark_batch,
    run_streaming,
    spark_status,
)
from vehicle_anomaly.validation import validate_lines


def _assert_totals_match(summary: dict, generated) -> None:
    assert summary["invalid_rows"] == generated.malformed_events
    assert summary["duplicates_dropped"] == generated.duplicate_events
    assert summary["scored_rows"] == generated.clean_events
    assert summary["input_rows"] == generated.total_lines
    assert summary["valid_rows"] == generated.clean_events + generated.duplicate_events


def test_local_engine_end_to_end(stream_config: AppConfig) -> None:
    generated = generate_streaming_data(stream_config)
    summary = run_streaming(stream_config, engine="local", reset=True)

    _assert_totals_match(summary, generated)
    assert summary["run"]["engine"] == "local"
    assert summary["batches"] == 2  # 4 files, max_files_per_trigger=2
    assert sum(summary["invalid_by_category"].values()) == generated.malformed_events
    assert "ground_truth_comparison" in summary
    assert summary["ground_truth_comparison"]["recall"] > 0.5

    predictions = load_predictions(stream_config.paths.streaming_output)
    assert len(predictions) == generated.clean_events
    assert predictions["event_id"].is_unique
    assert {"anomaly_score", "is_anomaly", "batch_id", "model_name"} <= set(predictions.columns)
    invalid_files = list((stream_config.paths.streaming_output / "invalid").glob("*.jsonl"))
    quarantined = [json.loads(line) for f in invalid_files for line in f.read_text(encoding="utf-8").splitlines()]
    assert len(quarantined) == generated.malformed_events
    assert all(row["invalid_reason"] for row in quarantined)
    reports = stream_config.paths.reports_dir
    assert (reports / "streaming_summary.md").is_file() and (reports / "figures" / "streaming_batches.png").is_file()
    assert summary["run"]["feature_set"] == "window"
    assert (
        "battery_degradation" in summary["ground_truth_comparison"]["by_type"]
        or "gradual_overheating" in summary["ground_truth_comparison"]["by_type"]
    )
    state_files = sorted((stream_config.paths.streaming_output / "state").glob("batch_*.json"))
    assert [f.name for f in state_files] == ["batch_000000.json", "batch_000001.json"]
    state = json.loads(state_files[-1].read_text(encoding="utf-8"))
    assert set(state) == set(predictions["vehicle_id"])


def test_replayed_batch_after_crash_reproduces_window_scores(stream_config: AppConfig) -> None:
    """Simulate a crash after batch 1 wrote its outputs but before its checkpoint commit."""
    generate_streaming_data(stream_config)
    run_streaming(stream_config, engine="local", reset=True)
    original = load_predictions(stream_config.paths.streaming_output).set_index("event_id")["anomaly_score"]

    checkpoint = stream_config.paths.checkpoint_dir / "local" / "local_engine_state.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    checkpoint.write_text(
        json.dumps({"processed_files": state["processed_files"][:2], "next_batch_id": 1}), encoding="utf-8"
    )
    replay = run_streaming(stream_config, engine="local")
    assert replay["run"]["batches_this_run"] == 1 and replay["batches"] == 2
    replayed = load_predictions(stream_config.paths.streaming_output).set_index("event_id")["anomaly_score"]
    pd.testing.assert_series_equal(replayed.sort_index(), original.sort_index())


def test_batch_without_rows_does_not_reset_window_state(stream_config: AppConfig, tmp_path) -> None:
    """Spark skips empty micro-batches (no state file); the next batch must continue from the last state."""
    events = demo_fleet_events(seed=3, start=0, n=800, anomaly_rate=0.05, drift_rate=3.0)
    valid = events.loc[:, list(REQUIRED_FIELDS)]
    no_invalid = pd.DataFrame(columns=["raw_value", "invalid_reason"])
    predictor = WindowAnomalyPredictor.load(stream_config.paths.model_dir)
    expected, _ = predictor.predict_frame(valid)

    for restart in (False, True):
        out = tmp_path / f"restart_{restart}"
        processor = BatchProcessor(predictor, out)
        processor.process(0, valid.iloc[:400], no_invalid, 400, "spark")
        # batch 1 was empty: process() is never called for it, so no state/batch_000001.json exists
        if restart:
            processor = BatchProcessor(predictor, out)  # new process: state must come from disk
        processor.process(2, valid.iloc[400:], no_invalid, 400, "spark")
        scores = load_predictions(out).set_index("event_id")["anomaly_score"]
        np.testing.assert_allclose(
            scores.loc[expected["event_id"]].to_numpy(), expected["anomaly_score"].to_numpy(), rtol=0, atol=1e-12
        )


def test_point_feature_set_streaming(stream_config: AppConfig) -> None:
    from dataclasses import replace

    config = replace(stream_config, streaming=replace(stream_config.streaming, feature_set="point"))
    generated = generate_streaming_data(config)
    summary = run_streaming(config, engine="local", reset=True)
    assert summary["run"]["feature_set"] == "point" and summary["scored_rows"] == generated.clean_events
    assert not list((config.paths.streaming_output / "state").glob("*.json"))


def test_local_engine_recovers_from_checkpoint(stream_config: AppConfig) -> None:
    generate_streaming_data(stream_config)
    first = run_streaming(stream_config, engine="local", reset=True)

    rerun = run_streaming(stream_config, engine="local")
    assert rerun["run"]["batches_this_run"] == 0
    assert rerun["batches"] == first["batches"]  # nothing re-processed, metrics unchanged
    assert rerun["input_rows"] == first["input_rows"]

    extra = generate_telemetry(num_vehicles=1, events_per_vehicle=100, anomaly_rate=0.05, seed=99, vehicle_prefix="NEW")
    write_streaming_files(
        extra,
        stream_config.paths.streaming_input,
        num_files=1,
        malformed_rate=0.0,
        duplicate_rate=0.0,
        seed=1,
        file_prefix="zz_late",
    )
    resumed = run_streaming(stream_config, engine="local")
    assert resumed["run"]["batches_this_run"] == 1
    assert resumed["batches"] == first["batches"] + 1
    assert resumed["input_rows"] == first["input_rows"] + 100
    last = load_batch_metrics(stream_config.paths.streaming_output)[-1]
    assert last["source_files"] == ["zz_late_00001.jsonl"]
    assert last["batch_id"] == first["batches"]


def test_csv_output_format(stream_config: AppConfig) -> None:
    from dataclasses import replace

    config = replace(stream_config, streaming=replace(stream_config.streaming, output_format="csv"))
    generated = generate_streaming_data(config)
    run_streaming(config, engine="local", reset=True)
    files = list((config.paths.streaming_output / "predictions").glob("*.csv"))
    assert files
    assert sum(len(pd.read_csv(f)) for f in files) == generated.clean_events


@pytest.mark.spark
@pytest.mark.skipif(not spark_status(file_io=False)[0], reason=f"Spark unavailable: {spark_status(False)[1]}")
def test_spark_validation_and_batch_processing_match_local(
    stream_config: AppConfig, predictor: AnomalyPredictor, tmp_path
) -> None:
    """Runs the Spark parsing/validation expressions and foreachBatch body on an in-memory DataFrame.

    This avoids Hadoop's local filesystem, so it also runs on Windows without winutils.exe.
    """
    generate_streaming_data(stream_config)
    lines = [
        line
        for path in sorted(stream_config.paths.streaming_input.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    local = validate_lines(lines)
    local_metrics = BatchProcessor(predictor, tmp_path / "local").process(
        0, local.valid, local.invalid, local.input_rows, "local"
    )

    spark = build_spark_session(stream_config)
    try:
        validated = parse_raw_lines(spark.createDataFrame([(line,) for line in lines], "value string"))
        spark_metrics = process_spark_batch(validated, 0, BatchProcessor(predictor, tmp_path / "spark"))
    finally:
        spark.stop()

    assert spark_metrics is not None
    for field in (
        "input_rows",
        "valid_rows",
        "invalid_rows",
        "duplicates_dropped",
        "scored_rows",
        "anomalies_detected",
        "invalid_by_reason",
    ):
        assert getattr(spark_metrics, field) == getattr(local_metrics, field), field
    merged = load_predictions(tmp_path / "local").merge(
        load_predictions(tmp_path / "spark"), on="event_id", suffixes=("_local", "_spark")
    )
    assert len(merged) == local_metrics.scored_rows
    assert (merged["anomaly_score_local"] - merged["anomaly_score_spark"]).abs().max() < 1e-9
    for column in ("speed", "battery_voltage", "latitude"):
        assert (merged[f"{column}_local"] == merged[f"{column}_spark"]).all()
    assert (merged["event_timestamp_local"] == merged["event_timestamp_spark"]).all()


@pytest.mark.spark
@pytest.mark.skipif(not spark_status()[0], reason=f"Spark file streaming unavailable: {spark_status()[1]}")
def test_spark_engine_matches_local_engine(stream_config: AppConfig) -> None:
    generated = generate_streaming_data(stream_config)
    local = run_streaming(stream_config, engine="local", reset=True)
    local_predictions = load_predictions(stream_config.paths.streaming_output)

    spark = run_streaming(stream_config, engine="spark", reset=True)
    assert spark["run"]["engine"] == "spark"
    _assert_totals_match(spark, generated)
    assert spark["invalid_by_reason"] == local["invalid_by_reason"]
    assert spark["anomalies_detected"] == local["anomalies_detected"]

    spark_predictions = load_predictions(stream_config.paths.streaming_output)
    merged = local_predictions.merge(spark_predictions, on="event_id", suffixes=("_local", "_spark"))
    assert len(merged) == generated.clean_events
    assert (merged["is_anomaly_local"] == merged["is_anomaly_spark"]).all()
    assert (merged["anomaly_score_local"] - merged["anomaly_score_spark"]).abs().max() < 1e-9

    # Restarting from the Spark checkpoint must not re-process already committed files.
    rerun = run_streaming(stream_config, engine="spark")
    assert rerun["input_rows"] == spark["input_rows"]
