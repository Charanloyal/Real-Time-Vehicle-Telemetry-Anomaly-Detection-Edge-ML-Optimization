from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vehicle_anomaly.data_generator import (
    DEMO_STREAM_LENGTH,
    demo_fleet_events,
    generate_telemetry,
    write_streaming_files,
)
from vehicle_anomaly.schema import (
    ANOMALY_TYPE_COLUMN,
    DRIFT_ANOMALY_TYPES,
    LABEL_COLUMN,
    NORMAL_TYPE,
    POINT_ANOMALY_TYPES,
    REQUIRED_FIELDS,
    VALID_RANGES,
)
from vehicle_anomaly.validation import validate_lines


@pytest.fixture(scope="module")
def telemetry() -> pd.DataFrame:
    return generate_telemetry(num_vehicles=10, events_per_vehicle=1000, anomaly_rate=0.05, seed=7)


def test_same_seed_is_reproducible() -> None:
    first = generate_telemetry(num_vehicles=3, events_per_vehicle=200, anomaly_rate=0.05, seed=11)
    second = generate_telemetry(num_vehicles=3, events_per_vehicle=200, anomaly_rate=0.05, seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_data() -> None:
    first = generate_telemetry(num_vehicles=2, events_per_vehicle=200, anomaly_rate=0.05, seed=1)
    second = generate_telemetry(num_vehicles=2, events_per_vehicle=200, anomaly_rate=0.05, seed=2)
    assert not np.allclose(first["speed"], second["speed"])


def test_schema_and_identifiers(telemetry: pd.DataFrame) -> None:
    assert list(telemetry.columns) == [*REQUIRED_FIELDS, LABEL_COLUMN, ANOMALY_TYPE_COLUMN]
    assert len(telemetry) == 10_000
    assert telemetry["event_id"].is_unique
    assert telemetry["vehicle_id"].nunique() == 10
    assert telemetry[list(REQUIRED_FIELDS)].notna().all().all()


def test_values_within_valid_ranges(telemetry: pd.DataFrame) -> None:
    for name, (low, high) in VALID_RANGES.items():
        assert telemetry[name].between(low, high).all(), name


def test_timestamps_are_utc_and_increase_per_vehicle(telemetry: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(telemetry["event_timestamp"], utc=True)
    assert timestamps.notna().all()
    assert timestamps.groupby(telemetry["vehicle_id"]).apply(lambda s: s.is_monotonic_increasing).all()


def test_anomaly_labels(telemetry: pd.DataFrame) -> None:
    rate = telemetry[LABEL_COLUMN].mean()
    assert 0.035 < rate < 0.065
    assert set(telemetry[ANOMALY_TYPE_COLUMN]) == {NORMAL_TYPE, *POINT_ANOMALY_TYPES}  # drift is off by default
    assert (telemetry[LABEL_COLUMN].astype(bool) == (telemetry[ANOMALY_TYPE_COLUMN] != NORMAL_TYPE)).all()


def test_anomalies_differ_from_normal_behaviour(telemetry: pd.DataFrame) -> None:
    normal = telemetry[telemetry[LABEL_COLUMN] == 0]
    overheating = telemetry[telemetry[ANOMALY_TYPE_COLUMN] == "overheating"]
    battery = telemetry[telemetry[ANOMALY_TYPE_COLUMN] == "battery_voltage"]
    assert overheating["engine_temperature"].median() > normal["engine_temperature"].quantile(0.99)
    assert normal["battery_voltage"].between(12.8, 15.2).mean() > 0.999
    assert not battery["battery_voltage"].between(13.25, 14.85).any()


def test_gradual_drift_injection() -> None:
    df = generate_telemetry(
        num_vehicles=12, events_per_vehicle=2000, anomaly_rate=0.0, seed=21, drift_episodes_per_1000_events=1.5
    )
    assert set(df[ANOMALY_TYPE_COLUMN]) == {NORMAL_TYPE, *DRIFT_ANOMALY_TYPES}
    position = df.groupby("vehicle_id").cumcount()
    assert position[df[LABEL_COLUMN] == 1].min() >= 300  # drifts start after a baseline exists
    for vehicle, group in df.groupby("vehicle_id"):
        normal = group[group[ANOMALY_TYPE_COLUMN] == NORMAL_TYPE]
        hot = group[group[ANOMALY_TYPE_COLUMN] == "gradual_overheating"]
        weak = group[group[ANOMALY_TYPE_COLUMN] == "battery_degradation"]
        if len(hot):
            assert hot["engine_temperature"].mean() > normal["engine_temperature"].tail(500).mean() + 4, vehicle
        if len(weak):
            assert weak["battery_voltage"].mean() < normal["battery_voltage"].median() - 0.4, vehicle


def test_warm_start_rate_controls_engine_warm_up() -> None:
    cold = generate_telemetry(num_vehicles=5, events_per_vehicle=400, anomaly_rate=0.0, seed=2, warm_start_rate=0.0)
    warm = generate_telemetry(num_vehicles=5, events_per_vehicle=400, anomaly_rate=0.0, seed=2, warm_start_rate=1.0)
    first_cold = cold.groupby("vehicle_id")["engine_temperature"].first()
    first_warm = warm.groupby("vehicle_id")["engine_temperature"].first()
    assert first_cold.max() < 60 < 80 < first_warm.min()


def test_demo_fleet_stream_is_contiguous_and_deterministic() -> None:
    whole = demo_fleet_events(seed=5, start=0, n=200, anomaly_rate=0.05, drift_rate=3.0)
    pieces = pd.concat(
        [demo_fleet_events(seed=5, start=s, n=50, anomaly_rate=0.05, drift_rate=3.0) for s in range(0, 200, 50)],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(whole, pieces)
    assert whole["vehicle_id"].nunique() == 4
    assert pd.to_datetime(whole["event_timestamp"], utc=True).is_monotonic_increasing
    assert whole["engine_temperature"].median() > 80  # warm engines from the first event
    full = demo_fleet_events(seed=5, start=0, n=DEMO_STREAM_LENGTH - 1, anomaly_rate=0.05, drift_rate=3.0)
    assert set(DRIFT_ANOMALY_TYPES) <= set(full[ANOMALY_TYPE_COLUMN])
    for bad in ((-1, 10), (DEMO_STREAM_LENGTH - 5, 10), (0, 0)):
        with pytest.raises(ValueError):
            demo_fleet_events(seed=5, start=bad[0], n=bad[1], anomaly_rate=0.05, drift_rate=3.0)


def test_streaming_files_contain_expected_faults(tmp_path) -> None:
    events = generate_telemetry(num_vehicles=2, events_per_vehicle=500, anomaly_rate=0.05, seed=3)
    summary = write_streaming_files(events, tmp_path, num_files=3, malformed_rate=0.05, duplicate_rate=0.02, seed=5)
    files = sorted(tmp_path.glob("*.jsonl"))
    assert [f.name for f in files] == summary.files
    assert not list(tmp_path.glob(".*"))  # no temp files left behind
    lines = [line for f in files for line in f.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == summary.total_lines == len(events) + summary.duplicate_events
    assert summary.malformed_events == 50 and summary.duplicate_events == 20
    assert sum(summary.malformed_by_kind.values()) == summary.malformed_events

    result = validate_lines(lines)
    assert len(result.invalid) == summary.malformed_events
    assert len(result.valid) == summary.clean_events + summary.duplicate_events
    assert result.valid["event_id"].duplicated().sum() == summary.duplicate_events
    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert all(LABEL_COLUMN not in record for record in parsed)  # labels never leak into streaming input
