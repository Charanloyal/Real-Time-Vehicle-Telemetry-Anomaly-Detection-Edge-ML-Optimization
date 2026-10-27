from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vehicle_anomaly.data_generator import generate_telemetry, write_streaming_files
from vehicle_anomaly.schema import (
    ANOMALY_TYPE_COLUMN,
    ANOMALY_TYPES,
    LABEL_COLUMN,
    NORMAL_TYPE,
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
    assert set(telemetry[ANOMALY_TYPE_COLUMN]) == {NORMAL_TYPE, *ANOMALY_TYPES}
    assert (telemetry[LABEL_COLUMN].astype(bool) == (telemetry[ANOMALY_TYPE_COLUMN] != NORMAL_TYPE)).all()


def test_anomalies_differ_from_normal_behaviour(telemetry: pd.DataFrame) -> None:
    normal = telemetry[telemetry[LABEL_COLUMN] == 0]
    overheating = telemetry[telemetry[ANOMALY_TYPE_COLUMN] == "overheating"]
    battery = telemetry[telemetry[ANOMALY_TYPE_COLUMN] == "battery_voltage"]
    assert overheating["engine_temperature"].median() > normal["engine_temperature"].quantile(0.99)
    assert normal["battery_voltage"].between(12.8, 15.2).mean() > 0.999
    assert not battery["battery_voltage"].between(13.25, 14.85).any()


def test_streaming_files_contain_expected_faults(tmp_path) -> None:
    events = generate_telemetry(num_vehicles=2, events_per_vehicle=500, anomaly_rate=0.05, seed=3)
    summary = write_streaming_files(
        events, tmp_path, num_files=3, malformed_rate=0.05, duplicate_rate=0.02, seed=5
    )
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
