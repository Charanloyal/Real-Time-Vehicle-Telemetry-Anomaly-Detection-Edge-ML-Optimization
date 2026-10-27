from __future__ import annotations

import json
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from vehicle_anomaly.data_generator import generate_telemetry
from vehicle_anomaly.preprocessing import (
    TelemetryFeatureBuilder,
    build_preprocessor,
    compute_derived_features,
    split_by_vehicle,
)
from vehicle_anomaly.schema import FEATURE_COLUMNS, MODEL_INPUT_FIELDS
from vehicle_anomaly.validation import validate_lines, validate_record

VALID_EVENT = {
    "event_id": "VEH-0001-0000001",
    "vehicle_id": "VEH-0001",
    "event_timestamp": "2026-01-01T00:00:01.000Z",
    "speed": 50.0,
    "engine_rpm": 2400.0,
    "engine_temperature": 90.0,
    "fuel_level": 55.0,
    "battery_voltage": 14.2,
    "brake_pressure": 0.5,
    "acceleration": 0.4,
    "latitude": 40.1,
    "longitude": -95.2,
}


def test_derived_features_identical_for_numpy_and_pandas() -> None:
    frame = pd.DataFrame([VALID_EVENT, {**VALID_EVENT, "speed": 0.0, "acceleration": -2.0}])
    from_pandas = compute_derived_features(frame)
    from_numpy = compute_derived_features({k: frame[k].to_numpy() for k in MODEL_INPUT_FIELDS})
    for name, values in from_pandas.items():
        np.testing.assert_array_equal(np.asarray(values), from_numpy[name])
    assert from_numpy["rpm_per_speed"][0] == pytest.approx(2400 / 51)
    assert from_numpy["brake_accel_product"][1] == 0.0  # braking while decelerating is normal


def test_feature_builder_selects_and_orders_columns() -> None:
    frame = pd.DataFrame([VALID_EVENT])
    features = TelemetryFeatureBuilder().fit_transform(frame)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert "latitude" not in features.columns and "vehicle_id" not in features.columns


def test_feature_builder_rejects_missing_columns() -> None:
    with pytest.raises(KeyError, match="speed"):
        TelemetryFeatureBuilder().transform(pd.DataFrame([{k: v for k, v in VALID_EVENT.items() if k != "speed"}]))


def test_preprocessor_imputes_missing_values_with_training_medians() -> None:
    train = pd.DataFrame([{**VALID_EVENT, "fuel_level": value} for value in (10.0, 20.0, 90.0)])
    preprocessor = build_preprocessor().fit(train)
    transformed = preprocessor.transform(pd.DataFrame([{**VALID_EVENT, "fuel_level": None}]))
    assert transformed.shape == (1, len(FEATURE_COLUMNS))
    assert transformed[0, FEATURE_COLUMNS.index("fuel_level")] == 20.0


def test_split_by_vehicle_has_no_overlap_and_is_deterministic() -> None:
    df = generate_telemetry(num_vehicles=20, events_per_vehicle=50, anomaly_rate=0.05, seed=1)
    train, val, test = split_by_vehicle(df, 0.15, 0.15, seed=42)
    sets = [set(part["vehicle_id"]) for part in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(train) + len(val) + len(test) == len(df)
    assert len(sets[0]) == 14 and len(sets[1]) == 3 and len(sets[2]) == 3
    again = split_by_vehicle(df, 0.15, 0.15, seed=42)
    pd.testing.assert_frame_equal(test, again[2])


def test_split_rejects_invalid_sizes() -> None:
    df = generate_telemetry(num_vehicles=5, events_per_vehicle=10, anomaly_rate=0.0, seed=1)
    with pytest.raises(ValueError):
        split_by_vehicle(df, 0.6, 0.5, seed=1)


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ([1, 2, 3], "malformed_json"),
        ({k: v for k, v in VALID_EVENT.items() if k != "vehicle_id"}, "missing_field:vehicle_id"),
        ({**VALID_EVENT, "event_id": "   "}, "missing_field:event_id"),
        ({**VALID_EVENT, "battery_voltage": None}, "missing_field:battery_voltage"),
        ({**VALID_EVENT, "speed": "fast"}, "non_numeric:speed"),
        ({**VALID_EVENT, "engine_rpm": True}, "non_numeric:engine_rpm"),
        ({**VALID_EVENT, "fuel_level": "NaN"}, "non_numeric:fuel_level"),
        ({**VALID_EVENT, "event_timestamp": "yesterday"}, "invalid_timestamp"),
        ({**VALID_EVENT, "speed": 999.0}, "out_of_range:speed"),
        ({**VALID_EVENT, "acceleration": -45.0}, "out_of_range:acceleration"),
    ],
)
def test_validate_record_reasons(record: object, reason: str) -> None:
    assert validate_record(record) == (reason, None)


def test_validate_record_returns_typed_values() -> None:
    reason, typed = validate_record({**VALID_EVENT, "speed": "50.5", "event_id": 123})
    assert reason is None and typed is not None
    assert typed["speed"] == 50.5
    assert typed["event_id"] == "123"
    assert typed["event_timestamp"].tzinfo == timezone.utc


def test_validate_lines_handles_blank_and_corrupt_lines() -> None:
    lines = [json.dumps(VALID_EVENT), "", '{"event_id": "x", "speed": ', json.dumps({**VALID_EVENT, "speed": -1})]
    result = validate_lines(lines)
    assert result.input_rows == 3
    assert len(result.valid) == 1
    assert sorted(result.invalid["invalid_reason"]) == ["malformed_json", "out_of_range:speed"]
    assert str(result.valid["event_timestamp"].dtype) == "datetime64[ns, UTC]"
