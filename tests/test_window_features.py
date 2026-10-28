from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vehicle_anomaly.data_generator import generate_telemetry
from vehicle_anomaly.schema import WINDOW_FEATURES
from vehicle_anomaly.window_features import (
    HISTORY_CAP,
    VehicleState,
    compute_window_features,
    state_from_json,
    state_to_json,
)


@pytest.fixture(scope="module")
def stream() -> pd.DataFrame:
    df = generate_telemetry(
        num_vehicles=5,
        events_per_vehicle=1200,
        anomaly_rate=0.05,
        seed=4,
        drift_episodes_per_1000_events=2.0,
        max_start_offset_seconds=30,
    )
    return df.sort_values(["event_timestamp", "event_id"]).reset_index(drop=True)


def test_chunked_processing_matches_full_history(stream: pd.DataFrame) -> None:
    full, full_state = compute_window_features(stream)
    state = None
    parts = []
    for chunk in np.array_split(np.arange(len(stream)), 23):
        features, state = compute_window_features(stream.iloc[chunk], state)
        parts.append(features)
    chunked = pd.concat(parts).loc[stream.index]
    np.testing.assert_allclose(chunked.to_numpy(), full.to_numpy(), rtol=0, atol=1e-9)
    assert state == full_state


def test_input_order_and_duplicate_index_do_not_matter(stream: pd.DataFrame) -> None:
    full, _ = compute_window_features(stream)
    shuffled = stream.sample(frac=1, random_state=3)
    features, _ = compute_window_features(shuffled)
    np.testing.assert_array_equal(features.loc[stream.index].to_numpy(), full.to_numpy())
    duplicate_index = stream.iloc[:20].copy()
    duplicate_index.index = [7] * 20
    features, _ = compute_window_features(duplicate_index)
    np.testing.assert_array_equal(features.to_numpy(), full.iloc[:20].to_numpy())


def test_first_event_of_new_vehicle_has_zero_deviation(stream: pd.DataFrame) -> None:
    features, state = compute_window_features(stream.iloc[:1])
    assert list(features.columns) == list(WINDOW_FEATURES)
    assert features.iloc[0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    vehicle = stream.iloc[0]["vehicle_id"]
    assert state[vehicle].count == 1
    assert state[vehicle].temp_long == pytest.approx(stream.iloc[0]["engine_temperature"])


def test_history_counter_and_cap(stream: pd.DataFrame) -> None:
    features, state = compute_window_features(stream)
    per_vehicle = features.groupby(stream["vehicle_id"])["history_events"]
    assert (per_vehicle.min() == 0).all()
    assert features["history_events"].max() == HISTORY_CAP
    assert all(s.count == 1200 for s in state.values())


def test_deviation_tracks_gradual_drift() -> None:
    df = generate_telemetry(
        num_vehicles=6,
        events_per_vehicle=1500,
        anomaly_rate=0.0,
        seed=8,
        drift_episodes_per_1000_events=2.0,
        warm_start_rate=1.0,
    )
    features, _ = compute_window_features(df)
    overheating = df["anomaly_type"] == "gradual_overheating"
    degradation = df["anomaly_type"] == "battery_degradation"
    normal = df["anomaly_type"] == "normal"
    assert overheating.any() and degradation.any()
    assert features.loc[overheating, "temp_dev"].median() > 3.0 > features.loc[normal, "temp_dev"].abs().median()
    assert (
        features.loc[degradation, "battery_dev"].median() < -0.4 < -features.loc[normal, "battery_dev"].abs().median()
    )


def test_state_json_roundtrip_and_existing_vehicles_kept(stream: pd.DataFrame) -> None:
    _, state = compute_window_features(stream.iloc[:500])
    restored = state_from_json(state_to_json(state))
    assert restored == state
    other = {"OTHER-1": VehicleState(90.0, 90.0, 14.0, 14.0, 42)}
    _, merged = compute_window_features(stream.iloc[500:510], {**restored, **other})
    assert merged["OTHER-1"] == other["OTHER-1"]


def test_missing_readings_do_not_break_state(stream: pd.DataFrame) -> None:
    frame = stream.iloc[:50].copy()
    frame.loc[frame.index[10], "engine_temperature"] = np.nan
    features, state = compute_window_features(frame)
    assert np.isfinite(features.to_numpy()).all()
    assert all(np.isfinite([s.temp_long, s.temp_short]).all() for s in state.values())


def test_empty_frame() -> None:
    empty = pd.DataFrame(columns=["vehicle_id", "event_timestamp", "engine_temperature", "battery_voltage"])
    features, state = compute_window_features(empty, {"A": VehicleState(1, 1, 1, 1, 1)})
    assert features.empty and list(features.columns) == list(WINDOW_FEATURES)
    assert state == {"A": VehicleState(1, 1, 1, 1, 1)}
