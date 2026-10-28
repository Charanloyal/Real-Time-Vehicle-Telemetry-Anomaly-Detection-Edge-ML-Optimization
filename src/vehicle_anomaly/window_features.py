"""Causal per-vehicle history ("window") features with resumable state.

Gradual faults (slowly rising engine temperature, a degrading battery) are hard to see in a single
reading because each vehicle has its own normal baseline. These features compare every event with
exponentially weighted moving averages (EWMA) of the same vehicle's *earlier* events:

    temp_dev       = engine_temperature - long EWMA of engine_temperature (before this event)
    temp_trend     = short EWMA - long EWMA of engine_temperature (before this event)
    battery_dev    = battery_voltage - long EWMA of battery_voltage
    battery_trend  = short EWMA - long EWMA of battery_voltage
    history_events = number of earlier events seen for the vehicle (capped)

Half-lives are measured in events (the simulator emits one event per second). The EWMA recursion is
fully described by a few numbers per vehicle (:class:`VehicleState`), so a stream can be processed in
any number of chunks - training (whole history), streaming micro-batches (state checkpointed per
batch) and the ``/predict-stream`` API (state returned to the client) - with identical results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from vehicle_anomaly.schema import TIMESTAMP_FIELD, WINDOW_FEATURES

LONG_HALFLIFE_EVENTS = 300
SHORT_HALFLIFE_EVENTS = 15
HISTORY_CAP = 1000
_SIGNALS = {"temp": "engine_temperature", "battery": "battery_voltage"}


def _alpha(halflife: float) -> float:
    return 1.0 - math.exp(math.log(0.5) / halflife)


ALPHA_LONG = _alpha(LONG_HALFLIFE_EVENTS)
ALPHA_SHORT = _alpha(SHORT_HALFLIFE_EVENTS)


@dataclass(frozen=True)
class VehicleState:
    """EWMA values after the last processed event of one vehicle."""

    temp_long: float
    temp_short: float
    battery_long: float
    battery_short: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VehicleState:
        return cls(
            temp_long=float(data["temp_long"]),
            temp_short=float(data["temp_short"]),
            battery_long=float(data["battery_long"]),
            battery_short=float(data["battery_short"]),
            count=int(data["count"]),
        )


StateMap = dict[str, VehicleState]


def state_to_json(state: StateMap) -> dict[str, dict[str, float | int]]:
    return {vehicle: value.to_dict() for vehicle, value in state.items()}


def state_from_json(data: Mapping[str, Mapping[str, Any]] | None) -> StateMap:
    return {str(vehicle): VehicleState.from_dict(value) for vehicle, value in (data or {}).items()}


def _ewm_previous(
    vehicles: np.ndarray, values: np.ndarray, seeds: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """For rows grouped contiguously by vehicle: EWMA *before* each row, and the EWMA after each row.

    ``seeds`` holds the EWMA value before each vehicle's first row in this chunk (state or, for a new
    vehicle, its first reading so that the first deviation is 0).
    """
    starts = np.r_[True, vehicles[1:] != vehicles[:-1]]
    group_ids = np.cumsum(starts) - 1
    # Prepend each group's seed as an extra observation: with adjust=False the EWMA of the augmented
    # series continues exactly from the seed.
    aug_values = np.insert(values, np.flatnonzero(starts), seeds)
    aug_groups = np.insert(group_ids, np.flatnonzero(starts), np.arange(len(seeds)))
    ewm = (
        pd.Series(aug_values)
        .groupby(aug_groups, sort=False)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
        .to_numpy()
    )
    is_data = np.ones(len(aug_values), dtype=bool)
    is_data[np.flatnonzero(starts) + np.arange(len(seeds))] = False
    data_positions = np.flatnonzero(is_data)
    return ewm[data_positions - 1], ewm[data_positions]


def compute_window_features(
    frame: pd.DataFrame, state: Mapping[str, VehicleState] | None = None
) -> tuple[pd.DataFrame, StateMap]:
    """Window features aligned to ``frame.index`` plus the updated state.

    Rows are processed per vehicle in timestamp order (stable for ties); ``frame`` itself may be in
    any order. Vehicles absent from ``frame`` keep their previous state.
    """
    state = dict(state or {})
    if frame.empty:
        return pd.DataFrame(columns=list(WINDOW_FEATURES), index=frame.index, dtype="float64"), state

    timestamps = pd.to_datetime(frame[TIMESTAMP_FIELD], utc=True, format="ISO8601")
    order = (
        pd.DataFrame({"vehicle": frame["vehicle_id"].astype(str).to_numpy(), "ts": timestamps.to_numpy()})
        .sort_values(["vehicle", "ts"], kind="stable")
        .index.to_numpy()
    )
    vehicles = frame["vehicle_id"].astype(str).to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, vehicles[1:] != vehicles[:-1]])
    first_vehicles = vehicles[starts]
    group_sizes = np.diff(np.r_[starts, len(vehicles)])
    prior = [state.get(v) for v in first_vehicles]

    outputs: dict[str, np.ndarray] = {}
    last_values: dict[str, np.ndarray] = {}
    for key, column in _SIGNALS.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)[order]
        values = pd.Series(values).groupby(np.repeat(np.arange(len(starts)), group_sizes)).ffill().to_numpy()
        first_values = values[starts]
        long_seed = np.array(
            [getattr(p, f"{key}_long") if p else first_values[i] for i, p in enumerate(prior)], dtype=np.float64
        )
        short_seed = np.array(
            [getattr(p, f"{key}_short") if p else first_values[i] for i, p in enumerate(prior)], dtype=np.float64
        )
        # A reading that is missing even after forward fill inherits the previous EWMA (no update).
        missing = np.isnan(values)
        if missing.any():
            fill = np.repeat(long_seed, group_sizes)
            values = np.where(missing, fill, values)
        long_prev, long_after = _ewm_previous(vehicles, values, long_seed, ALPHA_LONG)
        short_prev, short_after = _ewm_previous(vehicles, values, short_seed, ALPHA_SHORT)
        outputs[f"{key}_dev"] = values - long_prev
        outputs[f"{key}_trend"] = short_prev - long_prev
        last_idx = starts + group_sizes - 1
        last_values[f"{key}_long"] = long_after[last_idx]
        last_values[f"{key}_short"] = short_after[last_idx]

    prior_counts = np.array([p.count if p else 0 for p in prior], dtype=np.int64)
    within = np.arange(len(vehicles)) - np.repeat(starts, group_sizes)
    history = np.repeat(prior_counts, group_sizes) + within

    sorted_features = np.column_stack(
        [
            outputs["temp_dev"],
            outputs["temp_trend"],
            outputs["battery_dev"],
            outputs["battery_trend"],
            np.minimum(history, HISTORY_CAP).astype(np.float64),
        ]
    )
    aligned = np.empty_like(sorted_features)
    aligned[order] = sorted_features  # undo the sort by position (frame.index may contain duplicates)
    features = pd.DataFrame(aligned, columns=list(WINDOW_FEATURES), index=frame.index)

    for i, vehicle in enumerate(first_vehicles):
        state[str(vehicle)] = VehicleState(
            temp_long=float(last_values["temp_long"][i]),
            temp_short=float(last_values["temp_short"][i]),
            battery_long=float(last_values["battery_long"][i]),
            battery_short=float(last_values["battery_short"][i]),
            count=int(prior_counts[i] + group_sizes[i]),
        )
    return features, state
