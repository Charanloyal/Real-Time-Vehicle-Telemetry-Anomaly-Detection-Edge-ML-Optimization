"""Synthetic vehicle telemetry generator.

All data produced here is synthetic. Each vehicle follows a simple driving simulation (speed regimes
with first-order dynamics, gear-based RPM, engine warm-up or a warm mid-trip start, fuel burn, GPS
drift). Labelled anomalies are injected as single-event faults and as gradual drifts relative to the
vehicle's own baseline. Streaming files additionally contain malformed and duplicate events to
exercise validation and de-duplication.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from vehicle_anomaly.config import AppConfig, load_config
from vehicle_anomaly.logging_utils import configure_logging, get_logger
from vehicle_anomaly.schema import (
    ANOMALY_TYPE_COLUMN,
    DRIFT_ANOMALY_TYPES,
    LABEL_COLUMN,
    NORMAL_TYPE,
    NUMERIC_FIELDS,
    POINT_ANOMALY_TYPES,
    REQUIRED_FIELDS,
    VALID_RANGES,
)

logger = get_logger(__name__)

# Engine RPM gained per km/h in each gear (index = gear, 0 unused).
_GEAR_RPM_PER_KMH = np.array([0.0, 80.0, 50.0, 38.0, 30.0, 25.0, 21.0])
_ROUNDING = {
    "speed": 2,
    "engine_rpm": 0,
    "engine_temperature": 2,
    "fuel_level": 2,
    "battery_voltage": 3,
    "brake_pressure": 2,
    "acceleration": 3,
    "latitude": 6,
    "longitude": 6,
}
MALFORMED_KINDS: tuple[str, ...] = (
    "malformed_json",
    "missing_field",
    "null_value",
    "non_numeric",
    "out_of_range",
    "invalid_timestamp",
)


def _speed_profile(rng: np.random.Generator, n: int, interval: float, hard_braking_rate: float) -> np.ndarray:
    targets = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(60, 400))
        regime = rng.choice(3, p=[0.15, 0.55, 0.30])  # stopped, city, highway
        targets[i : i + length] = (0.0, rng.uniform(25, 60), rng.uniform(85, 125))[regime]
        i += length

    gain = rng.uniform(0.03, 0.07) * interval
    noise = rng.normal(0.0, 0.8, n)
    hard_brake = rng.random(n) < hard_braking_rate
    hard_brake_drop = rng.uniform(12.0, 20.0, n) * interval
    speed = np.empty(n)
    current = rng.uniform(0, 30)
    for t in range(n):
        current = current + gain * (targets[t] - current) + noise[t]
        if hard_brake[t] and current > 40:
            current -= hard_brake_drop[t]
        current = max(0.0, current)
        speed[t] = current
    return speed


def _simulate_vehicle(
    rng: np.random.Generator,
    n: int,
    interval: float,
    hard_braking_rate: float,
    base_lat: float,
    base_lon: float,
    warm_start: bool,
) -> dict[str, np.ndarray]:
    speed = _speed_profile(rng, n, interval, hard_braking_rate)
    acceleration = np.diff(speed, prepend=speed[0]) / 3.6 / interval

    gear = np.clip(np.ceil(speed / 22.0), 1, 6).astype(int)
    rpm = np.where(
        speed < 1.0,
        rng.normal(780, 25, n),
        800 + speed * _GEAR_RPM_PER_KMH[gear] + 180 * np.clip(acceleration, 0, None),
    ) + rng.normal(0, 90, n)
    rpm = np.clip(rpm, 600, None)

    # Per-vehicle baselines (engine condition, battery health) create overlap between a healthy
    # vehicle's tail and another vehicle's mild anomaly, so the model must generalise across vehicles.
    operating = rng.uniform(84, 97)
    # A recording starts either with a cold engine or mid-trip with the engine already warm.
    ambient = operating if warm_start else rng.uniform(5, 35)
    elapsed = np.arange(n) * interval
    temperature = (
        operating
        - (operating - ambient) * np.exp(-elapsed / rng.uniform(200, 400))
        + 1.2 * (rpm - 800) / 1000
        + rng.normal(0, 0.7, n)
    )

    consumption = (0.00004 * speed + 0.0003) * interval
    fuel = np.clip(rng.uniform(30, 100) - np.cumsum(consumption), 2, 100)

    battery = rng.normal(rng.uniform(13.5, 14.5), 0.15, n) - 0.15 * (speed < 1.0)

    braking = np.clip(-acceleration, 0, None)
    brake = np.where(
        acceleration < -0.3,
        braking * 14 + rng.normal(0, 2, n),
        np.abs(rng.normal(0, 1.0, n)),
    )
    hard = acceleration < -3.0
    brake[hard] = rng.uniform(50, 90, hard.sum())
    brake = np.clip(brake, 0, None)

    heading = np.cumsum(rng.normal(0, 0.05, n)) + rng.uniform(0, 2 * np.pi)
    distance_m = speed / 3.6 * interval
    lat0 = base_lat + rng.uniform(-0.5, 0.5)
    lon0 = base_lon + rng.uniform(-0.5, 0.5)
    latitude = lat0 + np.cumsum(distance_m * np.cos(heading)) / 111_320
    longitude = lon0 + np.cumsum(distance_m * np.sin(heading)) / (111_320 * np.cos(np.radians(lat0)))

    return {
        "speed": speed,
        "engine_rpm": rpm,
        "engine_temperature": temperature,
        "fuel_level": fuel,
        "battery_voltage": battery,
        "brake_pressure": brake,
        "acceleration": acceleration,
        "latitude": latitude,
        "longitude": longitude,
    }


def _inject_drift(
    rng: np.random.Generator,
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    episodes_per_1000_events: float,
    warm_start: bool,
) -> None:
    """Gradual faults: a linear ramp away from the vehicle's baseline.

    Only the part of a ramp that has moved clearly away from the baseline (>= 4 deg C or <= -0.5 V) is
    labelled anomalous; the first minutes of a drift are indistinguishable from normal noise. Drifts start
    after the engine has warmed up (cold starts) or after a short history (warm starts).
    """
    n = len(labels)
    min_start = 300 if warm_start else 1200
    count = int(rng.poisson(episodes_per_1000_events * n / 1000)) if episodes_per_1000_events > 0 else 0
    for _ in range(count):
        kind = DRIFT_ANOMALY_TYPES[int(rng.integers(0, len(DRIFT_ANOMALY_TYPES)))]
        if kind == "gradual_overheating":
            duration, peak, column, label_at, sign = (
                int(rng.integers(90, 241)),
                rng.uniform(10, 22),
                "engine_temperature",
                4.0,
                1.0,
            )
        else:
            duration, peak, column, label_at, sign = (
                int(rng.integers(120, 301)),
                rng.uniform(1.0, 1.8),
                "battery_voltage",
                0.5,
                -1.0,
            )
        if n - duration <= min_start:
            continue
        start = int(rng.integers(min_start, n - duration))
        idx = np.arange(start, start + duration)
        offset = peak * (idx - start + 1) / duration
        values[column][idx] += sign * offset
        flagged = idx[offset >= label_at]
        labels[flagged] = kind


def _inject_anomalies(
    rng: np.random.Generator,
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    anomaly_rate: float,
    interval: float,
) -> None:
    n = len(values["speed"])
    candidates = np.flatnonzero(rng.random(n) < anomaly_rate)
    candidates = candidates[candidates > 0]
    kinds = rng.integers(0, len(POINT_ANOMALY_TYPES), len(candidates))
    for idx, kind_idx in zip(candidates.tolist(), kinds.tolist(), strict=True):
        kind = POINT_ANOMALY_TYPES[kind_idx]
        labels[idx] = kind
        prev_speed = values["speed"][idx - 1]
        if kind == "overheating":
            values["engine_temperature"][idx] = rng.uniform(98, 125)
        elif kind == "rpm_anomaly":
            if values["speed"][idx] > 30:
                values["engine_rpm"][idx] = rng.uniform(150, 700)  # stall-like reading while moving
            else:
                values["engine_rpm"][idx] = rng.uniform(4000, 7500)  # over-revving at low speed
        elif kind == "harsh_acceleration":
            magnitude = rng.uniform(3.5, 9.0)
            accel = magnitude if (prev_speed < 20 or rng.random() < 0.5) else -magnitude
            values["acceleration"][idx] = accel
            values["speed"][idx] = max(0.0, prev_speed + accel * 3.6 * interval)
            if accel < 0:
                values["brake_pressure"][idx] = rng.uniform(70, 160)
        elif kind == "battery_voltage":
            values["battery_voltage"][idx] = rng.uniform(11.0, 13.2) if rng.random() < 0.5 else rng.uniform(14.9, 16.5)
        elif kind == "sensor_combination":
            accel = rng.uniform(0.8, 3.0)
            values["acceleration"][idx] = accel
            values["speed"][idx] = prev_speed + accel * 3.6 * interval
            values["brake_pressure"][idx] = rng.uniform(20, 80)


def generate_telemetry(
    num_vehicles: int,
    events_per_vehicle: int,
    anomaly_rate: float,
    seed: int,
    start_time: str = "2026-01-01T00:00:00+00:00",
    event_interval_seconds: float = 1.0,
    hard_braking_rate: float = 0.004,
    base_latitude: float = 40.0,
    base_longitude: float = -95.0,
    vehicle_prefix: str = "VEH",
    drift_episodes_per_1000_events: float = 0.0,
    warm_start_rate: float = 0.5,
    max_start_offset_seconds: int = 3600,
) -> pd.DataFrame:
    """Generate a labelled telemetry DataFrame. Same arguments always produce identical output."""
    if num_vehicles < 1 or events_per_vehicle < 2:
        raise ValueError("num_vehicles must be >= 1 and events_per_vehicle >= 2")
    start = pd.Timestamp(start_time)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    children = np.random.SeedSequence(seed).spawn(num_vehicles)

    frames = []
    for vehicle_idx, child in enumerate(children):
        rng = np.random.default_rng(child)
        warm_start = bool(rng.random() < warm_start_rate)
        values = _simulate_vehicle(
            rng,
            events_per_vehicle,
            event_interval_seconds,
            hard_braking_rate,
            base_latitude,
            base_longitude,
            warm_start,
        )
        anomaly_types = np.full(events_per_vehicle, NORMAL_TYPE, dtype=object)
        _inject_drift(rng, values, anomaly_types, drift_episodes_per_1000_events, warm_start)
        _inject_anomalies(rng, values, anomaly_types, anomaly_rate, event_interval_seconds)
        offset_seconds = float(rng.integers(0, max_start_offset_seconds + 1))
        timestamps = start + pd.to_timedelta(
            offset_seconds + np.arange(events_per_vehicle) * event_interval_seconds, unit="s"
        )
        vehicle_id = f"{vehicle_prefix}-{vehicle_idx:04d}"
        frame = pd.DataFrame(
            {
                "event_id": [f"{vehicle_id}-{t:07d}" for t in range(events_per_vehicle)],
                "vehicle_id": vehicle_id,
                "event_timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z",
                **{name: np.round(values[name], _ROUNDING[name]) for name in NUMERIC_FIELDS},
                LABEL_COLUMN: (anomaly_types != NORMAL_TYPE).astype(np.int8),
                ANOMALY_TYPE_COLUMN: anomaly_types.astype(str),
            }
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    for name, (low, high) in VALID_RANGES.items():
        df[name] = df[name].clip(low, high)
    return df


DEMO_FLEET_VEHICLES = 4
DEMO_EVENTS_PER_VEHICLE = 1000
DEMO_STREAM_LENGTH = DEMO_FLEET_VEHICLES * DEMO_EVENTS_PER_VEHICLE


@lru_cache(maxsize=16)
def _demo_fleet(seed: int, anomaly_rate: float, drift_rate: float) -> pd.DataFrame:
    df = generate_telemetry(
        num_vehicles=DEMO_FLEET_VEHICLES,
        events_per_vehicle=DEMO_EVENTS_PER_VEHICLE,
        anomaly_rate=anomaly_rate,
        seed=seed,
        vehicle_prefix=f"LIVE{seed % 10000:04d}",
        drift_episodes_per_1000_events=drift_rate,
        warm_start_rate=1.0,
        max_start_offset_seconds=3,
    )
    return df.sort_values(["event_timestamp", "vehicle_id"], kind="stable").reset_index(drop=True)


def demo_fleet_events(seed: int, start: int, n: int, anomaly_rate: float, drift_rate: float) -> pd.DataFrame:
    """A contiguous slice of a deterministic 4-vehicle live-demo stream (time-ordered, labelled).

    Vehicles start mid-trip with warm engines and drive concurrently, so consecutive slices form a
    realistic stream for per-vehicle window features. The whole fleet is generated once per
    ``(seed, anomaly_rate, drift_rate)`` and cached.
    """
    if n < 1 or start < 0 or start + n > DEMO_STREAM_LENGTH:
        raise ValueError(
            f"Requested events [{start}, {start + n}) are outside the demo stream (0-{DEMO_STREAM_LENGTH})"
        )
    return _demo_fleet(seed, anomaly_rate, drift_rate).iloc[start : start + n].reset_index(drop=True)


@dataclass
class StreamingFilesSummary:
    files: list[str] = field(default_factory=list)
    total_lines: int = 0
    clean_events: int = 0
    duplicate_events: int = 0
    malformed_events: int = 0
    malformed_by_kind: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": len(self.files),
            "total_lines": self.total_lines,
            "clean_events": self.clean_events,
            "duplicate_events": self.duplicate_events,
            "malformed_events": self.malformed_events,
            "malformed_by_kind": self.malformed_by_kind,
        }


def _corrupt(record: dict[str, Any], kind: str, rng: np.random.Generator) -> str:
    record = dict(record)
    if kind == "malformed_json":
        text = json.dumps(record)
        return text[: int(rng.integers(5, len(text) - 5))]
    if kind == "missing_field":
        record.pop(REQUIRED_FIELDS[int(rng.integers(0, len(REQUIRED_FIELDS)))])
    elif kind == "null_value":
        record[NUMERIC_FIELDS[int(rng.integers(0, len(NUMERIC_FIELDS)))]] = None
    elif kind == "non_numeric":
        record[NUMERIC_FIELDS[int(rng.integers(0, len(NUMERIC_FIELDS)))]] = rng.choice(["N/A", "error", "--"])
    elif kind == "out_of_range":
        name = NUMERIC_FIELDS[int(rng.integers(0, len(NUMERIC_FIELDS)))]
        low, high = VALID_RANGES[name]
        record[name] = high + abs(high - low) if rng.random() < 0.5 else low - abs(high - low)
    elif kind == "invalid_timestamp":
        record["event_timestamp"] = "not-a-timestamp"
    return json.dumps(record)


def _write_atomic(path: Path, lines: Sequence[str]) -> None:
    # Write to a hidden temp file first; Spark's file source ignores names starting with '.'.
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def write_streaming_files(
    events: pd.DataFrame,
    output_dir: Path,
    num_files: int,
    malformed_rate: float,
    duplicate_rate: float,
    seed: int,
    file_prefix: str = "telemetry",
) -> StreamingFilesSummary:
    """Split events (time-ordered) into JSON Lines files with injected malformed and duplicate rows."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = events.sort_values(["event_timestamp", "event_id"], kind="stable").reset_index(drop=True)
    records = cast(list[dict[str, Any]], events.loc[:, list(REQUIRED_FIELDS)].to_dict("records"))

    n = len(records)
    order = rng.permutation(n)
    n_malformed = round(n * malformed_rate)
    n_duplicates = round(n * duplicate_rate)
    malformed_idx = set(order[:n_malformed].tolist())
    duplicate_idx = set(order[n_malformed : n_malformed + n_duplicates].tolist())
    kinds: Counter[str] = Counter()

    summary = StreamingFilesSummary()
    for file_no, chunk in enumerate(np.array_split(np.arange(n), num_files), start=1):
        lines: list[str] = []
        for idx in chunk.tolist():
            if idx in malformed_idx:
                kind = MALFORMED_KINDS[int(rng.integers(0, len(MALFORMED_KINDS)))]
                kinds[kind] += 1
                lines.append(_corrupt(records[idx], kind, rng))
                continue
            line = json.dumps(records[idx])
            lines.append(line)
            if idx in duplicate_idx:
                lines.append(line)
        path = output_dir / f"{file_prefix}_{file_no:05d}.jsonl"
        _write_atomic(path, lines)
        summary.files.append(path.name)
        summary.total_lines += len(lines)

    summary.malformed_events = n_malformed
    summary.duplicate_events = n_duplicates
    summary.clean_events = n - n_malformed
    summary.malformed_by_kind = dict(sorted(kinds.items()))
    return summary


def generate_training_data(config: AppConfig) -> pd.DataFrame:
    gen = config.data_generation
    df = generate_telemetry(
        num_vehicles=gen.num_vehicles,
        events_per_vehicle=gen.events_per_vehicle,
        anomaly_rate=gen.anomaly_rate,
        seed=config.random_seed,
        start_time=gen.start_time,
        event_interval_seconds=gen.event_interval_seconds,
        hard_braking_rate=gen.hard_braking_rate,
        base_latitude=gen.base_latitude,
        base_longitude=gen.base_longitude,
        drift_episodes_per_1000_events=gen.drift_episodes_per_1000_events,
        warm_start_rate=gen.warm_start_rate,
    )
    path = config.paths.raw_data
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(
        "Training data written",
        extra={"path": str(path), "rows": len(df), "anomalies": int(df[LABEL_COLUMN].sum())},
    )
    return df


def generate_streaming_data(config: AppConfig, clean_existing: bool = True) -> StreamingFilesSummary:
    gen = config.data_generation
    stream = config.streaming_data
    events_per_vehicle = max(2, -(-stream.num_files * stream.events_per_file // stream.num_vehicles))
    df = generate_telemetry(
        num_vehicles=stream.num_vehicles,
        events_per_vehicle=events_per_vehicle,
        anomaly_rate=stream.anomaly_rate,
        seed=config.random_seed + stream.seed_offset,
        start_time=gen.start_time,
        event_interval_seconds=gen.event_interval_seconds,
        hard_braking_rate=gen.hard_braking_rate,
        base_latitude=gen.base_latitude,
        base_longitude=gen.base_longitude,
        vehicle_prefix="STR",
        drift_episodes_per_1000_events=stream.drift_episodes_per_1000_events,
        warm_start_rate=gen.warm_start_rate,
    ).iloc[: stream.num_files * stream.events_per_file]

    input_dir = config.paths.streaming_input
    if clean_existing and input_dir.exists():
        for old in input_dir.glob("*.jsonl"):
            old.unlink()
    summary = write_streaming_files(
        df,
        input_dir,
        num_files=stream.num_files,
        malformed_rate=stream.malformed_rate,
        duplicate_rate=stream.duplicate_rate,
        seed=config.random_seed + stream.seed_offset + 1,
    )
    truth_path = config.paths.streaming_ground_truth
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, ["event_id", "vehicle_id", LABEL_COLUMN, ANOMALY_TYPE_COLUMN]].to_parquet(truth_path, index=False)
    logger.info("Streaming input written", extra={"directory": str(input_dir), **summary.to_dict()})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle telemetry.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--target", choices=["training", "streaming", "all"], default="all", help="Which dataset to create"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    configure_logging(config.logging.level, config.logging.json)
    if args.target in {"training", "all"}:
        generate_training_data(config)
    if args.target in {"streaming", "all"}:
        generate_streaming_data(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
