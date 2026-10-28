# Data

**All data in this project is synthetic.** It is produced by
[`vehicle_anomaly.data_generator`](../src/vehicle_anomaly/data_generator.py). No real vehicle, fleet or
production telemetry is used, and no public dataset is integrated.

| Path | Content | In Git |
|---|---|---|
| `raw/telemetry.parquet` | Labelled training data (default 100 vehicles x 2,000 events = 200,000 rows) | No (regenerate) |
| `raw/streaming_ground_truth.parquet` | Labels for the streaming input, used only to score the streaming run | No |
| `processed/test.parquet`, `processed/splits.json` | Held-out test split and the vehicle split written by training | No |
| `streaming_input/*.jsonl` | Streaming demo input (default 20 files x 5,000 events, plus 1% malformed and 0.5% duplicate lines) | No |
| `streaming_output/` | Predictions, quarantined invalid events and per-batch metrics | No |
| `checkpoints/` | Streaming checkpoints (`spark/` or `local/`) | No |
| `sample/sample_events.json` | Five example events for API and CLI demos | Yes |

Regenerate everything (about 5 seconds on the development laptop):

```powershell
python scripts/generate_data.py            # --target training | streaming | all
```

The output is fully determined by `random_seed` in `configs/config.yaml`.

## Generator design

- Each vehicle is simulated independently from its own `SeedSequence` child. It switches between stopped, city
  and highway speed regimes with first-order dynamics, gear-based RPM, fuel burn and a GPS track.
- A recording starts either with a cold engine (warm-up curve from ambient temperature) or mid-trip with the
  engine already warm (`warm_start_rate`, default 50%).
- Engine operating temperature and battery health vary per vehicle. One vehicle's healthy readings can
  therefore look like another vehicle's mild fault. This is why splits are grouped by vehicle.
- About 0.4% of events are legitimate emergency braking (normal label), which overlaps with harsh deceleration.
- Single-event faults (default 5% of events) are labelled with `anomaly_type`: `overheating`, `rpm_anomaly`
  (stall at speed or over-revving), `harsh_acceleration`, `battery_voltage` (low or high) and
  `sensor_combination` (high brake pressure while accelerating).
- Gradual faults (`drift_episodes_per_1000_events`, default 0.3): `gradual_overheating` (+10–22 °C over 90–240
  events) and `battery_degradation` (−1.0–1.8 V over 120–300 events). Each is a linear ramp away from the
  vehicle's own baseline that starts after a baseline exists. Only the part of the ramp beyond 4 °C or 0.5 V is
  labelled anomalous.
- Streaming files are written atomically (hidden temp file, then rename). They contain no labels. Injected faults
  are truncated JSON, missing fields, nulls, non-numeric values, out-of-range values and invalid timestamps.
- The live demo uses `demo_fleet_events`: a deterministic 4-vehicle, 4,000-event stream (warm starts, concurrent
  vehicles), cached per seed and served in contiguous slices.

Units: speed km/h, engine_rpm rev/min, engine_temperature °C, fuel_level %, battery_voltage V,
brake_pressure bar, acceleration m/s², latitude/longitude decimal degrees.
