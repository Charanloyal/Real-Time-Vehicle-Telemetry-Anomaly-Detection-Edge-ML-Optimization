"""Telemetry data contract shared by the generator, model, streaming pipeline and API.

Units: speed km/h, engine_rpm rev/min, engine_temperature deg C, fuel_level %, battery_voltage V,
brake_pressure bar, acceleration m/s^2, latitude/longitude decimal degrees.
"""

from __future__ import annotations

ID_FIELDS: tuple[str, ...] = ("event_id", "vehicle_id")
TIMESTAMP_FIELD = "event_timestamp"
NUMERIC_FIELDS: tuple[str, ...] = (
    "speed",
    "engine_rpm",
    "engine_temperature",
    "fuel_level",
    "battery_voltage",
    "brake_pressure",
    "acceleration",
    "latitude",
    "longitude",
)
REQUIRED_FIELDS: tuple[str, ...] = (*ID_FIELDS, TIMESTAMP_FIELD, *NUMERIC_FIELDS)

LABEL_COLUMN = "is_anomaly"
ANOMALY_TYPE_COLUMN = "anomaly_type"
NORMAL_TYPE = "normal"
ANOMALY_TYPES: tuple[str, ...] = (
    "overheating",
    "rpm_anomaly",
    "harsh_acceleration",
    "battery_voltage",
    "sensor_combination",
)

# Physically plausible sensor ranges. Values outside are rejected as invalid input (sensor faults
# or corrupt payloads). They are intentionally much wider than the anomaly ranges so genuine
# anomalies are scored by the model rather than discarded by validation.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "speed": (0.0, 300.0),
    "engine_rpm": (0.0, 10000.0),
    "engine_temperature": (-40.0, 200.0),
    "fuel_level": (0.0, 100.0),
    "battery_voltage": (0.0, 30.0),
    "brake_pressure": (0.0, 300.0),
    "acceleration": (-30.0, 30.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
}

# Model inputs. Identifiers, timestamps, GPS position and labels are deliberately excluded to avoid
# leakage (vehicle identity / time) and location memorisation.
MODEL_INPUT_FIELDS: tuple[str, ...] = (
    "speed",
    "engine_rpm",
    "engine_temperature",
    "fuel_level",
    "battery_voltage",
    "brake_pressure",
    "acceleration",
)
DERIVED_FEATURES: tuple[str, ...] = ("rpm_per_speed", "abs_acceleration", "brake_accel_product")
FEATURE_COLUMNS: tuple[str, ...] = (*MODEL_INPUT_FIELDS, *DERIVED_FEATURES)
