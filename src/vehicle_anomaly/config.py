"""Typed application configuration loaded from YAML with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

CONFIG_ENV_VAR = "VTAD_CONFIG"
MODEL_DIR_ENV_VAR = "VTAD_MODEL_DIR"
LOG_LEVEL_ENV_VAR = "LOG_LEVEL"
INFERENCE_THREADS_ENV_VAR = "VTAD_INFERENCE_THREADS"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class PathsConfig:
    raw_data: Path
    processed_dir: Path
    streaming_input: Path
    streaming_ground_truth: Path
    streaming_output: Path
    checkpoint_dir: Path
    model_dir: Path
    reports_dir: Path

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


@dataclass(frozen=True)
class DataGenerationConfig:
    num_vehicles: int = 100
    events_per_vehicle: int = 2000
    anomaly_rate: float = 0.05
    hard_braking_rate: float = 0.004
    event_interval_seconds: float = 1.0
    start_time: str = "2026-01-01T00:00:00+00:00"
    base_latitude: float = 40.0
    base_longitude: float = -95.0


@dataclass(frozen=True)
class StreamingDataConfig:
    num_vehicles: int = 50
    num_files: int = 20
    events_per_file: int = 5000
    anomaly_rate: float = 0.05
    malformed_rate: float = 0.01
    duplicate_rate: float = 0.005
    seed_offset: int = 1000


@dataclass(frozen=True)
class TrainingConfig:
    validation_size: float = 0.15
    test_size: float = 0.15
    models: tuple[str, ...] = ("random_forest", "xgboost")
    selection_metric: str = "average_precision"
    random_forest: dict[str, Any] = field(default_factory=dict)
    xgboost: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    model: str | None = None
    single_iterations: int = 2000
    warmup_iterations: int = 100
    batch_size: int = 1000
    batch_iterations: int = 20


@dataclass(frozen=True)
class StreamingConfig:
    engine: str = "auto"
    trigger: str = "available_now"
    processing_interval_seconds: int = 5
    max_runtime_seconds: int | None = None
    max_files_per_trigger: int = 2
    output_format: str = "parquet"
    spark_master: str = "local[*]"
    shuffle_partitions: int = 4


@dataclass(frozen=True)
class ApiConfig:
    max_batch_size: int = 1000
    default_port: int = 8000
    inference_threads: int | None = None


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    json: bool = True


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    random_seed: int
    paths: PathsConfig
    data_generation: DataGenerationConfig
    streaming_data: StreamingDataConfig
    training: TrainingConfig
    benchmark: BenchmarkConfig
    streaming: StreamingConfig
    api: ApiConfig
    logging: LoggingConfig


T = TypeVar("T")


def _build_section(cls: type[T], data: dict[str, Any] | None, section: str) -> T:
    data = dict(data or {})
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{section}': {sorted(unknown)}")
    if "models" in data and data["models"] is not None:
        data["models"] = tuple(data["models"])
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"Invalid '{section}' section: {exc}") from exc


def _resolve_paths(raw: dict[str, Any], project_root: Path) -> PathsConfig:
    required = {f.name for f in fields(PathsConfig)}
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Missing path(s) in 'paths': {sorted(missing)}")
    unknown = set(raw) - required
    if unknown:
        raise ConfigError(f"Unknown key(s) in 'paths': {sorted(unknown)}")
    resolved = {}
    for name, value in raw.items():
        path = Path(value).expanduser()
        resolved[name] = path if path.is_absolute() else (project_root / path).resolve()
    model_dir_override = os.environ.get(MODEL_DIR_ENV_VAR)
    if model_dir_override:
        resolved["model_dir"] = Path(model_dir_override).expanduser().resolve()
    return PathsConfig(**resolved)


def default_config_path() -> Path:
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path.cwd() / DEFAULT_CONFIG_RELATIVE).resolve()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration. Relative paths resolve against the folder containing ``configs/``."""
    config_path = Path(path).expanduser().resolve() if path else default_config_path()
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}. Run from the project root or set {CONFIG_ENV_VAR}."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {config_path} must contain a mapping")

    project_root = (
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    )
    logging_cfg = _build_section(LoggingConfig, raw.get("logging"), "logging")
    level_override = os.environ.get(LOG_LEVEL_ENV_VAR)
    if level_override:
        logging_cfg = LoggingConfig(level=level_override.upper(), json=logging_cfg.json)
    api_cfg = _build_section(ApiConfig, raw.get("api"), "api")
    threads_override = os.environ.get(INFERENCE_THREADS_ENV_VAR)
    if threads_override:
        try:
            api_cfg = replace(api_cfg, inference_threads=int(threads_override))
        except ValueError as exc:
            raise ConfigError(f"{INFERENCE_THREADS_ENV_VAR} must be an integer") from exc

    config = AppConfig(
        project_root=project_root,
        random_seed=int(raw.get("random_seed", 42)),
        paths=_resolve_paths(raw.get("paths") or {}, project_root),
        data_generation=_build_section(DataGenerationConfig, raw.get("data_generation"), "data_generation"),
        streaming_data=_build_section(StreamingDataConfig, raw.get("streaming_data"), "streaming_data"),
        training=_build_section(TrainingConfig, raw.get("training"), "training"),
        benchmark=_build_section(BenchmarkConfig, raw.get("benchmark"), "benchmark"),
        streaming=_build_section(StreamingConfig, raw.get("streaming"), "streaming"),
        api=api_cfg,
        logging=logging_cfg,
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.streaming.engine not in {"auto", "spark", "local"}:
        raise ConfigError("streaming.engine must be one of: auto, spark, local")
    if config.streaming.trigger not in {"available_now", "processing_time"}:
        raise ConfigError("streaming.trigger must be one of: available_now, processing_time")
    if config.streaming.output_format not in {"parquet", "csv"}:
        raise ConfigError("streaming.output_format must be parquet or csv")
    split_total = config.training.validation_size + config.training.test_size
    if not 0 < split_total < 1:
        raise ConfigError("training.validation_size + training.test_size must be between 0 and 1")
    for name, rate in {
        "data_generation.anomaly_rate": config.data_generation.anomaly_rate,
        "streaming_data.anomaly_rate": config.streaming_data.anomaly_rate,
        "streaming_data.malformed_rate": config.streaming_data.malformed_rate,
        "streaming_data.duplicate_rate": config.streaming_data.duplicate_rate,
    }.items():
        if not 0 <= rate < 1:
            raise ConfigError(f"{name} must be in [0, 1)")
    if config.api.max_batch_size < 1:
        raise ConfigError("api.max_batch_size must be >= 1")
    if config.api.inference_threads is not None and config.api.inference_threads < 1:
        raise ConfigError("api.inference_threads must be >= 1 or null")
