"""Streaming pipeline entry point: PySpark Structured Streaming engine plus local fallback.

Spark pipeline (``engine=spark``)::

    text file source (JSON Lines, maxFilesPerTrigger)
      -> from_json (all fields as strings, PERMISSIVE + _corrupt_record)
      -> validation expressions (malformed / missing / non-numeric / timestamp / range)
      -> foreachBatch: quarantine invalid rows, de-duplicate, score with the trained model,
         write predictions + metrics (idempotent per batch id)
      -> checkpointLocation for recovery

Scoring happens on the driver inside ``foreachBatch`` with the model loaded once per process. That
is appropriate for a single-node demo; on a cluster the same ``AnomalyPredictor`` could be applied
with ``mapInPandas`` on executors instead.

When Java or PySpark is unavailable, ``engine=auto`` falls back to the pandas engine in
``local_stream.py``, which applies identical validation rules and the same sink.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vehicle_anomaly.artifacts import ModelArtifactsError
from vehicle_anomaly.config import AppConfig, load_config
from vehicle_anomaly.local_stream import run_local_stream
from vehicle_anomaly.logging_utils import configure_logging, get_logger
from vehicle_anomaly.predict import AnomalyPredictor, WindowAnomalyPredictor
from vehicle_anomaly.schema import ID_FIELDS, NUMERIC_FIELDS, REQUIRED_FIELDS, TIMESTAMP_FIELD, VALID_RANGES
from vehicle_anomaly.stream_common import BatchMetrics, BatchProcessor, summarize_run

logger = get_logger(__name__)
SPARK_CHECKPOINT_SUBDIR = "spark"
LOCAL_CHECKPOINT_SUBDIR = "local"


def java_status() -> tuple[bool, str]:
    executable = "java.exe" if sys.platform == "win32" else "java"
    java_home = os.environ.get("JAVA_HOME")
    if java_home and (Path(java_home) / "bin" / executable).is_file():
        return True, f"JAVA_HOME={java_home}"
    found = shutil.which("java")
    if found:
        return True, found
    return False, "Java runtime not found (set JAVA_HOME or put java on PATH)"


def hadoop_windows_status() -> tuple[bool, str]:
    """Spark's file source and checkpoints use Hadoop's local filesystem, which on Windows needs
    winutils.exe and hadoop.dll (not shipped by Apache). Other platforms need nothing extra."""
    if sys.platform == "win32":
        hadoop_home = os.environ.get("HADOOP_HOME")
        if hadoop_home and (Path(hadoop_home) / "bin" / "winutils.exe").is_file():
            return True, f"HADOOP_HOME={hadoop_home}"
        return False, (
            "Spark file streaming on Windows requires HADOOP_HOME with bin\\winutils.exe and bin\\hadoop.dll; "
            "use the local engine, Docker, WSL or Linux instead"
        )
    return True, "not required on this platform"


def spark_status(file_io: bool = True) -> tuple[bool, str]:
    """Whether the Spark engine can run. ``file_io=False`` checks only what in-memory Spark needs."""
    if importlib.util.find_spec("pyspark") is None:
        return False, "pyspark is not installed (pip install -r requirements-spark.txt)"
    java_ok, java_info = java_status()
    if not java_ok or not file_io:
        return java_ok, java_info
    return hadoop_windows_status()


def build_spark_session(config: AppConfig) -> Any:
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    builder = (
        SparkSession.builder.appName("vehicle-telemetry-anomaly-detection")
        .master(config.streaming.spark_master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(config.streaming.shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
    )
    if importlib.util.find_spec("pyarrow") is not None:
        builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def raw_event_schema() -> Any:
    from pyspark.sql.types import StringType, StructField, StructType

    # Every field is read as a string so type errors become validation reasons instead of nulls.
    return StructType(
        [StructField(name, StringType(), True) for name in REQUIRED_FIELDS]
        + [StructField("_corrupt_record", StringType(), True)]
    )


def add_validation_columns(parsed: Any) -> Any:
    """Spark SQL version of ``validation.validate_record`` (same rule order)."""
    from pyspark.sql import functions as F

    event = F.col("e")
    checks = [F.when(event.isNull() | F.col("e._corrupt_record").isNotNull(), F.lit("malformed_json"))]
    for name in REQUIRED_FIELDS:
        raw = F.col(f"e.{name}")
        checks.append(F.when(raw.isNull() | (F.trim(raw) == ""), F.lit(f"missing_field:{name}")))
    numbers = {name: F.expr(f"try_cast(trim(e.{name}) AS DOUBLE)") for name in NUMERIC_FIELDS}
    for name, value in numbers.items():
        checks.append(F.when(value.isNull() | F.isnan(value), F.lit(f"non_numeric:{name}")))
    timestamp = F.expr(f"try_cast(trim(e.{TIMESTAMP_FIELD}) AS TIMESTAMP)")
    checks.append(F.when(timestamp.isNull(), F.lit("invalid_timestamp")))
    for name, value in numbers.items():
        low, high = VALID_RANGES[name]
        checks.append(F.when((value < low) | (value > high), F.lit(f"out_of_range:{name}")))

    return parsed.select(
        "raw_value",
        "source_file",
        F.coalesce(*checks).alias("invalid_reason"),
        *[F.trim(F.col(f"e.{name}")).alias(name) for name in ID_FIELDS],
        timestamp.alias(TIMESTAMP_FIELD),
        *[value.alias(name) for name, value in numbers.items()],
    )


def parse_raw_lines(raw: Any) -> Any:
    """Parse a DataFrame with a ``value`` text column (streaming or batch) into validated rows."""
    from pyspark.sql import functions as F

    options = {"mode": "PERMISSIVE", "columnNameOfCorruptRecord": "_corrupt_record"}
    parsed = raw.where(F.trim(F.col("value")) != "").select(
        F.col("value").alias("raw_value"),
        F.input_file_name().alias("source_file"),
        F.from_json(F.col("value"), raw_event_schema(), options).alias("e"),
    )
    return add_validation_columns(parsed)


def process_spark_batch(
    batch_df: Any, batch_id: int, processor: BatchProcessor, started: float | None = None
) -> BatchMetrics | None:
    """``foreachBatch`` body: split a validated micro-batch, then score and write via ``processor``."""
    from pyspark.sql import functions as F

    started = time.perf_counter() if started is None else started
    batch_df = batch_df.persist()
    try:
        input_rows = batch_df.count()
        if input_rows == 0:
            return None
        is_invalid = F.col("invalid_reason").isNotNull()
        invalid = batch_df.where(is_invalid).select("raw_value", "invalid_reason").toPandas()
        valid = batch_df.where(~is_invalid).select(*REQUIRED_FIELDS).toPandas()
        files = sorted(
            Path(row["source_file"]).name
            for row in batch_df.select("source_file").distinct().collect()
            if row["source_file"]
        )
        return processor.process(int(batch_id), valid, invalid, input_rows, "spark", files, started)
    except Exception:
        logger.exception("Spark micro-batch failed", extra={"batch_id": batch_id})
        raise
    finally:
        batch_df.unpersist()


def run_spark_stream(config: AppConfig, processor: BatchProcessor, checkpoint_dir: Path) -> int:
    """Run the Spark query; returns the number of micro-batches processed."""
    spark = build_spark_session(config)
    processed_batches = 0

    def handle_batch(batch_df: Any, batch_id: int) -> None:
        nonlocal processed_batches
        if process_spark_batch(batch_df, batch_id, processor) is not None:
            processed_batches += 1

    try:
        raw = (
            spark.readStream.format("text")
            .option("maxFilesPerTrigger", config.streaming.max_files_per_trigger)
            .option("pathGlobFilter", "*.jsonl")
            .load(str(config.paths.streaming_input))
        )
        writer = (
            parse_raw_lines(raw)
            .writeStream.queryName("vehicle_telemetry_anomaly_detection")
            .foreachBatch(handle_batch)
            .option("checkpointLocation", str(checkpoint_dir))
        )
        if config.streaming.trigger == "available_now":
            query = writer.trigger(availableNow=True).start()
            query.awaitTermination()
        else:
            query = writer.trigger(processingTime=f"{config.streaming.processing_interval_seconds} seconds").start()
            try:
                query.awaitTermination(timeout=config.streaming.max_runtime_seconds)
            except KeyboardInterrupt:
                logger.info("Spark streaming interrupted by user")
            query.stop()
        progress = query.lastProgress
        if progress:
            logger.info(
                "Spark query progress",
                extra={
                    "batch_id": progress.get("batchId"),
                    "num_input_rows": progress.get("numInputRows"),
                    "processed_rows_per_second": progress.get("processedRowsPerSecond"),
                },
            )
        return processed_batches
    finally:
        spark.stop()


def run_streaming(config: AppConfig, engine: str | None = None, reset: bool = False) -> dict[str, Any]:
    paths = config.paths
    requested = engine or config.streaming.engine
    if reset:
        for directory in (paths.streaming_output, paths.checkpoint_dir):
            if directory.exists():
                shutil.rmtree(directory)
        logger.info("Streaming output and checkpoints reset")

    predictor: AnomalyPredictor | WindowAnomalyPredictor
    feature_set = config.streaming.feature_set
    if feature_set == "window":
        try:
            predictor = WindowAnomalyPredictor.load(paths.model_dir)
        except ModelArtifactsError as exc:
            logger.warning("No window model available; using the point model", extra={"reason": str(exc)})
            predictor, feature_set = AnomalyPredictor.load(paths.model_dir), "point"
    else:
        predictor = AnomalyPredictor.load(paths.model_dir)
    processor = BatchProcessor(predictor, paths.streaming_output, config.streaming.output_format)

    spark_ok, spark_info = spark_status()
    if requested == "spark" and not spark_ok:
        raise RuntimeError(f"Spark engine requested but unavailable: {spark_info}")
    selected = "spark" if requested == "spark" or (requested == "auto" and spark_ok) else "local"
    if requested == "auto" and not spark_ok:
        logger.warning("Spark unavailable, using local engine", extra={"reason": spark_info})

    started = time.perf_counter()
    note = ""
    batches_this_run = 0
    if selected == "spark":
        try:
            batches_this_run = run_spark_stream(config, processor, paths.checkpoint_dir / SPARK_CHECKPOINT_SUBDIR)
        except Exception as exc:
            if requested != "auto":
                raise
            logger.warning("Spark engine failed; falling back to local engine", extra={"error": str(exc)[:500]})
            note = f"Spark failed ({type(exc).__name__}); local fallback used."
            selected = "local"
        else:
            logger.info("Spark streaming finished", extra={"batches": batches_this_run})
    if selected == "local":
        s = config.streaming
        batches_this_run = len(
            run_local_stream(
                paths.streaming_input,
                paths.checkpoint_dir / LOCAL_CHECKPOINT_SUBDIR,
                processor,
                s.max_files_per_trigger,
                s.trigger,
                s.processing_interval_seconds,
                s.max_runtime_seconds,
            )
        )
    wall_clock = time.perf_counter() - started

    summary = summarize_run(
        paths.streaming_output,
        paths.reports_dir,
        paths.streaming_ground_truth,
        run_info={
            "engine": selected,
            "requested_engine": requested,
            "wall_clock_seconds": round(wall_clock, 2),
            "batches_this_run": batches_this_run,
            "model": predictor.model_name,
            "feature_set": feature_set,
            "note": note,
        },
    )
    logger.info(
        "Streaming run complete",
        extra={
            k: summary[k]
            for k in ("batches", "input_rows", "valid_rows", "invalid_rows", "duplicates_dropped", "anomalies_detected")
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the telemetry streaming pipeline.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--engine", choices=["auto", "spark", "local"], default=None)
    parser.add_argument("--trigger", choices=["available_now", "processing_time"], default=None)
    parser.add_argument("--max-runtime", type=int, default=None, help="Seconds to run with processing_time trigger")
    parser.add_argument("--reset", action="store_true", help="Delete previous outputs and checkpoints first")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config.logging.level, config.logging.json)
    if args.trigger or args.max_runtime is not None:
        from dataclasses import replace

        config = replace(
            config,
            streaming=replace(
                config.streaming,
                trigger=args.trigger or config.streaming.trigger,
                max_runtime_seconds=args.max_runtime
                if args.max_runtime is not None
                else config.streaming.max_runtime_seconds,
            ),
        )
    summary = run_streaming(config, engine=args.engine, reset=args.reset)
    print(
        f"engine={summary['run']['engine']} batches_this_run={summary['run']['batches_this_run']} "
        f"total_batches={summary['batches']} input={summary['input_rows']:,} "
        f"valid={summary['valid_rows']:,} invalid={summary['invalid_rows']:,} "
        f"duplicates_dropped={summary['duplicates_dropped']:,} anomalies={summary['anomalies_detected']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
