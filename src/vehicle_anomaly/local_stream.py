"""Local micro-batch streaming engine (pandas) - fallback when Java/PySpark are unavailable.

It mirrors Spark's file-source semantics: new ``*.jsonl`` files in the input directory are picked up
in name order, at most ``max_files_per_trigger`` per micro-batch. Progress is checkpointed to a JSON
file *after* a batch's outputs are written, so a crash causes that batch to be re-processed and its
(idempotent, batch-id named) outputs to be overwritten.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from vehicle_anomaly.logging_utils import get_logger
from vehicle_anomaly.stream_common import BatchMetrics, BatchProcessor
from vehicle_anomaly.validation import validate_lines

logger = get_logger(__name__)
CHECKPOINT_FILE = "local_engine_state.json"
INPUT_GLOB = "*.jsonl"


@dataclass
class LocalCheckpoint:
    path: Path
    processed_files: list[str] = field(default_factory=list)
    next_batch_id: int = 0

    @classmethod
    def load(cls, checkpoint_dir: Path) -> "LocalCheckpoint":
        path = checkpoint_dir / CHECKPOINT_FILE
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
            return cls(path, list(state["processed_files"]), int(state["next_batch_id"]))
        return cls(path)

    def commit(self, files: list[str]) -> None:
        self.processed_files.extend(files)
        self.next_batch_id += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(
            json.dumps({"processed_files": self.processed_files, "next_batch_id": self.next_batch_id}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)


def _pending_files(input_dir: Path, processed: set[str]) -> list[Path]:
    return sorted(
        p for p in input_dir.glob(INPUT_GLOB)
        if p.is_file() and not p.name.startswith((".", "_")) and p.name not in processed
    )


def process_available(
    input_dir: Path, checkpoint: LocalCheckpoint, processor: BatchProcessor, max_files_per_trigger: int
) -> list[BatchMetrics]:
    """Process every pending file in micro-batches; returns metrics for the batches run."""
    results: list[BatchMetrics] = []
    while True:
        pending = _pending_files(input_dir, set(checkpoint.processed_files))[: max(1, max_files_per_trigger)]
        if not pending:
            return results
        started = time.perf_counter()
        lines: list[str] = []
        for path in pending:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                lines.extend(handle.read().splitlines())
        validation = validate_lines(lines)
        names = [p.name for p in pending]
        try:
            metrics = processor.process(
                checkpoint.next_batch_id,
                validation.valid,
                validation.invalid,
                validation.input_rows,
                engine="local",
                source_files=names,
                started=started,
            )
        except Exception:
            logger.exception("Micro-batch failed; checkpoint not advanced", extra={"files": names})
            raise
        checkpoint.commit(names)
        results.append(metrics)


def run_local_stream(
    input_dir: Path,
    checkpoint_dir: Path,
    processor: BatchProcessor,
    max_files_per_trigger: int,
    trigger: str = "available_now",
    processing_interval_seconds: int = 5,
    max_runtime_seconds: int | None = None,
) -> list[BatchMetrics]:
    input_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = LocalCheckpoint.load(checkpoint_dir)
    logger.info(
        "Local streaming engine started",
        extra={"input_dir": str(input_dir), "resume_from_batch": checkpoint.next_batch_id,
               "already_processed_files": len(checkpoint.processed_files), "trigger": trigger},
    )
    if trigger == "available_now":
        return process_available(input_dir, checkpoint, processor, max_files_per_trigger)

    results: list[BatchMetrics] = []
    deadline = time.monotonic() + max_runtime_seconds if max_runtime_seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            results.extend(process_available(input_dir, checkpoint, processor, max_files_per_trigger))
            time.sleep(processing_interval_seconds)
    except KeyboardInterrupt:
        logger.info("Local streaming engine interrupted by user")
    return results
