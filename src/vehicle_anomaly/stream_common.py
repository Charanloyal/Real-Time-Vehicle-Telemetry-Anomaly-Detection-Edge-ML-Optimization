"""Engine-independent micro-batch processing: de-duplication, scoring, sinks, metrics, summary.

Both the Spark engine and the local pandas engine hand each validated micro-batch to
:class:`BatchProcessor`. Output files are named by batch id and overwritten on replay, so a batch
re-executed after a failure (before its checkpoint commit) does not create duplicate output.

With a window model, per-vehicle history state is carried between micro-batches and written to
``state/batch_NNNNNN.json`` together with the batch outputs. Batch ``k`` always starts from the state
written by batch ``k - 1``, so a replayed batch produces the same features as the original run.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_anomaly.artifacts import utc_now_iso
from vehicle_anomaly.logging_utils import get_logger
from vehicle_anomaly.predict import AnomalyPredictor, WindowAnomalyPredictor
from vehicle_anomaly.reporting import markdown_table, plot_streaming_batches
from vehicle_anomaly.schema import ANOMALY_TYPE_COLUMN, LABEL_COLUMN, NORMAL_TYPE, REQUIRED_FIELDS, TIMESTAMP_FIELD
from vehicle_anomaly.window_features import StateMap, state_from_json, state_to_json

logger = get_logger(__name__)
PREDICTIONS_DIR = "predictions"
INVALID_DIR = "invalid"
METRICS_DIR = "metrics"
STATE_DIR = "state"
STATE_FILES_KEPT = 3


@dataclass
class BatchMetrics:
    batch_id: int
    engine: str
    input_rows: int
    valid_rows: int
    invalid_rows: int
    duplicates_dropped: int
    scored_rows: int
    anomalies_detected: int
    invalid_by_reason: dict[str, int] = field(default_factory=dict)
    processing_seconds: float = 0.0
    rows_per_second: float = 0.0
    processed_at: str = ""
    source_files: list[str] = field(default_factory=list)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class BatchProcessor:
    def __init__(
        self,
        predictor: AnomalyPredictor | WindowAnomalyPredictor,
        output_dir: Path,
        output_format: str = "parquet",
    ) -> None:
        if output_format not in {"parquet", "csv"}:
            raise ValueError("output_format must be 'parquet' or 'csv'")
        self.predictor = predictor
        self.output_dir = output_dir
        self.output_format = output_format
        self.stateful = isinstance(predictor, WindowAnomalyPredictor)
        self._state: StateMap = {}
        self._state_batch_id: int | None = None
        for sub in (PREDICTIONS_DIR, INVALID_DIR, METRICS_DIR, STATE_DIR):
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

    def _state_path(self, batch_id: int) -> Path:
        return self.output_dir / STATE_DIR / f"batch_{batch_id:06d}.json"

    def _state_before(self, batch_id: int) -> StateMap:
        """State after the newest batch before ``batch_id``.

        Normally that is batch ``batch_id - 1``. A batch that produced no rows (e.g. a Spark micro-batch
        for an empty file) saves no state and changes nothing, so the newest earlier state is correct.
        """
        if batch_id == 0:
            return {}
        if self._state_batch_id is not None and self._state_batch_id < batch_id:
            return self._state
        earlier = [
            path
            for path in (self.output_dir / STATE_DIR).glob("batch_*.json")
            if int(path.stem.split("_")[1]) < batch_id
        ]
        if earlier:
            newest = max(earlier, key=lambda p: int(p.stem.split("_")[1]))
            return state_from_json(json.loads(newest.read_text(encoding="utf-8")))
        logger.warning("No earlier window state found; starting vehicles fresh", extra={"batch_id": batch_id})
        return {}

    def _save_state(self, batch_id: int, state: StateMap) -> None:
        _atomic_write_text(self._state_path(batch_id), json.dumps(state_to_json(state)))
        self._state, self._state_batch_id = state, batch_id
        for old in sorted((self.output_dir / STATE_DIR).glob("batch_*.json"))[:-STATE_FILES_KEPT]:
            old.unlink(missing_ok=True)

    def _write_predictions(self, frame: pd.DataFrame, batch_id: int) -> None:
        path = self.output_dir / PREDICTIONS_DIR / f"batch_{batch_id:06d}.{self.output_format}"
        tmp = path.with_name(f".{path.name}.tmp")
        if self.output_format == "parquet":
            frame.to_parquet(tmp, index=False)
        else:
            frame.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def _write_invalid(self, invalid: pd.DataFrame, batch_id: int) -> None:
        path = self.output_dir / INVALID_DIR / f"batch_{batch_id:06d}.jsonl"
        if invalid.empty:
            path.unlink(missing_ok=True)
            return
        lines = (json.dumps({"batch_id": batch_id, **row}) for row in invalid.to_dict("records"))
        _atomic_write_text(path, "\n".join(lines) + "\n")

    def process(
        self,
        batch_id: int,
        valid: pd.DataFrame,
        invalid: pd.DataFrame,
        input_rows: int,
        engine: str,
        source_files: list[str] | None = None,
        started: float | None = None,
    ) -> BatchMetrics:
        started = time.perf_counter() if started is None else started
        valid = valid.loc[:, list(REQUIRED_FIELDS)].copy()
        valid[TIMESTAMP_FIELD] = pd.to_datetime(valid[TIMESTAMP_FIELD], utc=True)
        deduplicated = valid.drop_duplicates(subset="event_id", keep="first").reset_index(drop=True)
        new_state: StateMap | None = None
        if isinstance(self.predictor, WindowAnomalyPredictor):
            scored, new_state = self.predictor.predict_frame(deduplicated, self._state_before(batch_id))
        else:
            scored = self.predictor.predict_frame(deduplicated)
        scored["model_name"] = self.predictor.model_name
        scored["batch_id"] = batch_id
        scored["processed_at"] = pd.Timestamp.now(tz="UTC")

        self._write_predictions(scored, batch_id)
        self._write_invalid(invalid, batch_id)
        if new_state is not None:
            self._save_state(batch_id, new_state)

        elapsed = time.perf_counter() - started
        metrics = BatchMetrics(
            batch_id=batch_id,
            engine=engine,
            input_rows=int(input_rows),
            valid_rows=len(valid),
            invalid_rows=len(invalid),
            duplicates_dropped=int(len(valid) - len(deduplicated)),
            scored_rows=len(scored),
            anomalies_detected=int(scored["is_anomaly"].sum()) if len(scored) else 0,
            invalid_by_reason=dict(Counter(invalid["invalid_reason"])) if len(invalid) else {},
            processing_seconds=round(elapsed, 4),
            rows_per_second=round(input_rows / elapsed, 1) if elapsed > 0 else 0.0,
            processed_at=utc_now_iso(),
            source_files=source_files or [],
        )
        _atomic_write_text(
            self.output_dir / METRICS_DIR / f"batch_{batch_id:06d}.json", json.dumps(asdict(metrics), indent=2)
        )
        logger.info(
            "Micro-batch processed",
            extra={k: v for k, v in asdict(metrics).items() if k not in {"source_files", "invalid_by_reason"}},
        )
        if metrics.invalid_rows:
            logger.warning(
                "Invalid events quarantined",
                extra={"batch_id": batch_id, "invalid_by_reason": metrics.invalid_by_reason},
            )
        return metrics


def load_batch_metrics(output_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = output_dir / METRICS_DIR
    if not metrics_dir.is_dir():
        return []
    batches = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(metrics_dir.glob("batch_*.json"))]
    return sorted(batches, key=lambda b: b["batch_id"])


def load_predictions(output_dir: Path) -> pd.DataFrame:
    pred_dir = output_dir / PREDICTIONS_DIR
    frames = [pd.read_parquet(p) for p in sorted(pred_dir.glob("batch_*.parquet"))]
    frames += [pd.read_csv(p) for p in sorted(pred_dir.glob("batch_*.csv"))]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_run(
    output_dir: Path, reports_dir: Path, ground_truth_path: Path | None = None, run_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Aggregate per-batch metrics into reports/streaming_summary.{json,md} and a figure."""
    batches = load_batch_metrics(output_dir)
    reasons: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for batch in batches:
        for reason, count in batch["invalid_by_reason"].items():
            reasons[reason] += count
            categories[reason.split(":", 1)[0]] += count
    totals = {
        key: int(sum(b[key] for b in batches))
        for key in (
            "input_rows",
            "valid_rows",
            "invalid_rows",
            "duplicates_dropped",
            "scored_rows",
            "anomalies_detected",
        )
    }
    processing_seconds = float(sum(b["processing_seconds"] for b in batches))
    summary: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "run": run_info or {},
        "batches": len(batches),
        "engines": sorted({b["engine"] for b in batches}),
        **totals,
        "invalid_by_category": dict(categories.most_common()),
        "invalid_by_reason": dict(reasons.most_common()),
        "batch_processing_seconds_total": round(processing_seconds, 3),
        "rows_per_second_in_batch_processing": round(totals["input_rows"] / processing_seconds, 1)
        if processing_seconds
        else None,
    }

    if ground_truth_path and ground_truth_path.is_file() and totals["scored_rows"]:
        predictions = load_predictions(output_dir)
        truth_columns = ["event_id", LABEL_COLUMN, ANOMALY_TYPE_COLUMN]
        truth = (
            pd.read_parquet(ground_truth_path)
            .loc[:, truth_columns]
            .rename(columns={LABEL_COLUMN: "actual_is_anomaly", ANOMALY_TYPE_COLUMN: "actual_type"})
        )
        joined = predictions.merge(truth, on="event_id", how="inner")
        if len(joined):
            actual = joined["actual_is_anomaly"].astype(bool)
            predicted = joined["is_anomaly"].astype(bool)
            tp = int((actual & predicted).sum())
            fp = int((~actual & predicted).sum())
            fn = int((actual & ~predicted).sum())
            by_type = {
                str(kind): {"events": len(group), "flagged_share": float(group["is_anomaly"].astype(bool).mean())}
                for kind, group in joined.groupby("actual_type")
            }
            summary["ground_truth_comparison"] = {
                "matched_events": len(joined),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "by_type": by_type,
            }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "streaming_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if batches:
        plot_streaming_batches(batches, reports_dir / "figures" / "streaming_batches.png")
    (reports_dir / "streaming_summary.md").write_text(_render_markdown(summary, batches), encoding="utf-8")
    return summary


def _render_markdown(summary: dict[str, Any], batches: list[dict[str, Any]]) -> str:
    run = summary.get("run", {})
    lines = [
        "# Streaming run summary",
        "",
        f"Generated {summary['generated_at']}. Engine(s): {', '.join(summary['engines']) or 'n/a'}. "
        f"Input: synthetic JSON Lines files. {run.get('note', '')}".rstrip(),
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Micro-batches", summary["batches"]],
                ["Input events", summary["input_rows"]],
                ["Valid events", summary["valid_rows"]],
                ["Invalid events (quarantined)", summary["invalid_rows"]],
                ["Duplicates dropped", summary["duplicates_dropped"]],
                ["Events scored", summary["scored_rows"]],
                ["Anomalies detected", summary["anomalies_detected"]],
                ["Batch processing time, all batches (s)", summary["batch_processing_seconds_total"]],
                ["Rows/s inside batch processing", summary["rows_per_second_in_batch_processing"]],
                ["Batches processed by the last run", run.get("batches_this_run", "n/a")],
                ["Wall-clock time of the last run (s)", run.get("wall_clock_seconds", "n/a")],
            ],
        ),
        "",
        "Totals cover every committed micro-batch in the output directory; a re-run that resumes from the "
        "checkpoint only adds new batches.",
        "",
        "## Invalid events by category",
        "",
        markdown_table(["Category", "Events"], list(summary["invalid_by_category"].items()) or [["none", 0]]),
        "",
    ]
    comparison = summary.get("ground_truth_comparison")
    if comparison:
        type_rows = [
            [kind, stats["events"], stats["flagged_share"]]
            for kind, stats in sorted(comparison.get("by_type", {}).items(), key=lambda kv: kv[0] != NORMAL_TYPE)
        ]
        lines += [
            "## Detection vs. synthetic ground truth",
            "",
            f"Model: `{run.get('model', 'n/a')}` ({run.get('feature_set', 'n/a')} features). Matched "
            f"{comparison['matched_events']:,} scored events to generator labels: precision "
            f"{comparison['precision']:.4f}, recall {comparison['recall']:.4f} "
            f"(TP={comparison['true_positives']:,}, FP={comparison['false_positives']:,}, "
            f"FN={comparison['false_negatives']:,}).",
            "",
            markdown_table(["Ground-truth type", "Events", "Flagged as anomaly"], type_rows),
            "",
        ]
    lines += [
        "## Per-batch metrics",
        "",
        markdown_table(
            ["Batch", "Engine", "Input", "Valid", "Invalid", "Dupes", "Anomalies", "Seconds"],
            [
                [
                    b["batch_id"],
                    b["engine"],
                    b["input_rows"],
                    b["valid_rows"],
                    b["invalid_rows"],
                    b["duplicates_dropped"],
                    b["anomalies_detected"],
                    b["processing_seconds"],
                ]
                for b in batches
            ],
        ),
        "",
        "![Streaming batches](figures/streaming_batches.png)",
        "",
    ]
    return "\n".join(lines)
