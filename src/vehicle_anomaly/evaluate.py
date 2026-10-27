"""Evaluate saved models on the held-out test split and write reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from vehicle_anomaly.artifacts import load_metadata, load_model, load_preprocessor, save_metadata
from vehicle_anomaly.config import AppConfig, load_config
from vehicle_anomaly.logging_utils import configure_logging, get_logger
from vehicle_anomaly.reporting import (
    markdown_table,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_roc_pr_curves,
)
from vehicle_anomaly.schema import ANOMALY_TYPE_COLUMN, FEATURE_COLUMNS, LABEL_COLUMN, NORMAL_TYPE
from vehicle_anomaly.train import TEST_SET_FILE, classification_metrics

logger = get_logger(__name__)


def _recall_by_type(test_df: pd.DataFrame, predictions: np.ndarray) -> dict[str, dict[str, Any]]:
    result = {}
    for kind, group in test_df.groupby(ANOMALY_TYPE_COLUMN):
        flagged = predictions[group.index.to_numpy()]
        key = "false_positive_rate" if kind == NORMAL_TYPE else "recall"
        result[str(kind)] = {"events": int(len(group)), key: float(flagged.mean())}
    return result


def _render_markdown(metadata: dict[str, Any], results: dict[str, Any]) -> str:
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append([
            name + (" (selected)" if name == metadata["selected_model"] else ""),
            m["threshold"], m["precision"], m["recall"], m["f1"], m["roc_auc"], m["average_precision"],
        ])
    type_rows = []
    kinds = sorted({k for res in results.values() for k in res["by_anomaly_type"]})
    for kind in kinds:
        row: list[Any] = [kind]
        for res in results.values():
            stats = res["by_anomaly_type"].get(kind, {})
            row.append(stats.get("recall", stats.get("false_positive_rate")))
        type_rows.append(row)
    ds = metadata["datasets"]
    lines = [
        "# Model evaluation (test split)",
        "",
        "Data: **synthetic** telemetry from `vehicle_anomaly.data_generator`. Splits are grouped by vehicle; "
        "thresholds and model selection use the validation split only. The test split is used once for "
        "the figures below.",
        "",
        f"- Train: {ds['train']['rows']:,} events / {ds['train']['vehicles']} vehicles",
        f"- Validation: {ds['validation']['rows']:,} events / {ds['validation']['vehicles']} vehicles",
        f"- Test: {ds['test']['rows']:,} events / {ds['test']['vehicles']} vehicles "
        f"({ds['test']['anomaly_rate']:.2%} anomalies)",
        f"- Selection metric (validation): `{metadata['selection_metric']}`",
        "",
        markdown_table(["Model", "Threshold", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], rows),
        "",
        "## Recall by anomaly type (false-positive rate for `normal`)",
        "",
        markdown_table(["Type", *results.keys()], type_rows),
        "",
        "## Confusion matrices",
        "",
    ]
    for name, res in results.items():
        (tn, fp), (fn, tp) = res["metrics"]["confusion_matrix"]
        lines.append(f"- **{name}**: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    lines += [
        "",
        "![Confusion matrices](figures/confusion_matrices.png)",
        "![ROC and PR curves](figures/roc_pr_curves.png)",
        "![Feature importance](figures/feature_importance.png)",
        "",
    ]
    return "\n".join(lines)


def evaluate_models(config: AppConfig) -> dict[str, Any]:
    paths = config.paths
    metadata = load_metadata(paths.model_dir)
    test_path = paths.processed_dir / TEST_SET_FILE
    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found at {test_path}. Run training first.")
    test_df = pd.read_parquet(test_path).reset_index(drop=True)
    preprocessor = load_preprocessor(paths.model_dir)
    X_test = preprocessor.transform(test_df)
    y_test = test_df[LABEL_COLUMN].to_numpy(dtype=int)

    results: dict[str, Any] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    importances: dict[str, np.ndarray] = {}
    for name, entry in metadata["models"].items():
        model = load_model(paths.model_dir, name)
        scores = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, scores, entry["threshold"])
        predictions = scores >= entry["threshold"]
        results[name] = {"metrics": metrics, "by_anomaly_type": _recall_by_type(test_df, predictions)}
        curves[name] = (y_test, scores)
        if hasattr(model, "feature_importances_"):
            importances[name] = np.asarray(model.feature_importances_)
        entry["test_metrics"] = metrics
        logger.info(
            "Test evaluation",
            extra={
                "model": name,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "roc_auc": round(metrics["roc_auc"] or 0.0, 4),
            },
        )
    save_metadata(paths.model_dir, metadata)

    figures = paths.figures_dir
    plot_confusion_matrices({n: r["metrics"]["confusion_matrix"] for n, r in results.items()},
                            figures / "confusion_matrices.png")
    plot_roc_pr_curves(curves, figures / "roc_pr_curves.png")
    if importances:
        plot_feature_importance(importances, FEATURE_COLUMNS, figures / "feature_importance.png")

    report = {
        "evaluated_at": metadata.get("created_at"),
        "selected_model": metadata["selected_model"],
        "selection_metric": metadata["selection_metric"],
        "datasets": metadata["datasets"],
        "results": results,
    }
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / "model_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (paths.reports_dir / "model_evaluation.md").write_text(_render_markdown(metadata, results), encoding="utf-8")
    logger.info("Evaluation report written", extra={"reports_dir": str(paths.reports_dir)})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test split.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    configure_logging(config.logging.level, config.logging.json)
    evaluate_models(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
