"""Evaluate saved models on the held-out test split and write reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
from vehicle_anomaly.schema import (
    ANOMALY_TYPE_COLUMN,
    ANOMALY_TYPES,
    DRIFT_ANOMALY_TYPES,
    FEATURE_SETS,
    LABEL_COLUMN,
    NORMAL_TYPE,
)
from vehicle_anomaly.train import TEST_SET_FILE, classification_metrics, feature_matrix, model_scores

logger = get_logger(__name__)


def _recall_by_type(test_df: pd.DataFrame, predictions: np.ndarray) -> dict[str, dict[str, Any]]:
    result = {}
    for kind, group in test_df.groupby(ANOMALY_TYPE_COLUMN):
        flagged = predictions[group.index.to_numpy()]
        key = "false_positive_rate" if kind == NORMAL_TYPE else "recall"
        result[str(kind)] = {"events": len(group), key: float(flagged.mean())}
    return result


def _fmt_size(size: int | None) -> str:
    return f"{size / 1024:,.0f} KB" if size else "-"


def _render_markdown(metadata: dict[str, Any], results: dict[str, Any]) -> str:
    models = metadata["models"]
    selected = {metadata["selected_model"], metadata.get("selected_window_model")}
    rows = []
    for name, res in results.items():
        m, entry = res["metrics"], models[name]
        rows.append(
            [
                name + (" (selected)" if name in selected else ""),
                entry["feature_set"],
                "yes" if entry["supervised"] else "no",
                m["threshold"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["roc_auc"],
                m["average_precision"],
            ]
        )
    kinds = [k for k in (NORMAL_TYPE, *ANOMALY_TYPES) if any(k in r["by_anomaly_type"] for r in results.values())]
    type_rows = []
    for kind in kinds:
        row: list[Any] = [kind + (" (drift)" if kind in DRIFT_ANOMALY_TYPES else "")]
        row.append(next(iter(results.values()))["by_anomaly_type"].get(kind, {}).get("events", 0))
        for res in results.values():
            stats = res["by_anomaly_type"].get(kind, {})
            row.append(stats.get("recall", stats.get("false_positive_rate")))
        type_rows.append(row)
    onnx_rows = []
    for name, entry in models.items():
        onnx = entry.get("onnx")
        if onnx:
            parity = onnx["validation_parity"]
            onnx_rows.append(
                [
                    name,
                    _fmt_size(entry.get("artifact_size_bytes")),
                    _fmt_size(onnx["size_bytes"]),
                    f"{parity['max_abs_score_diff']:.1e}",
                    f"{parity['decision_agreement']:.4%}",
                ]
            )
    ds = metadata["datasets"]
    lines = [
        "# Model evaluation (test split)",
        "",
        "Data: **synthetic** telemetry from `vehicle_anomaly.data_generator`. Splits are grouped by vehicle; "
        "thresholds (max F1), early stopping and model selection use the validation split only. The test split "
        "is scored once for this report.",
        "",
        f"- Train: {ds['train']['rows']:,} events / {ds['train']['vehicles']} vehicles",
        f"- Validation: {ds['validation']['rows']:,} events / {ds['validation']['vehicles']} vehicles",
        f"- Test: {ds['test']['rows']:,} events / {ds['test']['vehicles']} vehicles "
        f"({ds['test']['anomaly_rate']:.2%} anomalies)",
        f"- Selection metric (validation): `{metadata['selection_metric']}`; selected point model "
        f"`{metadata['selected_model']}`, selected window model `{metadata.get('selected_window_model')}`",
        "- Feature sets: **point** = the event's own readings (+3 derived); **window** = point + per-vehicle "
        "history features (EWMA deviation and trend of engine temperature and battery voltage, history length)",
        "- `isolation_forest` is unsupervised: it is fitted without labels; only its threshold uses validation labels",
        "",
        markdown_table(
            ["Model", "Features", "Supervised", "Threshold", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], rows
        ),
        "",
        "## Recall by anomaly type (false-positive rate for `normal`)",
        "",
        "Drift types are gradual faults relative to the vehicle's own baseline; single readings often stay "
        "inside the fleet's normal range.",
        "",
        markdown_table(["Type", "Test events", *results.keys()], type_rows),
        "",
        "## ONNX export (edge runtime)",
        "",
        "Parity is measured on the validation split: ONNX Runtime (float32) vs the native model (float64).",
        "",
        markdown_table(["Model", "joblib size", "ONNX size", "Max score diff", "Decision agreement"], onnx_rows)
        if onnx_rows
        else "ONNX export was not available in this environment.",
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
    y_test = test_df[LABEL_COLUMN].to_numpy(dtype=int)
    matrices = {feature_set: feature_matrix(preprocessor, test_df, feature_set) for feature_set in FEATURE_SETS}

    results: dict[str, Any] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    importances: dict[str, tuple[Sequence[str], np.ndarray]] = {}
    selected = [metadata["selected_model"], metadata.get("selected_window_model")]
    for name, entry in metadata["models"].items():
        feature_set = entry.get("feature_set", "point")
        model = load_model(paths.model_dir, name)
        scores = model_scores(model, matrices[feature_set])
        metrics = classification_metrics(y_test, scores, entry["threshold"])
        by_type = _recall_by_type(test_df, scores >= entry["threshold"])
        results[name] = {"feature_set": feature_set, "metrics": metrics, "by_anomaly_type": by_type}
        curves[name] = (y_test, scores)
        if name in selected and hasattr(model, "feature_importances_"):
            importances[name] = (FEATURE_SETS[feature_set], np.asarray(model.feature_importances_))
        entry["test_metrics"] = metrics
        entry["test_recall_by_type"] = {k: v.get("recall") for k, v in by_type.items() if k != NORMAL_TYPE}
        logger.info(
            "Test evaluation",
            extra={
                "model": name,
                "feature_set": feature_set,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "roc_auc": round(metrics["roc_auc"] or 0.0, 4),
            },
        )
    save_metadata(paths.model_dir, metadata)

    figures = paths.figures_dir
    shown = [n for n in (*selected, "isolation_forest") if n and n in results]
    plot_confusion_matrices(
        {n: results[n]["metrics"]["confusion_matrix"] for n in shown}, figures / "confusion_matrices.png"
    )
    plot_roc_pr_curves(curves, figures / "roc_pr_curves.png")
    if importances:
        plot_feature_importance(importances, figures / "feature_importance.png")

    report = {
        "evaluated_at": metadata.get("created_at"),
        "selected_model": metadata["selected_model"],
        "selected_window_model": metadata.get("selected_window_model"),
        "selection_metric": metadata["selection_metric"],
        "datasets": metadata["datasets"],
        "results": results,
        "onnx": {name: entry.get("onnx") for name, entry in metadata["models"].items() if entry.get("onnx")},
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
