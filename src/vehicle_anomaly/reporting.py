"""Matplotlib figures and small Markdown helpers for reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrices(matrices: Mapping[str, Sequence[Sequence[int]]], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.5 * len(matrices), 4), squeeze=False)
    for ax, (name, matrix) in zip(axes[0], matrices.items(), strict=True):
        data = np.asarray(matrix)
        ax.imshow(data, cmap="Blues")
        for (row, col), value in np.ndenumerate(data):
            color = "white" if value > data.max() / 2 else "black"
            ax.text(col, row, f"{value:,}", ha="center", va="center", color=color)
        ax.set_xticks([0, 1], labels=["normal", "anomaly"])
        ax.set_yticks([0, 1], labels=["normal", "anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} (test)")
    return _save(fig, path)


def plot_roc_pr_curves(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y_true, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        roc_ax.plot(fpr, tpr, label=name)
        pr_ax.plot(recall, precision, label=name)
    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    roc_ax.set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    pr_ax.set(title="Precision-recall curve (test)", xlabel="Recall", ylabel="Precision")
    roc_ax.legend()
    pr_ax.legend()
    return _save(fig, path)


def plot_feature_importance(importances: Mapping[str, tuple[Sequence[str], np.ndarray]], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(importances), figsize=(6 * len(importances), 5), squeeze=False)
    for ax, (name, (feature_names, values)) in zip(axes[0], importances.items(), strict=True):
        order = np.argsort(values)
        ax.barh(np.asarray(feature_names)[order], np.asarray(values)[order], color="#4C72B0")
        ax.set_title(f"{name} feature importance")
    return _save(fig, path)


def plot_benchmark(single: Mapping[str, Mapping[str, Any]], batch: Mapping[str, Mapping[str, Any]], path: Path) -> Path:
    fig, (lat_ax, thr_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(single)
    p50 = [single[n]["p50_ms"] for n in names]
    p95 = [single[n]["p95_ms"] for n in names]
    x = np.arange(len(names))
    lat_ax.bar(x - 0.2, p50, width=0.4, label="p50")
    lat_ax.bar(x + 0.2, p95, width=0.4, label="p95")
    lat_ax.set_xticks(x, labels=names, rotation=15)
    lat_ax.set(title="Single-event latency", ylabel="milliseconds")
    lat_ax.legend()
    batch_names = list(batch)
    thr_ax.bar(batch_names, [batch[n]["throughput_per_s"] for n in batch_names], color="#55A868")
    thr_ax.set(title="Batch throughput", ylabel="predictions / second")
    thr_ax.tick_params(axis="x", rotation=15)
    return _save(fig, path)


def plot_streaming_batches(batches: Sequence[Mapping[str, Any]], path: Path) -> Path:
    ids = [b["batch_id"] for b in batches]
    valid = np.array([b["valid_rows"] for b in batches])
    invalid = np.array([b["invalid_rows"] for b in batches])
    anomalies = np.array([b["anomalies_detected"] for b in batches])
    fig, (rows_ax, anom_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    rows_ax.bar(ids, valid, label="valid")
    rows_ax.bar(ids, invalid, bottom=valid, label="invalid", color="#C44E52")
    rows_ax.set(title="Rows per micro-batch", xlabel="batch id", ylabel="rows")
    rows_ax.legend()
    anom_ax.bar(ids, anomalies, color="#DD8452")
    anom_ax.set(title="Anomalies detected per micro-batch", xlabel="batch id", ylabel="events")
    return _save(fig, path)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.1f}"
            return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)
