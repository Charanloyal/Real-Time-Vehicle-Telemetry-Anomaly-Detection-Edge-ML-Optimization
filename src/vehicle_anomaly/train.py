"""Model training: vehicle-grouped splits, point vs window feature sets, supervised vs unsupervised.

Models (``training.models`` in config.yaml):

=====================  ==================  ===========  =========================================
name                   algorithm           feature set  role
=====================  ==================  ===========  =========================================
random_forest          RandomForest        point        baseline, one event at a time
xgboost                XGBoost             point        boosted trees, one event at a time
random_forest_window   RandomForest        window       + per-vehicle history features
xgboost_window         XGBoost             window       + per-vehicle history features
isolation_forest       IsolationForest     window       unsupervised baseline (labels unused in fit)
=====================  ==================  ===========  =========================================

Decision thresholds (max F1), XGBoost early stopping and model selection use the validation split
only. Supervised models are also exported to ONNX and checked for score parity.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from vehicle_anomaly.artifacts import (
    library_versions,
    model_file_name,
    save_metadata,
    save_model,
    save_preprocessor,
    utc_now_iso,
)
from vehicle_anomaly.config import AppConfig, load_config
from vehicle_anomaly.logging_utils import configure_logging, get_logger
from vehicle_anomaly.onnx_export import OnnxScorer, export_to_onnx, onnx_status, parity_report
from vehicle_anomaly.preprocessing import build_preprocessor, split_by_vehicle
from vehicle_anomaly.schema import (
    ANOMALY_TYPE_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_SETS,
    LABEL_COLUMN,
    MODEL_INPUT_FIELDS,
    REQUIRED_FIELDS,
    WINDOW_FEATURE_COLUMNS,
)
from vehicle_anomaly.window_features import (
    LONG_HALFLIFE_EVENTS,
    SHORT_HALFLIFE_EVENTS,
    compute_window_features,
)

logger = get_logger(__name__)
MODEL_SPECS: dict[str, tuple[str, str]] = {
    "random_forest": ("random_forest", "point"),
    "xgboost": ("xgboost", "point"),
    "random_forest_window": ("random_forest", "window"),
    "xgboost_window": ("xgboost", "window"),
    "isolation_forest": ("isolation_forest", "window"),
}
SUPERVISED_ALGORITHMS = ("random_forest", "xgboost")
SUPPORTED_MODELS = tuple(MODEL_SPECS)
TEST_SET_FILE = "test.parquet"
SPLITS_FILE = "splits.json"


def xgboost_status() -> tuple[bool, str]:
    try:
        import xgboost

        return True, xgboost.__version__
    except Exception as exc:  # ImportError or a broken native library
        return False, f"{type(exc).__name__}: {exc}"


def build_model(algorithm: str, params: dict[str, Any], seed: int, scale_pos_weight: float) -> Any:
    if algorithm == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
            **params,
        )
    if algorithm == "isolation_forest":
        return IsolationForest(random_state=seed, **params)
    raise ValueError(f"Unsupported algorithm '{algorithm}'")


def model_scores(model: Any, features: np.ndarray) -> np.ndarray:
    """Anomaly score, higher = more anomalous: class-1 probability, or the IsolationForest anomaly score."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    return -np.asarray(model.score_samples(features), dtype=np.float64)


def feature_matrix(preprocessor: Pipeline, frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    """Model input for a frame holding complete per-vehicle histories (training / evaluation)."""
    point = np.asarray(preprocessor.transform(frame), dtype=np.float64)
    if feature_set == "point":
        return point
    window, _ = compute_window_features(frame)
    return np.hstack([point, window.to_numpy(dtype=np.float64)])


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Decision threshold maximising F1 on the validation split."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0
    )
    return float(thresholds[int(np.argmax(f1[:-1]))])


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    both_classes = 0 < y_true.sum() < len(y_true)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(y_true, scores)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "support": {"normal": int((y_true == 0).sum()), "anomaly": int((y_true == 1).sum())},
    }


def _dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "vehicles": int(frame["vehicle_id"].nunique()),
        "anomalies": int(frame[LABEL_COLUMN].sum()),
        "anomaly_rate": float(frame[LABEL_COLUMN].mean()),
        "anomaly_types": {str(k): int(v) for k, v in frame[ANOMALY_TYPE_COLUMN].value_counts().items()},
    }


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found at {path}. Run: python scripts/generate_data.py")
    df = pd.read_parquet(path)
    missing = [c for c in (*REQUIRED_FIELDS, LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing column(s): {missing}")
    return df


def _select(models_meta: dict[str, Any], feature_set: str, metric: str) -> str | None:
    candidates = [
        name for name, entry in models_meta.items() if entry["feature_set"] == feature_set and entry["supervised"]
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda n: models_meta[n]["validation_metrics"].get(metric) or 0.0)


def _export_onnx(
    name: str, model: Any, x_val: np.ndarray, val_scores: np.ndarray, threshold: float, model_dir: Path
) -> dict[str, Any] | None:
    ok, info = onnx_status()
    if not ok:
        logger.warning("onnxruntime unavailable; skipping ONNX export", extra={"model": name, "reason": info})
        return None
    try:
        payload = export_to_onnx(model, x_val.shape[1])
    except Exception as exc:
        logger.warning("ONNX export failed", extra={"model": name, "error": str(exc)[:300]})
        return None
    path = model_dir / f"{name}.onnx"
    path.write_bytes(payload)
    parity = parity_report(val_scores, OnnxScorer(payload)(x_val), threshold)
    logger.info("ONNX exported", extra={"model": name, "size_bytes": len(payload), **parity})
    return {"file": path.name, "size_bytes": len(payload), "validation_parity": parity}


def run_training(config: AppConfig, evaluate: bool = True) -> dict[str, Any]:
    paths = config.paths
    training = config.training
    df = load_training_data(paths.raw_data)
    train_df, val_df, test_df = split_by_vehicle(df, training.validation_size, training.test_size, config.random_seed)
    logger.info(
        "Dataset split by vehicle",
        extra={"train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df)},
    )

    preprocessor = build_preprocessor().fit(train_df)
    matrices = {
        feature_set: (
            feature_matrix(preprocessor, train_df, feature_set),
            feature_matrix(preprocessor, val_df, feature_set),
        )
        for feature_set in FEATURE_SETS
    }
    y_train = train_df[LABEL_COLUMN].to_numpy(dtype=int)
    y_val = val_df[LABEL_COLUMN].to_numpy(dtype=int)
    scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    xgb_ok, xgb_info = xgboost_status()
    models_meta: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name in training.models:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unsupported model '{name}' in training.models. Choose from {SUPPORTED_MODELS}")
        algorithm, feature_set = MODEL_SPECS[name]
        if algorithm == "xgboost" and not xgb_ok:
            logger.warning("XGBoost unavailable; skipping", extra={"model": name, "reason": xgb_info})
            skipped[name] = xgb_info
            continue
        x_train, x_val = matrices[feature_set]
        params = dict(getattr(training, algorithm))
        model = build_model(algorithm, params, config.random_seed, scale_pos_weight)
        started = time.perf_counter()
        if algorithm == "xgboost":
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        elif algorithm == "isolation_forest":
            model.fit(x_train)  # unsupervised: labels are not used
        else:
            model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        val_scores = model_scores(model, x_val)
        threshold = tune_threshold(y_val, val_scores)
        val_metrics = classification_metrics(y_val, val_scores, threshold)
        model_path = save_model(paths.model_dir, name, model)
        supervised = algorithm in SUPERVISED_ALGORITHMS
        entry: dict[str, Any] = {
            "file": model_file_name(name),
            "model_type": type(model).__name__,
            "algorithm": algorithm,
            "feature_set": feature_set,
            "feature_columns": list(FEATURE_SETS[feature_set]),
            "supervised": supervised,
            "threshold": threshold,
            "params": params,
            "fit_seconds": round(fit_seconds, 3),
            "artifact_size_bytes": model_path.stat().st_size,
            "validation_metrics": val_metrics,
        }
        if algorithm == "xgboost" and getattr(model, "best_iteration", None) is not None:
            entry["best_iteration"] = int(model.best_iteration)
        if supervised:
            entry["onnx"] = _export_onnx(name, model, x_val, val_scores, threshold, paths.model_dir)
        models_meta[name] = entry
        logger.info(
            "Model trained",
            extra={
                "model": name,
                "feature_set": feature_set,
                "fit_seconds": round(fit_seconds, 2),
                "val_f1": round(val_metrics["f1"], 4),
                "val_average_precision": round(val_metrics["average_precision"] or 0.0, 4),
                "threshold": round(threshold, 4),
            },
        )

    metric = training.selection_metric
    selected = _select(models_meta, "point", metric)
    if selected is None:
        raise RuntimeError("At least one supervised point model (random_forest or xgboost) is required")
    selected_window = _select(models_meta, "window", metric)

    save_preprocessor(paths.model_dir, preprocessor)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(paths.processed_dir / TEST_SET_FILE, index=False)
    (paths.processed_dir / SPLITS_FILE).write_text(
        json.dumps(
            {
                "train_vehicles": sorted(train_df["vehicle_id"].unique().tolist()),
                "validation_vehicles": sorted(val_df["vehicle_id"].unique().tolist()),
                "test_vehicles": sorted(test_df["vehicle_id"].unique().tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "created_at": utc_now_iso(),
        "data_source": "synthetic (vehicle_anomaly.data_generator)",
        "random_seed": config.random_seed,
        "selected_model": selected,
        "selected_window_model": selected_window,
        "selection_metric": metric,
        "feature_columns": list(FEATURE_COLUMNS),
        "window_feature_columns": list(WINDOW_FEATURE_COLUMNS),
        "window_feature_settings": {
            "long_halflife_events": LONG_HALFLIFE_EVENTS,
            "short_halflife_events": SHORT_HALFLIFE_EVENTS,
        },
        "model_input_fields": list(MODEL_INPUT_FIELDS),
        "excluded_columns": [
            "event_id",
            "vehicle_id",
            "event_timestamp",
            "latitude",
            "longitude",
            LABEL_COLUMN,
            ANOMALY_TYPE_COLUMN,
        ],
        "split_strategy": "GroupShuffleSplit by vehicle_id (no vehicle shared between splits)",
        "datasets": {
            "train": _dataset_summary(train_df),
            "validation": _dataset_summary(val_df),
            "test": _dataset_summary(test_df),
        },
        "models": models_meta,
        "skipped_models": skipped,
        "library_versions": library_versions(),
    }
    save_metadata(paths.model_dir, metadata)
    logger.info(
        "Artifacts saved",
        extra={"model_dir": str(paths.model_dir), "selected_model": selected, "selected_window_model": selected_window},
    )

    if evaluate:
        from vehicle_anomaly.evaluate import evaluate_models

        evaluate_models(config)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train anomaly detection models.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not evaluate on the test split")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    configure_logging(config.logging.level, config.logging.json)
    run_training(config, evaluate=not args.skip_evaluation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
