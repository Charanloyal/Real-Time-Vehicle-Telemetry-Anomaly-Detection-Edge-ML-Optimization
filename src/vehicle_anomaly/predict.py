"""Optimized prediction interface used by the API, the streaming pipeline and the benchmark.

Optimizations compared with the reference path (``pandas.DataFrame`` -> sklearn ``Pipeline`` ->
``predict_proba``), all measured in ``benchmark.py``:

1. Feature extraction builds a float64 NumPy matrix directly from the required input fields only,
   skipping DataFrame construction, dtype coercion and column selection.
2. The fitted imputer statistics are applied with NumPy instead of calling the sklearn pipeline.
3. Model and preprocessing objects are loaded once and reused.
4. Low-overhead model calls: XGBoost uses ``Booster.inplace_predict``; scikit-learn forests score
   small batches single-threaded to avoid thread-pool start-up cost.
5. Batch prediction scores many events with one vectorized call.

Scores are numerically identical to the reference path (verified in tests and in the benchmark).
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from vehicle_anomaly.artifacts import ModelArtifactsError, load_metadata, load_model, load_preprocessor
from vehicle_anomaly.preprocessing import compute_derived_features
from vehicle_anomaly.schema import FEATURE_COLUMNS, MODEL_INPUT_FIELDS

# Below this many rows, forests are scored single-threaded (thread start-up dominates otherwise).
SMALL_BATCH_ROWS = 256


@dataclass(frozen=True)
class Prediction:
    event_id: str | None
    vehicle_id: str | None
    anomaly_score: float
    is_anomaly: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnomalyPredictor:
    """Loads artifacts once and scores telemetry events."""

    def __init__(
        self,
        preprocessor: Any,
        model: Any,
        model_name: str,
        threshold: float,
        metadata: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.model_name = model_name
        self.threshold = float(threshold)
        self.metadata = metadata or {}
        self.n_jobs = n_jobs
        imputer = preprocessor.named_steps["imputer"]
        self._fill_values = np.asarray(imputer.statistics_, dtype=np.float64)
        if self._fill_values.shape[0] != len(FEATURE_COLUMNS):
            raise ModelArtifactsError("Preprocessor feature count does not match the schema")
        self._score_fn = self._build_score_fn(model, n_jobs)

    @classmethod
    def load(cls, model_dir: Path, model_name: str | None = None, n_jobs: int | None = None) -> "AnomalyPredictor":
        """Load artifacts. ``n_jobs`` caps inference threads (useful on small shared-CPU hosts)."""
        metadata = load_metadata(model_dir)
        name = model_name or metadata["selected_model"]
        if name not in metadata.get("models", {}):
            raise ModelArtifactsError(
                f"Model '{name}' is not listed in metadata. Available: {sorted(metadata.get('models', {}))}"
            )
        if tuple(metadata.get("feature_columns", ())) != FEATURE_COLUMNS:
            raise ModelArtifactsError("Saved feature columns do not match the current schema; retrain the model")
        return cls(
            preprocessor=load_preprocessor(model_dir),
            model=load_model(model_dir, name),
            model_name=name,
            threshold=metadata["models"][name]["threshold"],
            metadata=metadata,
            n_jobs=n_jobs,
        )

    @staticmethod
    def _build_score_fn(model: Any, n_jobs: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
        try:
            from xgboost import XGBClassifier
        except ImportError:  # pragma: no cover - xgboost is optional at runtime
            XGBClassifier = None  # type: ignore[assignment,misc]

        if XGBClassifier is not None and isinstance(model, XGBClassifier):
            booster = model.get_booster()
            if n_jobs is not None:
                booster.set_param({"nthread": n_jobs})
            best_iteration = getattr(model, "best_iteration", None)
            iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)

            def score_xgb(X: np.ndarray) -> np.ndarray:
                return np.asarray(
                    booster.inplace_predict(X, iteration_range=iteration_range, validate_features=False),
                    dtype=np.float64,
                )

            return score_xgb

        if not hasattr(model, "predict_proba"):
            raise ModelArtifactsError(f"Unsupported model type: {type(model).__name__}")
        single_threaded = model
        if "n_jobs" in model.get_params():
            single_threaded = copy.copy(model)
            single_threaded.set_params(n_jobs=1)
            if n_jobs is not None:
                model = copy.copy(model)
                model.set_params(n_jobs=n_jobs)

        def score_sklearn(X: np.ndarray) -> np.ndarray:
            estimator = single_threaded if X.shape[0] < SMALL_BATCH_ROWS else model
            return estimator.predict_proba(X)[:, 1]

        return score_sklearn

    def _complete_features(self, raw: np.ndarray) -> np.ndarray:
        n_inputs = len(MODEL_INPUT_FIELDS)
        features = np.empty((raw.shape[0], len(FEATURE_COLUMNS)), dtype=np.float64)
        features[:, :n_inputs] = raw
        derived = compute_derived_features({name: raw[:, i] for i, name in enumerate(MODEL_INPUT_FIELDS)})
        for offset, values in enumerate(derived.values()):
            features[:, n_inputs + offset] = values
        missing = np.isnan(features)
        if missing.any():
            features[missing] = self._fill_values[np.nonzero(missing)[1]]
        return features

    def features_from_records(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Build the model matrix from event dicts (only the required fields are read)."""
        raw = np.array(
            [[record.get(name) for name in MODEL_INPUT_FIELDS] for record in records], dtype=np.float64
        ).reshape(len(records), len(MODEL_INPUT_FIELDS))
        return self._complete_features(raw)

    def features_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame.loc[:, list(MODEL_INPUT_FIELDS)].to_numpy(dtype=np.float64, na_value=np.nan)
        return self._complete_features(raw)

    def score(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        return self._score_fn(features)

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        if not records:
            return []
        scores = self.score(self.features_from_records(records))
        return [
            Prediction(
                event_id=record.get("event_id"),
                vehicle_id=record.get("vehicle_id"),
                anomaly_score=float(score),
                is_anomaly=bool(score >= self.threshold),
            )
            for record, score in zip(records, scores)
        ]

    def predict_one(self, record: Mapping[str, Any]) -> Prediction:
        return self.predict_records([record])[0]

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``frame`` with ``anomaly_score`` and ``is_anomaly`` columns."""
        result = frame.copy()
        scores = self.score(self.features_from_frame(frame)) if len(frame) else np.empty(0)
        result["anomaly_score"] = scores
        result["is_anomaly"] = scores >= self.threshold
        return result

    def info(self) -> dict[str, Any]:
        model_meta = self.metadata.get("models", {}).get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "threshold": self.threshold,
            "inference_threads": self.n_jobs,
            "feature_columns": list(FEATURE_COLUMNS),
            "required_input_fields": list(MODEL_INPUT_FIELDS),
            "trained_at": self.metadata.get("created_at"),
            "selection_metric": self.metadata.get("selection_metric"),
            "validation_metrics": model_meta.get("validation_metrics"),
            "test_metrics": model_meta.get("test_metrics"),
            "library_versions": self.metadata.get("library_versions"),
            "data_source": self.metadata.get("data_source", "synthetic"),
        }


def main(argv: Sequence[str] | None = None) -> int:
    from vehicle_anomaly.config import load_config

    parser = argparse.ArgumentParser(description="Score telemetry events from a JSON or JSON Lines file.")
    parser.add_argument("input", type=Path, help="JSON object/array file or JSON Lines file")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Model name (defaults to the selected model)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    predictor = AnomalyPredictor.load(config.paths.model_dir, args.model)
    text = args.input.read_text(encoding="utf-8").strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    for prediction in predictor.predict_records(records):
        print(json.dumps(prediction.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
