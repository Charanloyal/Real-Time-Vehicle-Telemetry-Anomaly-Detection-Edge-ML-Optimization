"""Prediction interfaces used by the API, the streaming pipeline and the benchmark.

``AnomalyPredictor`` (point models) is the optimized path. Compared with the reference path
(``pandas.DataFrame`` -> sklearn ``Pipeline`` -> ``predict_proba``), measured in ``benchmark.py``:

1. Feature extraction builds a float64 NumPy matrix directly from the required input fields only,
   skipping DataFrame construction, dtype coercion and column selection.
2. The fitted imputer statistics are applied with NumPy instead of calling the sklearn pipeline.
3. Model and preprocessing objects are loaded once and reused.
4. Low-overhead model calls: XGBoost uses ``Booster.inplace_predict``; scikit-learn forests score
   small batches single-threaded to avoid thread-pool start-up cost.
5. Batch prediction scores many events with one vectorized call.
6. Optional ``runtime="onnx"``: the same trees exported to ONNX and executed by ONNX Runtime.

Native scores are identical to the reference path; ONNX scores agree within float32 precision
(both verified in tests and in the benchmark).

``WindowAnomalyPredictor`` scores models trained with per-vehicle history features and carries the
per-vehicle state between calls (see ``window_features.py``).
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vehicle_anomaly.artifacts import ModelArtifactsError, load_metadata, load_model, load_preprocessor
from vehicle_anomaly.preprocessing import compute_derived_features
from vehicle_anomaly.schema import FEATURE_COLUMNS, FEATURE_SETS, MODEL_INPUT_FIELDS
from vehicle_anomaly.window_features import StateMap, VehicleState, compute_window_features

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


def _model_entry(metadata: dict[str, Any], name: str) -> dict[str, Any]:
    models: dict[str, Any] = metadata.get("models", {})
    if name not in models:
        raise ModelArtifactsError(f"Model '{name}' is not listed in metadata. Available: {sorted(models)}")
    entry: dict[str, Any] = models[name]
    return entry


class AnomalyPredictor:
    """Loads artifacts once and scores telemetry events with a point (single-event) model."""

    def __init__(
        self,
        preprocessor: Any,
        model: Any,
        model_name: str,
        threshold: float,
        metadata: dict[str, Any] | None = None,
        n_jobs: int | None = None,
        runtime: str = "native",
        onnx_path: Path | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.model_name = model_name
        self.threshold = float(threshold)
        self.metadata = metadata or {}
        self.n_jobs = n_jobs
        self.runtime = runtime
        imputer = preprocessor.named_steps["imputer"]
        self._fill_values = np.asarray(imputer.statistics_, dtype=np.float64)
        if self._fill_values.shape[0] != len(FEATURE_COLUMNS):
            raise ModelArtifactsError("Preprocessor feature count does not match the schema")
        if runtime == "onnx":
            if onnx_path is None or not onnx_path.is_file():
                raise ModelArtifactsError(f"ONNX runtime requested but no ONNX file exists for '{model_name}'")
            from vehicle_anomaly.onnx_export import OnnxScorer

            self._score_fn: Callable[[np.ndarray], np.ndarray] = OnnxScorer(onnx_path, n_jobs)
        elif runtime == "native":
            self._score_fn = self._build_score_fn(model, n_jobs)
        else:
            raise ValueError(f"Unknown runtime '{runtime}' (use 'native' or 'onnx')")

    @classmethod
    def load(
        cls,
        model_dir: Path,
        model_name: str | None = None,
        n_jobs: int | None = None,
        runtime: str = "native",
        feature_set: str = "point",
    ) -> AnomalyPredictor:
        """Load artifacts. ``n_jobs`` caps inference threads (useful on small shared-CPU hosts)."""
        metadata = load_metadata(model_dir)
        if tuple(metadata.get("feature_columns", ())) != FEATURE_COLUMNS:
            raise ModelArtifactsError("Saved feature columns do not match the current schema; retrain the model")
        default_key = "selected_model" if feature_set == "point" else "selected_window_model"
        name = model_name or metadata.get(default_key)
        if not name:
            raise ModelArtifactsError(f"No {feature_set} model was trained; retrain with a {feature_set} model")
        entry = _model_entry(metadata, name)
        actual = entry.get("feature_set", "point")
        if actual != feature_set:
            raise ModelArtifactsError(f"Model '{name}' uses the '{actual}' feature set, not '{feature_set}'")
        if not entry.get("supervised", True):
            raise ModelArtifactsError(f"Model '{name}' is an unsupervised baseline and is not served")
        onnx_meta = entry.get("onnx") or {}
        return cls(
            preprocessor=load_preprocessor(model_dir),
            model=load_model(model_dir, name),
            model_name=name,
            threshold=entry["threshold"],
            metadata=metadata,
            n_jobs=n_jobs,
            runtime=runtime,
            onnx_path=model_dir / onnx_meta["file"] if onnx_meta.get("file") else None,
        )

    @staticmethod
    def _build_score_fn(model: Any, n_jobs: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
        try:
            from xgboost import XGBClassifier
        except ImportError:  # pragma: no cover - xgboost is optional at runtime
            XGBClassifier = None  # type: ignore[assignment,misc]  # noqa: N806

        if XGBClassifier is not None and isinstance(model, XGBClassifier):
            booster = model.get_booster()
            if n_jobs is not None:
                booster.set_param({"nthread": n_jobs})
            best_iteration = getattr(model, "best_iteration", None)
            iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)

            def score_xgb(features: np.ndarray) -> np.ndarray:
                return np.asarray(
                    booster.inplace_predict(features, iteration_range=iteration_range, validate_features=False),
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

        def score_sklearn(features: np.ndarray) -> np.ndarray:
            estimator = single_threaded if features.shape[0] < SMALL_BATCH_ROWS else model
            return np.asarray(estimator.predict_proba(features)[:, 1], dtype=np.float64)

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

    def _predictions(self, records: Sequence[Mapping[str, Any]], scores: np.ndarray) -> list[Prediction]:
        return [
            Prediction(
                event_id=record.get("event_id"),
                vehicle_id=record.get("vehicle_id"),
                anomaly_score=float(score),
                is_anomaly=bool(score >= self.threshold),
            )
            for record, score in zip(records, scores, strict=True)
        ]

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        if not records:
            return []
        return self._predictions(records, self.score(self.features_from_records(records)))

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
        feature_set = model_meta.get("feature_set", "point")
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "feature_set": feature_set,
            "runtime": self.runtime,
            "threshold": self.threshold,
            "inference_threads": self.n_jobs,
            "feature_columns": list(FEATURE_SETS[feature_set]),
            "required_input_fields": list(MODEL_INPUT_FIELDS),
            "trained_at": self.metadata.get("created_at"),
            "selection_metric": self.metadata.get("selection_metric"),
            "validation_metrics": model_meta.get("validation_metrics"),
            "test_metrics": model_meta.get("test_metrics"),
            "test_recall_by_type": model_meta.get("test_recall_by_type"),
            "onnx": model_meta.get("onnx"),
            "library_versions": self.metadata.get("library_versions"),
            "data_source": self.metadata.get("data_source", "synthetic"),
        }


class WindowAnomalyPredictor:
    """Scores events with a window model, carrying per-vehicle history state between calls."""

    def __init__(self, base: AnomalyPredictor) -> None:
        self.base = base

    @classmethod
    def load(
        cls, model_dir: Path, model_name: str | None = None, n_jobs: int | None = None, runtime: str = "native"
    ) -> WindowAnomalyPredictor:
        return cls(AnomalyPredictor.load(model_dir, model_name, n_jobs, runtime, feature_set="window"))

    @property
    def model_name(self) -> str:
        return self.base.model_name

    @property
    def threshold(self) -> float:
        return self.base.threshold

    def score_frame(
        self, frame: pd.DataFrame, state: Mapping[str, VehicleState] | None = None
    ) -> tuple[np.ndarray, StateMap]:
        """Scores aligned to ``frame`` rows and the state after the last event of each vehicle."""
        if frame.empty:
            return np.empty(0, dtype=np.float64), dict(state or {})
        window, new_state = compute_window_features(frame, state)
        features = np.hstack([self.base.features_from_frame(frame), window.to_numpy(dtype=np.float64)])
        return self.base.score(features), new_state

    def predict_frame(
        self, frame: pd.DataFrame, state: Mapping[str, VehicleState] | None = None
    ) -> tuple[pd.DataFrame, StateMap]:
        scores, new_state = self.score_frame(frame, state)
        result = frame.copy()
        result["anomaly_score"] = scores
        result["is_anomaly"] = scores >= self.threshold
        return result, new_state

    def predict_records(
        self, records: Sequence[Mapping[str, Any]], state: Mapping[str, VehicleState] | None = None
    ) -> tuple[list[Prediction], StateMap]:
        if not records:
            return [], dict(state or {})
        scores, new_state = self.score_frame(pd.DataFrame(list(records)), state)
        return self.base._predictions(records, scores), new_state

    def info(self) -> dict[str, Any]:
        info = self.base.info()
        info["window_feature_settings"] = self.base.metadata.get("window_feature_settings")
        return info


def main(argv: Sequence[str] | None = None) -> int:
    from vehicle_anomaly.config import load_config

    parser = argparse.ArgumentParser(description="Score telemetry events from a JSON or JSON Lines file.")
    parser.add_argument("input", type=Path, help="JSON object/array file or JSON Lines file")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Point model name (defaults to the selected model)")
    parser.add_argument("--runtime", choices=["native", "onnx"], default="native")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    predictor = AnomalyPredictor.load(config.paths.model_dir, args.model, runtime=args.runtime)
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
