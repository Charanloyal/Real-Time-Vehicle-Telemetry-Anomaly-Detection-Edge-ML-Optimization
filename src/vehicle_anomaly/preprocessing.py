"""Feature engineering, preprocessing pipeline and leakage-safe dataset splitting."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from vehicle_anomaly.schema import FEATURE_COLUMNS, MODEL_INPUT_FIELDS


def compute_derived_features(columns: Mapping[str, Any]) -> dict[str, Any]:
    """Derived features from raw inputs. Works on pandas Series and NumPy arrays alike so the
    training pipeline and the optimized inference path share one definition."""
    speed = columns["speed"]
    rpm = columns["engine_rpm"]
    acceleration = columns["acceleration"]
    brake = columns["brake_pressure"]
    return {
        "rpm_per_speed": rpm / (speed + 1.0),
        "abs_acceleration": np.abs(acceleration),
        "brake_accel_product": brake * np.maximum(acceleration, 0.0),
    }


class TelemetryFeatureBuilder(BaseEstimator, TransformerMixin):
    """Selects model input fields from a telemetry DataFrame and appends derived features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TelemetryFeatureBuilder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in MODEL_INPUT_FIELDS if name not in X.columns]
        if missing:
            raise KeyError(f"Missing required feature column(s): {missing}")
        features = X.loc[:, list(MODEL_INPUT_FIELDS)].apply(pd.to_numeric, errors="coerce").astype("float64")
        for name, values in compute_derived_features(features).items():
            features[name] = values
        return features.loc[:, list(FEATURE_COLUMNS)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(FEATURE_COLUMNS, dtype=object)


def build_preprocessor() -> Pipeline:
    """Reproducible preprocessing: feature construction followed by median imputation."""
    return Pipeline(
        steps=[
            ("features", TelemetryFeatureBuilder()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )


def split_by_vehicle(
    df: pd.DataFrame, validation_size: float, test_size: float, seed: int, group_column: str = "vehicle_id"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/validation/test split grouped by vehicle so no vehicle appears in more than one split.

    Consecutive events of one vehicle are strongly correlated; a random row split would leak
    near-identical neighbours between training and evaluation data.
    """
    if not 0 < validation_size + test_size < 1:
        raise ValueError("validation_size + test_size must be between 0 and 1")
    groups = df[group_column]
    holdout = validation_size + test_size
    outer = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, holdout_idx = next(outer.split(df, groups=groups))
    holdout_df = df.iloc[holdout_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=test_size / holdout, random_state=seed + 1)
    val_idx, test_idx = next(inner.split(holdout_df, groups=holdout_df[group_column]))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        holdout_df.iloc[val_idx].reset_index(drop=True),
        holdout_df.iloc[test_idx].reset_index(drop=True),
    )
