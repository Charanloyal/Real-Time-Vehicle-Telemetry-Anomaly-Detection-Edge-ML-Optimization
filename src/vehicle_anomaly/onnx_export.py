"""ONNX export and ONNX Runtime scoring for edge / low-latency inference.

Tree ensembles are converted to the ONNX ``TreeEnsembleClassifier`` operator (scikit-learn via
skl2onnx, XGBoost via onnxmltools) and executed with ONNX Runtime. ONNX trees use float32 inputs and
thresholds, so scores can differ from the native float64 path in the last digits; every export is
checked against the native model (max |score difference| and decision agreement) at training time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

ONNX_OPSET = 15  # highest opset supported by the onnxmltools XGBoost converter


def onnx_status() -> tuple[bool, str]:
    try:
        import onnxruntime

        return True, onnxruntime.__version__
    except Exception as exc:  # ImportError or a broken native library
        return False, f"{type(exc).__name__}: {exc}"


def export_to_onnx(model: Any, n_features: int) -> bytes:
    """Serialize a fitted RandomForestClassifier or XGBClassifier to ONNX bytes."""
    try:
        from xgboost import XGBClassifier
    except ImportError:  # pragma: no cover
        XGBClassifier = None  # type: ignore[assignment,misc]  # noqa: N806

    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as XgbFloatTensorType

        onnx_model = convert_xgboost(
            model, initial_types=[("input", XgbFloatTensorType([None, n_features]))], target_opset=ONNX_OPSET
        )
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onnx_model = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType([None, n_features]))],
            options={id(model): {"zipmap": False}},
            target_opset={"": ONNX_OPSET, "ai.onnx.ml": 3},
        )
    return bytes(onnx_model.SerializeToString())


class OnnxScorer:
    """Callable returning the positive-class probability for a float feature matrix."""

    def __init__(self, model: bytes | Path, n_threads: int | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread is the edge-like default: single-row tree scoring does not benefit from more.
        options.intra_op_num_threads = n_threads or 1
        options.inter_op_num_threads = 1
        source = str(model) if isinstance(model, Path) else model
        self.session = ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        outputs = [o.name for o in self.session.get_outputs()]
        self.output_name = next((name for name in outputs if "prob" in name.lower()), outputs[-1])

    def __call__(self, features: np.ndarray) -> np.ndarray:
        result = self.session.run([self.output_name], {self.input_name: features.astype(np.float32, copy=False)})
        return np.asarray(result[0], dtype=np.float64)[:, 1]


def parity_report(native_scores: np.ndarray, onnx_scores: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "max_abs_score_diff": float(np.max(np.abs(native_scores - onnx_scores))),
        "decision_agreement": float(np.mean((native_scores >= threshold) == (onnx_scores >= threshold))),
        "events_checked": len(native_scores),
    }
