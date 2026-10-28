"""FastAPI service for real-time anomaly scoring, plus a browser demo.

Run locally:  python scripts/run_api.py        (binds 0.0.0.0:$PORT, default 8000)
Demo UI:      http://localhost:8000/
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from vehicle_anomaly import __version__
from vehicle_anomaly.artifacts import ModelArtifactsError
from vehicle_anomaly.config import AppConfig, load_config
from vehicle_anomaly.data_generator import DEMO_STREAM_LENGTH, demo_fleet_events
from vehicle_anomaly.logging_utils import configure_logging, get_logger
from vehicle_anomaly.predict import AnomalyPredictor, Prediction, WindowAnomalyPredictor
from vehicle_anomaly.schema import ANOMALY_TYPE_COLUMN, LABEL_COLUMN, REQUIRED_FIELDS, VALID_RANGES
from vehicle_anomaly.window_features import StateMap, VehicleState, state_to_json

logger = get_logger(__name__)
DEMO_MAX_EVENTS = 200
MAX_STATE_VEHICLES = 1000
REPORT_FILES = {
    "model_evaluation": "model_evaluation.json",
    "benchmark": "benchmark_results.json",
    "streaming": "streaming_summary.json",
}
_FIGURE_NAME = re.compile(r"^[a-z0-9_]+\.png$")


_LOW = {name: bounds[0] for name, bounds in VALID_RANGES.items()}
_HIGH = {name: bounds[1] for name, bounds in VALID_RANGES.items()}


class TelemetryEvent(BaseModel):
    """A single vehicle telemetry reading. Ranges match the streaming validation rules."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "event_id": "VEH-0001-0000042",
                "vehicle_id": "VEH-0001",
                "event_timestamp": "2026-01-01T00:10:42.000Z",
                "speed": 72.4,
                "engine_rpm": 2950.0,
                "engine_temperature": 91.3,
                "fuel_level": 64.2,
                "battery_voltage": 14.18,
                "brake_pressure": 0.8,
                "acceleration": 0.35,
                "latitude": 40.112233,
                "longitude": -95.223344,
            }
        },
    )

    event_id: str = Field(min_length=1, max_length=128)
    vehicle_id: str = Field(min_length=1, max_length=64)
    event_timestamp: datetime
    speed: float = Field(description="km/h", allow_inf_nan=False, ge=_LOW["speed"], le=_HIGH["speed"])
    engine_rpm: float = Field(description="rev/min", allow_inf_nan=False, ge=_LOW["engine_rpm"], le=_HIGH["engine_rpm"])
    engine_temperature: float = Field(
        description="deg C", allow_inf_nan=False, ge=_LOW["engine_temperature"], le=_HIGH["engine_temperature"]
    )
    fuel_level: float = Field(description="percent", allow_inf_nan=False, ge=_LOW["fuel_level"], le=_HIGH["fuel_level"])
    battery_voltage: float = Field(
        description="volts", allow_inf_nan=False, ge=_LOW["battery_voltage"], le=_HIGH["battery_voltage"]
    )
    brake_pressure: float = Field(
        description="bar", allow_inf_nan=False, ge=_LOW["brake_pressure"], le=_HIGH["brake_pressure"]
    )
    acceleration: float = Field(
        description="m/s^2", allow_inf_nan=False, ge=_LOW["acceleration"], le=_HIGH["acceleration"]
    )
    latitude: float = Field(allow_inf_nan=False, ge=_LOW["latitude"], le=_HIGH["latitude"])
    longitude: float = Field(allow_inf_nan=False, ge=_LOW["longitude"], le=_HIGH["longitude"])


class BatchRequest(BaseModel):
    events: list[TelemetryEvent] = Field(min_length=1)


class PredictionResponse(BaseModel):
    event_id: str
    vehicle_id: str
    is_anomaly: bool
    anomaly_score: float = Field(description="Model probability that the event is anomalous")
    threshold: float
    model_name: str


class BatchPredictionResponse(BaseModel):
    model_name: str
    threshold: float
    count: int
    anomaly_count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    window_model_name: str | None = None
    runtime: str | None = None
    version: str
    detail: str | None = None


class VehicleStateModel(BaseModel):
    """Per-vehicle EWMA state (see window_features.py). Treat as opaque: send back what you received."""

    temp_long: float = Field(allow_inf_nan=False, ge=-100, le=400)
    temp_short: float = Field(allow_inf_nan=False, ge=-100, le=400)
    battery_long: float = Field(allow_inf_nan=False, ge=-10, le=60)
    battery_short: float = Field(allow_inf_nan=False, ge=-10, le=60)
    count: int = Field(ge=0, le=10**12)


class StreamRequest(BaseModel):
    events: list[TelemetryEvent] = Field(min_length=1)
    state: dict[str, VehicleStateModel] | None = Field(
        default=None,
        max_length=MAX_STATE_VEHICLES,
        description="The `state` returned by the previous call (omit or null for a new stream)",
    )


class StreamPredictionResponse(BaseModel):
    model_name: str
    threshold: float
    count: int
    anomaly_count: int
    predictions: list[PredictionResponse]
    state: dict[str, VehicleStateModel] = Field(description="Send this back with the next batch of events")


class GroundTruth(BaseModel):
    event_id: str
    is_anomaly: bool
    anomaly_type: str


class DemoEventsResponse(BaseModel):
    seed: int
    start: int
    next_start: int
    stream_length: int
    note: str
    events: list[dict[str, Any]]
    ground_truth: list[GroundTruth]


def _load_demo_page() -> str:
    return resources.files("vehicle_anomaly").joinpath("static", "demo.html").read_text(encoding="utf-8")


def _read_report(path: Path) -> dict[str, Any] | None:
    try:
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return report
    except (OSError, ValueError):
        return None


def _predictor(request: Request) -> AnomalyPredictor:
    predictor: AnomalyPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {request.app.state.load_error}",
        )
    return predictor


def _window_predictor(request: Request) -> WindowAnomalyPredictor:
    predictor: WindowAnomalyPredictor | None = getattr(request.app.state, "window_predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Window model not loaded: {request.app.state.window_load_error}",
        )
    return predictor


def _responses(predictions: list[Prediction], threshold: float, model_name: str) -> list[PredictionResponse]:
    return [
        PredictionResponse(
            event_id=p.event_id or "",
            vehicle_id=p.vehicle_id or "",
            is_anomaly=p.is_anomaly,
            anomaly_score=round(p.anomaly_score, 6),
            threshold=threshold,
            model_name=model_name,
        )
        for p in predictions
    ]


def _score(predictor: AnomalyPredictor, events: list[TelemetryEvent]) -> list[PredictionResponse]:
    records = [event.model_dump() for event in events]
    return _responses(predictor.predict_records(records), predictor.threshold, predictor.model_name)


def _check_batch_size(request: Request, size: int) -> None:
    max_size = request.app.state.config.api.max_batch_size
    if size > max_size:
        raise HTTPException(status_code=422, detail=f"Batch contains {size} events; the maximum is {max_size}.")


def create_app(config: AppConfig | None = None, predictor: AnomalyPredictor | None = None) -> FastAPI:
    """Application factory. Models are loaded once at startup and reused for every request."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_config = config or load_config()
        configure_logging(app_config.logging.level, app_config.logging.json)
        api_cfg = app_config.api
        app.state.config = app_config
        app.state.predictor = predictor
        app.state.load_error = None
        app.state.window_predictor = None
        app.state.window_load_error = None
        load_errors = (ModelArtifactsError, OSError, KeyError, ValueError, ImportError)
        if app.state.predictor is None:
            try:
                app.state.predictor = AnomalyPredictor.load(
                    app_config.paths.model_dir, n_jobs=api_cfg.inference_threads, runtime=api_cfg.runtime
                )
                logger.info(
                    "Model loaded",
                    extra={
                        "model": app.state.predictor.model_name,
                        "model_dir": str(app_config.paths.model_dir),
                        "runtime": api_cfg.runtime,
                        "inference_threads": api_cfg.inference_threads,
                    },
                )
            except load_errors as exc:
                app.state.load_error = str(exc)
                logger.error("Model could not be loaded; serving in degraded mode", extra={"error": str(exc)})
        try:
            app.state.window_predictor = WindowAnomalyPredictor.load(
                app_config.paths.model_dir, n_jobs=api_cfg.inference_threads, runtime=api_cfg.runtime
            )
            logger.info("Window model loaded", extra={"model": app.state.window_predictor.model_name})
        except load_errors as exc:
            app.state.window_load_error = str(exc)
            logger.warning("Window model unavailable; /predict-stream disabled", extra={"error": str(exc)})
        yield

    app = FastAPI(
        title="Vehicle Telemetry Anomaly Detection API",
        version=__version__,
        description=(
            "Scores vehicle telemetry events for anomalies. `/predict` and `/predict-batch` use a point model "
            "(one event at a time); `/predict-stream` uses a window model with per-vehicle history features. "
            "Models are trained on synthetic data."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(elapsed_ms, 3),
            },
        )
        return response

    demo_page = _load_demo_page()

    @app.get("/", include_in_schema=False)
    async def root() -> HTMLResponse:
        return HTMLResponse(demo_page)

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    async def health(request: Request, response: Response) -> HealthResponse:
        predictor_obj = getattr(request.app.state, "predictor", None)
        window_obj = getattr(request.app.state, "window_predictor", None)
        if predictor_obj is None:
            # 503 makes platform health checks (e.g. Render) reject a deploy without a usable model.
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if predictor_obj else "degraded",
            model_loaded=predictor_obj is not None,
            model_name=predictor_obj.model_name if predictor_obj else None,
            window_model_name=window_obj.model_name if window_obj else None,
            runtime=request.app.state.config.api.runtime,
            version=__version__,
            detail=None if predictor_obj else request.app.state.load_error,
        )

    @app.get("/model-info")
    async def model_info(request: Request) -> dict[str, Any]:
        info = _predictor(request).info()
        window_obj = getattr(request.app.state, "window_predictor", None)
        info["window_model"] = window_obj.info() if window_obj else None
        return info

    @app.post("/predict", response_model=PredictionResponse)
    def predict(event: TelemetryEvent, request: Request) -> PredictionResponse:
        """Score one event with the point model (no history needed)."""
        return _score(_predictor(request), [event])[0]

    @app.post("/predict-batch", response_model=BatchPredictionResponse)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchPredictionResponse:
        """Score independent events with the point model in one vectorized call."""
        predictor_obj = _predictor(request)
        _check_batch_size(request, len(batch.events))
        predictions = _score(predictor_obj, batch.events)
        return BatchPredictionResponse(
            model_name=predictor_obj.model_name,
            threshold=predictor_obj.threshold,
            count=len(predictions),
            anomaly_count=sum(p.is_anomaly for p in predictions),
            predictions=predictions,
        )

    @app.post("/predict-stream", response_model=StreamPredictionResponse)
    def predict_stream(body: StreamRequest, request: Request) -> StreamPredictionResponse:
        """Score a stream batch with the window model (per-vehicle history features).

        The server is stateless: it returns the per-vehicle state after this batch, and the client sends
        it back with the next batch. Events are processed per vehicle in timestamp order.
        """
        window_obj = _window_predictor(request)
        _check_batch_size(request, len(body.events))
        state: StateMap = {k: VehicleState(**v.model_dump()) for k, v in (body.state or {}).items()}
        records = [event.model_dump() for event in body.events]
        predictions, new_state = window_obj.predict_records(records, state)
        if len(new_state) > MAX_STATE_VEHICLES:
            raise HTTPException(
                status_code=422, detail=f"State would exceed {MAX_STATE_VEHICLES} vehicles; start a new stream."
            )
        responses = _responses(predictions, window_obj.threshold, window_obj.model_name)
        return StreamPredictionResponse(
            model_name=window_obj.model_name,
            threshold=window_obj.threshold,
            count=len(responses),
            anomaly_count=sum(p.is_anomaly for p in responses),
            predictions=responses,
            state={k: VehicleStateModel.model_validate(v) for k, v in state_to_json(new_state).items()},
        )

    @app.get("/demo/events", response_model=DemoEventsResponse, tags=["demo"])
    def demo_events(
        seed: int | None = Query(None, ge=0, le=2**31 - 1, description="Stream id; omit to start a new stream"),
        start: int = Query(0, ge=0, lt=DEMO_STREAM_LENGTH, description="Position in the stream"),
        n: int = Query(20, ge=1, le=DEMO_MAX_EVENTS, description="Number of events"),
        anomaly_rate: float = Query(0.05, ge=0.0, le=0.5, description="Share of single-event faults"),
        drift_rate: float = Query(3.0, ge=0.0, le=10.0, description="Gradual-fault episodes per 1000 events"),
    ) -> DemoEventsResponse:
        """Next slice of a deterministic synthetic 4-vehicle stream. Ground-truth labels are returned
        separately and are never part of the event payload sent to the model."""
        seed_value = seed if seed is not None else time.time_ns() % (2**31)
        count = min(n, DEMO_STREAM_LENGTH - start)
        frame = demo_fleet_events(seed_value, start, count, anomaly_rate, drift_rate)
        return DemoEventsResponse(
            seed=seed_value,
            start=start,
            next_start=start + count,
            stream_length=DEMO_STREAM_LENGTH,
            note="Synthetic events from vehicle_anomaly.data_generator; labels are for comparison only.",
            events=cast(list[dict[str, Any]], frame.loc[:, list(REQUIRED_FIELDS)].to_dict("records")),
            ground_truth=[
                GroundTruth(
                    event_id=str(row.event_id),
                    is_anomaly=bool(getattr(row, LABEL_COLUMN)),
                    anomaly_type=str(getattr(row, ANOMALY_TYPE_COLUMN)),
                )
                for row in frame.itertuples(index=False)
            ],
        )

    @app.get("/demo/reports", tags=["demo"])
    def demo_reports(request: Request) -> dict[str, Any]:
        """Offline reports: model evaluation, inference benchmark and streaming run summary."""
        reports_dir = request.app.state.config.paths.reports_dir
        return {key: _read_report(reports_dir / name) for key, name in REPORT_FILES.items()}

    @app.get("/demo/figures/{name}", tags=["demo"], response_class=FileResponse)
    def demo_figure(name: str, request: Request) -> FileResponse:
        figure = request.app.state.config.paths.figures_dir / name
        if not _FIGURE_NAME.fullmatch(name) or not figure.is_file():
            raise HTTPException(status_code=404, detail="Figure not found")
        return FileResponse(figure, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    return app


app = create_app()


def main() -> None:
    import uvicorn

    config = load_config()
    configure_logging(config.logging.level, config.logging.json)
    port = int(os.environ.get("PORT", config.api.default_port))
    host = os.environ.get("HOST", "0.0.0.0")
    # Request logging is done by the timing middleware, so uvicorn's access log is disabled.
    uvicorn.run(app, host=host, port=port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
