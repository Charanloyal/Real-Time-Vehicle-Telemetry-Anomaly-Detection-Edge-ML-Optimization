"""FastAPI service for real-time anomaly scoring.

Run locally:  python scripts/run_api.py        (binds 0.0.0.0:$PORT, default 8000)
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from vehicle_anomaly import __version__
from vehicle_anomaly.artifacts import ModelArtifactsError
from vehicle_anomaly.config import AppConfig, load_config
from vehicle_anomaly.logging_utils import configure_logging, get_logger
from vehicle_anomaly.predict import AnomalyPredictor
from vehicle_anomaly.schema import VALID_RANGES

logger = get_logger(__name__)


def _range(name: str) -> dict[str, float]:
    low, high = VALID_RANGES[name]
    return {"ge": low, "le": high}


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
    speed: float = Field(description="km/h", allow_inf_nan=False, **_range("speed"))
    engine_rpm: float = Field(description="rev/min", allow_inf_nan=False, **_range("engine_rpm"))
    engine_temperature: float = Field(description="deg C", allow_inf_nan=False, **_range("engine_temperature"))
    fuel_level: float = Field(description="percent", allow_inf_nan=False, **_range("fuel_level"))
    battery_voltage: float = Field(description="volts", allow_inf_nan=False, **_range("battery_voltage"))
    brake_pressure: float = Field(description="bar", allow_inf_nan=False, **_range("brake_pressure"))
    acceleration: float = Field(description="m/s^2", allow_inf_nan=False, **_range("acceleration"))
    latitude: float = Field(allow_inf_nan=False, **_range("latitude"))
    longitude: float = Field(allow_inf_nan=False, **_range("longitude"))


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
    version: str
    detail: str | None = None


def _predictor(request: Request) -> AnomalyPredictor:
    predictor: AnomalyPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {request.app.state.load_error}",
        )
    return predictor


def _score(predictor: AnomalyPredictor, events: list[TelemetryEvent]) -> list[PredictionResponse]:
    records = [event.model_dump() for event in events]
    return [
        PredictionResponse(
            event_id=p.event_id or "",
            vehicle_id=p.vehicle_id or "",
            is_anomaly=p.is_anomaly,
            anomaly_score=round(p.anomaly_score, 6),
            threshold=predictor.threshold,
            model_name=predictor.model_name,
        )
        for p in predictor.predict_records(records)
    ]


def create_app(config: AppConfig | None = None, predictor: AnomalyPredictor | None = None) -> FastAPI:
    """Application factory. The model is loaded once at startup and reused for every request."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_config = config or load_config()
        configure_logging(app_config.logging.level, app_config.logging.json)
        app.state.config = app_config
        app.state.predictor = predictor
        app.state.load_error = None
        if app.state.predictor is None:
            try:
                app.state.predictor = AnomalyPredictor.load(app_config.paths.model_dir)
                logger.info("Model loaded", extra={"model": app.state.predictor.model_name,
                                                   "model_dir": str(app_config.paths.model_dir)})
            except (ModelArtifactsError, OSError, KeyError, ValueError) as exc:
                app.state.load_error = str(exc)
                logger.error("Model could not be loaded; serving in degraded mode", extra={"error": str(exc)})
        yield

    app = FastAPI(
        title="Vehicle Telemetry Anomaly Detection API",
        version=__version__,
        description="Scores vehicle telemetry events for anomalies. Models are trained on synthetic data.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
        logger.info("request", extra={"method": request.method, "path": request.url.path,
                                      "status_code": response.status_code, "duration_ms": round(elapsed_ms, 3)})
        return response

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        predictor_obj = getattr(request.app.state, "predictor", None)
        return HealthResponse(
            status="ok" if predictor_obj else "degraded",
            model_loaded=predictor_obj is not None,
            model_name=predictor_obj.model_name if predictor_obj else None,
            version=__version__,
            detail=None if predictor_obj else request.app.state.load_error,
        )

    @app.get("/model-info")
    async def model_info(request: Request) -> dict[str, Any]:
        return _predictor(request).info()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(event: TelemetryEvent, request: Request) -> PredictionResponse:
        return _score(_predictor(request), [event])[0]

    @app.post("/predict-batch", response_model=BatchPredictionResponse)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchPredictionResponse:
        predictor_obj = _predictor(request)
        max_size = request.app.state.config.api.max_batch_size
        if len(batch.events) > max_size:
            raise HTTPException(
                status_code=422,
                detail=f"Batch contains {len(batch.events)} events; the maximum is {max_size}.",
            )
        predictions = _score(predictor_obj, batch.events)
        return BatchPredictionResponse(
            model_name=predictor_obj.model_name,
            threshold=predictor_obj.threshold,
            count=len(predictions),
            anomaly_count=sum(p.is_anomaly for p in predictions),
            predictions=predictions,
        )

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
