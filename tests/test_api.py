from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vehicle_anomaly.api import create_app
from vehicle_anomaly.config import AppConfig
from vehicle_anomaly.data_generator import DEMO_STREAM_LENGTH
from vehicle_anomaly.schema import LABEL_COLUMN, REQUIRED_FIELDS
from vehicle_anomaly.train import TEST_SET_FILE


@pytest.fixture(scope="module")
def client(trained_config: AppConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config=trained_config)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def events(trained_config: AppConfig) -> list[dict]:
    frame = pd.read_parquet(trained_config.paths.processed_dir / TEST_SET_FILE)
    sample = pd.concat([frame[frame[LABEL_COLUMN] == 1].head(5), frame[frame[LABEL_COLUMN] == 0].head(5)])
    return sample.loc[:, list(REQUIRED_FIELDS)].to_dict("records")


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["model_name"]


def test_model_info(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert body["model_name"] and 0 < body["threshold"] < 1
    assert "speed" in body["required_input_fields"]
    assert body["data_source"].startswith("synthetic")


def test_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert "/predict-batch" in client.get("/openapi.json").json()["paths"]


def test_predict_single(client: TestClient, events: list[dict]) -> None:
    response = client.post("/predict", json=events[0])
    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == events[0]["event_id"]
    assert 0 <= body["anomaly_score"] <= 1
    assert body["is_anomaly"] == (body["anomaly_score"] >= body["threshold"])
    assert "x-process-time-ms" in response.headers


def test_predict_batch(client: TestClient, events: list[dict]) -> None:
    response = client.post("/predict-batch", json={"events": events})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(events) == len(body["predictions"])
    assert body["anomaly_count"] == sum(p["is_anomaly"] for p in body["predictions"])
    assert [p["event_id"] for p in body["predictions"]] == [e["event_id"] for e in events]


def test_batch_and_single_scores_agree(client: TestClient, events: list[dict]) -> None:
    batch = client.post("/predict-batch", json={"events": events}).json()["predictions"]
    single = [client.post("/predict", json=e).json() for e in events]
    assert [p["anomaly_score"] for p in batch] == [p["anomaly_score"] for p in single]


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"speed": "fast"}, "speed"),
        ({"battery_voltage": 99.0}, "battery_voltage"),
        ({"event_timestamp": "not-a-date"}, "event_timestamp"),
        ({"event_id": ""}, "event_id"),
    ],
)
def test_predict_validation_errors(client: TestClient, events: list[dict], change: dict, field: str) -> None:
    response = client.post("/predict", json={**events[0], **change})
    assert response.status_code == 422
    assert any(field in error["loc"] for error in response.json()["detail"])


def test_predict_missing_field(client: TestClient, events: list[dict]) -> None:
    payload = {k: v for k, v in events[0].items() if k != "engine_rpm"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "engine_rpm"


def test_empty_batch_rejected(client: TestClient) -> None:
    assert client.post("/predict-batch", json={"events": []}).status_code == 422


def test_batch_size_limit(trained_config: AppConfig, events: list[dict]) -> None:
    config = replace(trained_config, api=replace(trained_config.api, max_batch_size=3))
    with TestClient(create_app(config=config)) as small_client:
        response = small_client.post("/predict-batch", json={"events": events})
    assert response.status_code == 422
    assert "maximum is 3" in response.json()["detail"]


def test_degraded_mode_without_model(trained_config: AppConfig, tmp_path, events: list[dict]) -> None:
    config = replace(trained_config, paths=replace(trained_config.paths, model_dir=tmp_path))
    with TestClient(create_app(config=config)) as degraded:
        health_response = degraded.get("/health")
        assert health_response.status_code == 503  # lets platform health checks reject the deploy
        health = health_response.json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        response = degraded.post("/predict", json=events[0])
        assert response.status_code == 503
        assert "train_model.py" in response.json()["detail"]
        stream = degraded.post("/predict-stream", json={"events": events})
        assert stream.status_code == 503 and "Window model not loaded" in stream.json()["detail"]


def test_demo_page_served_at_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "point model vs window model" in response.text and "/predict-stream" in response.text


def test_demo_events_are_contiguous_and_unlabelled(client: TestClient) -> None:
    first = client.get("/demo/events", params={"seed": 5, "start": 0, "n": 30}).json()
    second = client.get("/demo/events", params={"seed": 5, "start": 30, "n": 30}).json()
    whole = client.get("/demo/events", params={"seed": 5, "start": 0, "n": 60}).json()
    assert first["seed"] == 5 and first["next_start"] == 30 and second["next_start"] == 60
    assert first["events"] + second["events"] == whole["events"]
    assert all(set(event) == set(REQUIRED_FIELDS) for event in first["events"])
    assert [e["event_id"] for e in first["events"]] == [g["event_id"] for g in first["ground_truth"]]
    scored = client.post("/predict-batch", json={"events": first["events"]})
    assert scored.status_code == 200 and scored.json()["count"] == 30


def test_demo_events_end_of_stream(client: TestClient) -> None:
    body = client.get("/demo/events", params={"seed": 5, "start": DEMO_STREAM_LENGTH - 10, "n": 50}).json()
    assert len(body["events"]) == 10 and body["next_start"] == body["stream_length"] == DEMO_STREAM_LENGTH


@pytest.mark.parametrize(
    "params",
    [
        {"n": 0},
        {"n": 201},
        {"anomaly_rate": 0.9},
        {"seed": -1},
        {"start": -1},
        {"start": DEMO_STREAM_LENGTH},
        {"drift_rate": 50},
    ],
)
def test_demo_events_rejects_bad_parameters(client: TestClient, params: dict) -> None:
    assert client.get("/demo/events", params=params).status_code == 422


def test_predict_stream_state_round_trip(client: TestClient) -> None:
    events = client.get("/demo/events", params={"seed": 9, "start": 0, "n": 200}).json()["events"]
    single = client.post("/predict-stream", json={"events": events}).json()
    first = client.post("/predict-stream", json={"events": events[:120]}).json()
    second = client.post("/predict-stream", json={"events": events[120:], "state": first["state"]}).json()
    assert [p["anomaly_score"] for p in first["predictions"] + second["predictions"]] == [
        p["anomaly_score"] for p in single["predictions"]
    ]
    assert second["state"] == single["state"]
    assert set(single["state"]) == {e["vehicle_id"] for e in events}
    assert single["model_name"] == client.get("/health").json()["window_model_name"]
    restarted = client.post("/predict-stream", json={"events": events[120:]}).json()
    assert [p["anomaly_score"] for p in restarted["predictions"]] != [p["anomaly_score"] for p in second["predictions"]]


@pytest.mark.parametrize(
    "state",
    [
        {"V": {"temp_long": "hot", "temp_short": 1, "battery_long": 1, "battery_short": 1, "count": 1}},
        {"V": {"temp_long": 90, "temp_short": 90, "battery_long": 14, "battery_short": 14, "count": -1}},
        {"V": {"temp_long": 90}},
        {
            f"V{i}": {"temp_long": 90, "temp_short": 90, "battery_long": 14, "battery_short": 14, "count": 1}
            for i in range(1001)
        },
    ],
)
def test_predict_stream_rejects_invalid_state(client: TestClient, events: list[dict], state: dict) -> None:
    assert client.post("/predict-stream", json={"events": events, "state": state}).status_code == 422


def test_model_info_includes_window_model(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert body["feature_set"] == "point" and body["runtime"] == "native"
    assert body["window_model"]["feature_set"] == "window"
    assert "temp_dev" in body["window_model"]["feature_columns"]


def test_onnx_runtime_app(trained_config: AppConfig, client: TestClient, events: list[dict]) -> None:
    config = replace(trained_config, api=replace(trained_config.api, runtime="onnx", inference_threads=1))
    native = client.post("/predict-batch", json={"events": events}).json()["predictions"]
    with TestClient(create_app(config=config)) as onnx_client:
        health = onnx_client.get("/health").json()
        assert health["runtime"] == "onnx" and health["status"] == "ok"
        scored = onnx_client.post("/predict-batch", json={"events": events}).json()["predictions"]
        assert onnx_client.post("/predict-stream", json={"events": events}).status_code == 200
    for a, b in zip(native, scored, strict=True):
        assert a["anomaly_score"] == pytest.approx(b["anomaly_score"], abs=1e-5)


def test_demo_reports(client: TestClient) -> None:
    body = client.get("/demo/reports").json()
    assert set(body) == {"model_evaluation", "benchmark", "streaming"}
    assert body["model_evaluation"]["selected_model"] and body["model_evaluation"]["selected_window_model"]


def test_demo_figures(client: TestClient) -> None:
    response = client.get("/demo/figures/confusion_matrices.png")
    assert response.status_code == 200 and response.content[:4] == b"\x89PNG"
    assert client.get("/demo/figures/missing.png").status_code == 404
    assert client.get("/demo/figures/..%2Fmodel_evaluation.json").status_code == 404
    assert client.get("/demo/figures/model_evaluation.json").status_code == 404


def test_inference_threads_setting(trained_config: AppConfig, events: list[dict]) -> None:
    config = replace(trained_config, api=replace(trained_config.api, inference_threads=1))
    with TestClient(create_app(config=config)) as limited:
        assert limited.get("/model-info").json()["inference_threads"] == 1
        assert limited.post("/predict-batch", json={"events": events}).status_code == 200
