from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vehicle_anomaly.api import create_app
from vehicle_anomaly.config import AppConfig
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


def test_demo_page_served_at_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Live stream simulation" in response.text and "/predict-batch" in response.text


def test_demo_events_are_scorable_and_unlabelled(client: TestClient) -> None:
    response = client.get("/demo/events", params={"n": 30, "anomaly_rate": 0.3, "seed": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["seed"] == 5 and len(body["events"]) == len(body["ground_truth"]) == 30
    assert all(set(event) == set(REQUIRED_FIELDS) for event in body["events"])
    assert [e["event_id"] for e in body["events"]] == [g["event_id"] for g in body["ground_truth"]]
    assert any(g["is_anomaly"] for g in body["ground_truth"])
    scored = client.post("/predict-batch", json={"events": body["events"]})
    assert scored.status_code == 200 and scored.json()["count"] == 30


def test_demo_events_reproducible_with_seed(client: TestClient) -> None:
    first = client.get("/demo/events", params={"n": 5, "seed": 11}).json()["events"]
    second = client.get("/demo/events", params={"n": 5, "seed": 11}).json()["events"]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "event_timestamp"} for r in rows]  # noqa: E731
    assert strip(first) == strip(second)


@pytest.mark.parametrize("params", [{"n": 0}, {"n": 201}, {"anomaly_rate": 0.9}, {"seed": -1}])
def test_demo_events_rejects_bad_parameters(client: TestClient, params: dict) -> None:
    assert client.get("/demo/events", params=params).status_code == 422


def test_demo_reports(client: TestClient) -> None:
    body = client.get("/demo/reports").json()
    assert set(body) == {"model_evaluation", "benchmark", "streaming"}
    assert body["model_evaluation"]["selected_model"]


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
