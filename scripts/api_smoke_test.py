"""Smoke-test a running API: /health, /model-info, /predict and /predict-batch.

Usage: python scripts/api_smoke_test.py [--url http://localhost:8000] [--events 50]
Events are taken from the held-out test split created during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

from vehicle_anomaly.config import load_config
from vehicle_anomaly.schema import LABEL_COLUMN, REQUIRED_FIELDS
from vehicle_anomaly.train import TEST_SET_FILE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    test = pd.read_parquet(config.paths.processed_dir / TEST_SET_FILE)
    sample = pd.concat(
        [test[test[LABEL_COLUMN] == 1].head(args.events // 2), test[test[LABEL_COLUMN] == 0].head(args.events // 2)]
    )
    events = sample.loc[:, list(REQUIRED_FIELDS)].to_dict("records")

    with httpx.Client(base_url=args.url, timeout=30) as client:
        health = client.get("/health")
        print("GET /health ->", health.status_code, health.json())
        info = client.get("/model-info")
        print("GET /model-info ->", info.status_code, {k: info.json().get(k) for k in ("model_name", "threshold")})
        single = client.post("/predict", json=events[0])
        print("POST /predict ->", single.status_code, json.dumps(single.json()))
        batch = client.post("/predict-batch", json={"events": events})
        body = batch.json()
        print("POST /predict-batch ->", batch.status_code, {k: body.get(k) for k in ("count", "anomaly_count")})
        invalid = client.post("/predict", json={**events[0], "speed": "fast"})
        print("POST /predict (invalid speed) ->", invalid.status_code, invalid.json()["detail"][0]["msg"])

    ok = health.status_code == info.status_code == single.status_code == batch.status_code == 200
    ok = ok and invalid.status_code == 422
    print("Smoke test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
