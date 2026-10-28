"""Smoke-test a deployed instance (local container or Render) using only the Python standard library.

Usage:
    python scripts/deploy_smoke_test.py --url https://<your-service>.onrender.com
    python scripts/deploy_smoke_test.py --url http://localhost:8000 --wait 120

Checks the demo page, /health, /model-info, /docs, /demo/events, /predict, /predict-batch,
request validation (422), /demo/reports and a bundled figure. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def request(base: str, path: str, payload: Any = None, timeout: float = 90.0) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", "User-Agent": "vtad-smoke-test"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def server_ms(headers: dict[str, str]) -> str | None:
    return headers.get("X-Process-Time-Ms") or headers.get("x-process-time-ms")


def wait_for_health(base: str, wait_seconds: float) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status, _, _ = request(base, "/health", timeout=15)
            if status == 200:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"Service at {base} did not become healthy within {wait_seconds:.0f}s")
        time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Base URL, e.g. https://example.onrender.com")
    parser.add_argument("--wait", type=float, default=90, help="Seconds to wait for /health (cold starts)")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    started = time.monotonic()
    wait_for_health(base, args.wait)
    print(f"healthy after {time.monotonic() - started:.1f}s")
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    status, headers, body = request(base, "/")
    check("GET / demo page", status == 200 and b"point model vs window model" in body, f"HTTP {status}")

    status, _, body = request(base, "/health")
    health = json.loads(body)
    check(
        "GET /health",
        status == 200 and health.get("model_loaded") is True and bool(health.get("window_model_name")),
        json.dumps(health),
    )

    status, _, body = request(base, "/model-info")
    info = json.loads(body) if status == 200 else {}
    window_info = info.get("window_model") or {}
    check(
        "GET /model-info",
        status == 200 and 0 < info.get("threshold", 0) < 1 and window_info.get("feature_set") == "window",
        f"point={info.get('model_name')} window={window_info.get('model_name')} runtime={info.get('runtime')}",
    )

    status, _, _ = request(base, "/docs")
    check("GET /docs", status == 200, f"HTTP {status}")

    # Positions 1200-1600 of the demo stream: vehicles have history, gradual faults are possible.
    status, _, body = request(base, "/demo/events?seed=7&start=1200&n=200")
    demo = json.loads(body) if status == 200 else {}
    events = demo.get("events", [])
    check(
        "GET /demo/events",
        status == 200 and len(events) == 200 and "is_anomaly" not in events[0] and demo.get("next_start") == 1400,
        f"{len(events)} events",
    )

    if events:
        truth = {g["event_id"]: g["is_anomaly"] for g in demo.get("ground_truth", [])}
        status, headers, body = request(base, "/predict", events[0])
        single = json.loads(body) if status == 200 else {}
        check(
            "POST /predict",
            status == 200 and 0 <= single.get("anomaly_score", -1) <= 1,
            f"score={single.get('anomaly_score')} server_ms={server_ms(headers)}",
        )

        status, headers, body = request(base, "/predict-batch", {"events": events})
        batch = json.loads(body) if status == 200 else {}
        preds = batch.get("predictions", [])
        agree = sum(p["is_anomaly"] == truth.get(p["event_id"]) for p in preds)
        check(
            "POST /predict-batch",
            status == 200 and batch.get("count") == len(events),
            f"anomalies={batch.get('anomaly_count')} agreement_with_truth={agree}/{len(preds)} "
            f"server_ms={server_ms(headers)}",
        )

        status, _, body = request(base, "/predict-stream", {"events": events[:100]})
        first = json.loads(body) if status == 200 else {}
        status2, headers, body = request(base, "/predict-stream", {"events": events[100:], "state": first.get("state")})
        second = json.loads(body) if status2 == 200 else {}
        stream_preds = first.get("predictions", []) + second.get("predictions", [])
        agree = sum(p["is_anomaly"] == truth.get(p["event_id"]) for p in stream_preds)
        check(
            "POST /predict-stream (state round trip)",
            status == status2 == 200 and len(stream_preds) == len(events) and len(second.get("state", {})) > 0,
            f"anomalies={sum(p['is_anomaly'] for p in stream_preds)} agreement_with_truth={agree}/{len(stream_preds)} "
            f"server_ms={server_ms(headers)}",
        )

        status, _, body = request(base, "/predict", {**events[0], "speed": "fast"})
        check("POST /predict invalid payload -> 422", status == 422, f"HTTP {status}")

    status, _, body = request(base, "/demo/reports")
    reports = json.loads(body) if status == 200 else {}
    check(
        "GET /demo/reports",
        status == 200 and all(reports.get(k) for k in ("model_evaluation", "benchmark", "streaming")),
        ", ".join(f"{k}={'yes' if v else 'missing'}" for k, v in reports.items()),
    )

    for figure in ("confusion_matrices.png", "benchmark_latency.png"):
        status, headers, body = request(base, f"/demo/figures/{figure}")
        check(f"GET /demo/figures/{figure}", status == 200 and body[:8] == b"\x89PNG\r\n\x1a\n", f"HTTP {status}")

    print("Smoke test", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
