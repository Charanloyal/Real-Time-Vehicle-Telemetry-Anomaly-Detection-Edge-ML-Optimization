"""Inference latency benchmark: reference path vs optimized path.

Reference ("baseline") path, i.e. the straightforward implementation before optimization:
    pandas.DataFrame([event]) -> fitted sklearn Pipeline.transform -> model.predict_proba

Variants measured:
    numpy_features : optimized NumPy feature extraction, unchanged model call (isolates feature
                     extraction cost)
    optimized      : NumPy feature extraction + low-overhead model call (AnomalyPredictor)

All variants reuse the same loaded objects, run on identical events, and are checked for score
parity. Results are written to reports/benchmark_results.{json,md}.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from vehicle_anomaly.artifacts import library_versions, load_metadata, utc_now_iso
from vehicle_anomaly.config import AppConfig, load_config
from vehicle_anomaly.logging_utils import configure_logging, get_logger
from vehicle_anomaly.predict import AnomalyPredictor
from vehicle_anomaly.reporting import markdown_table, plot_benchmark
from vehicle_anomaly.schema import REQUIRED_FIELDS
from vehicle_anomaly.train import TEST_SET_FILE

logger = get_logger(__name__)
MAX_BASELINE_LOOP_ITERATIONS = 3


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def total_memory_gb() -> float | None:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / 1024**3, 1)
        return None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024**2, 1)
    return None


def environment_details() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "total_memory_gb": total_memory_gb(),
        "python_implementation": platform.python_implementation(),
        "libraries": library_versions(),
        "in_container": Path("/.dockerenv").exists(),
    }


def latency_stats(durations_ns: np.ndarray, items_per_call: int = 1) -> dict[str, Any]:
    ms = durations_ns / 1e6
    mean_ms = float(ms.mean())
    return {
        "calls": int(len(ms)),
        "items_per_call": items_per_call,
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "throughput_per_s": float(items_per_call * 1000.0 / mean_ms) if mean_ms > 0 else None,
    }


def _time_calls(fn: Callable[[Any], Any], inputs: Sequence[Any], iterations: int, warmup: int) -> np.ndarray:
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    durations = np.empty(iterations, dtype=np.int64)
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for i in range(iterations):
            item = inputs[i % len(inputs)]
            start = time.perf_counter_ns()
            fn(item)
            durations[i] = time.perf_counter_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return durations


def improvement_pct(baseline: float, optimized: float) -> float:
    return (baseline - optimized) / baseline * 100.0 if baseline > 0 else 0.0


def load_benchmark_events(config: AppConfig) -> list[dict[str, Any]]:
    path = config.paths.processed_dir / TEST_SET_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Test split not found at {path}. Train a model first.")
    return pd.read_parquet(path, columns=list(REQUIRED_FIELDS)).to_dict("records")


def benchmark_model(
    predictor: AnomalyPredictor, events: list[dict[str, Any]], config: AppConfig
) -> dict[str, Any]:
    bench = config.benchmark
    preprocessor, model = predictor.preprocessor, predictor.model

    def baseline_one(event: Mapping[str, Any]) -> float:
        features = preprocessor.transform(pd.DataFrame([event]))
        return float(model.predict_proba(features)[:, 1][0])

    def numpy_features_one(event: Mapping[str, Any]) -> float:
        return float(model.predict_proba(predictor.features_from_records([event]))[:, 1][0])

    def optimized_one(event: Mapping[str, Any]) -> float:
        return predictor.predict_one(event).anomaly_score

    parity_events = events[: min(500, len(events))]
    baseline_scores = np.array([baseline_one(e) for e in parity_events])
    optimized_scores = np.array([optimized_one(e) for e in parity_events])
    batch_scores = np.array([p.anomaly_score for p in predictor.predict_records(parity_events)])
    parity = {
        "events_checked": len(parity_events),
        "max_abs_score_diff_single": float(np.max(np.abs(baseline_scores - optimized_scores))),
        "max_abs_score_diff_batch": float(np.max(np.abs(baseline_scores - batch_scores))),
        "prediction_agreement": float(
            np.mean((baseline_scores >= predictor.threshold) == (optimized_scores >= predictor.threshold))
        ),
    }

    single: dict[str, Any] = {}
    for name, fn in (("baseline", baseline_one), ("numpy_features", numpy_features_one), ("optimized", optimized_one)):
        durations = _time_calls(fn, events, bench.single_iterations, bench.warmup_iterations)
        single[name] = latency_stats(durations)
        logger.info("Single-event benchmark", extra={"model": predictor.model_name, "variant": name,
                                                     "p50_ms": round(single[name]["p50_ms"], 4)})

    batch_size = min(bench.batch_size, len(events))
    batches = [events[i : i + batch_size] for i in range(0, len(events) - batch_size + 1, batch_size)] or [events]
    loop_iterations = min(bench.batch_iterations, MAX_BASELINE_LOOP_ITERATIONS)
    batch_variants: list[tuple[str, Callable[[Any], Any], int]] = [
        ("baseline_per_event_loop", lambda b: [baseline_one(e) for e in b], loop_iterations),
        ("baseline_vectorized", lambda b: model.predict_proba(preprocessor.transform(pd.DataFrame(b)))[:, 1],
         bench.batch_iterations),
        ("optimized_batch", predictor.predict_records, bench.batch_iterations),
    ]
    batch: dict[str, Any] = {}
    for name, fn, iterations in batch_variants:
        durations = _time_calls(fn, batches, iterations, warmup=1)
        batch[name] = latency_stats(durations, items_per_call=batch_size)
        logger.info("Batch benchmark", extra={"model": predictor.model_name, "variant": name,
                                              "throughput_per_s": round(batch[name]["throughput_per_s"], 1)})

    return {
        "model_name": predictor.model_name,
        "model_type": type(model).__name__,
        "parity": parity,
        "single_event": single,
        "batch": batch,
        "improvement": {
            "single_mean_latency_pct": improvement_pct(single["baseline"]["mean_ms"], single["optimized"]["mean_ms"]),
            "single_p50_latency_pct": improvement_pct(single["baseline"]["p50_ms"], single["optimized"]["p50_ms"]),
            "feature_extraction_only_mean_latency_pct": improvement_pct(
                single["baseline"]["mean_ms"], single["numpy_features"]["mean_ms"]
            ),
            "batch_speedup_vs_per_event_loop": batch["optimized_batch"]["throughput_per_s"]
            / batch["baseline_per_event_loop"]["throughput_per_s"],
            "batch_speedup_vs_baseline_vectorized": batch["optimized_batch"]["throughput_per_s"]
            / batch["baseline_vectorized"]["throughput_per_s"],
        },
        "batch_size": batch_size,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    cfg = report["settings"]
    lines = [
        "# Inference benchmark results",
        "",
        f"Measured at {report['measured_at']} on the machine below. Numbers are wall-clock measurements "
        "from `time.perf_counter_ns` in a single Python process; other workloads on the machine were not "
        "controlled, so expect run-to-run variation.",
        "",
        markdown_table(["Property", "Value"], [
            ["OS", env["os"]], ["CPU", env["cpu"]], ["Logical CPUs", env["logical_cpus"]],
            ["RAM (GB)", env["total_memory_gb"]], ["Python", env["libraries"]["python"]],
            ["scikit-learn", env["libraries"]["scikit-learn"]], ["xgboost", env["libraries"]["xgboost"]],
            ["numpy / pandas", f"{env['libraries']['numpy']} / {env['libraries']['pandas']}"],
            ["In container", env["in_container"]],
        ]),
        "",
        f"Settings: {cfg['single_iterations']:,} timed single-event calls per variant after "
        f"{cfg['warmup_iterations']} warm-up calls; batch size {cfg['batch_size']:,} with "
        f"{cfg['batch_iterations']} timed batches ({MAX_BASELINE_LOOP_ITERATIONS} for the per-event loop). "
        f"Events come from the held-out test split ({cfg['event_pool']:,} events).",
        "",
    ]
    for result in report["models"]:
        s, b, imp, par = result["single_event"], result["batch"], result["improvement"], result["parity"]
        lines += [
            f"## {result['model_name']} ({result['model_type']})",
            "",
            "Single-event latency:",
            "",
            markdown_table(
                ["Variant", "Mean ms", "p50 ms", "p95 ms", "p99 ms", "Predictions/s"],
                [[n, v["mean_ms"], v["p50_ms"], v["p95_ms"], v["p99_ms"], v["throughput_per_s"]] for n, v in s.items()],
            ),
            "",
            f"- Mean latency improvement (baseline -> optimized): **{imp['single_mean_latency_pct']:.1f}%**",
            f"- p50 latency improvement: **{imp['single_p50_latency_pct']:.1f}%**",
            f"- Feature extraction change alone (baseline -> numpy_features): "
            f"**{imp['feature_extraction_only_mean_latency_pct']:.1f}%** mean latency",
            "",
            f"Batch scoring ({result['batch_size']:,} events per call):",
            "",
            markdown_table(
                ["Variant", "Mean ms / batch", "Predictions/s", "Timed batches"],
                [[n, v["mean_ms"], v["throughput_per_s"], v["calls"]] for n, v in b.items()],
            ),
            "",
            f"- Optimized batch vs per-event baseline loop: **{imp['batch_speedup_vs_per_event_loop']:.1f}x** throughput",
            f"- Optimized batch vs vectorized baseline: **{imp['batch_speedup_vs_baseline_vectorized']:.2f}x** throughput",
            f"- Score parity on {par['events_checked']} events: max |diff| = {par['max_abs_score_diff_single']:.2e} "
            f"(single), {par['max_abs_score_diff_batch']:.2e} (batch); decision agreement "
            f"{par['prediction_agreement']:.2%}",
            "",
        ]
    lines.append("![Benchmark](figures/benchmark_latency.png)")
    return "\n".join(lines) + "\n"


def run_benchmark(config: AppConfig, model_names: Sequence[str] | None = None) -> dict[str, Any]:
    metadata = load_metadata(config.paths.model_dir)
    names = list(model_names or ([config.benchmark.model] if config.benchmark.model else metadata["models"]))
    events = load_benchmark_events(config)
    results = []
    for name in names:
        predictor = AnomalyPredictor.load(config.paths.model_dir, name)
        results.append(benchmark_model(predictor, events, config))

    report = {
        "measured_at": utc_now_iso(),
        "environment": environment_details(),
        "settings": {
            "single_iterations": config.benchmark.single_iterations,
            "warmup_iterations": config.benchmark.warmup_iterations,
            "batch_size": config.benchmark.batch_size,
            "batch_iterations": config.benchmark.batch_iterations,
            "event_pool": len(events),
        },
        "models": results,
    }
    reports = config.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "benchmark_results.md").write_text(_render_markdown(report), encoding="utf-8")
    selected = results[0]
    plot_benchmark(
        {f"{r['model_name']}:{k}": v for r in results for k, v in r["single_event"].items()},
        {f"{r['model_name']}:{k}": v for r in results for k, v in r["batch"].items()},
        config.paths.figures_dir / "benchmark_latency.png",
    )
    logger.info(
        "Benchmark complete",
        extra={
            "report": str(reports / "benchmark_results.md"),
            "model": selected["model_name"],
            "single_mean_improvement_pct": round(selected["improvement"]["single_mean_latency_pct"], 2),
        },
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs optimized inference latency.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", action="append", help="Model name(s) to benchmark (default: all trained)")
    parser.add_argument("--iterations", type=int, default=None, help="Timed single-event calls per variant")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-iterations", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config.logging.level, config.logging.json)
    overrides = {
        key: value
        for key, value in {
            "single_iterations": args.iterations,
            "batch_size": args.batch_size,
            "batch_iterations": args.batch_iterations,
        }.items()
        if value is not None
    }
    if overrides:
        config = replace(config, benchmark=replace(config.benchmark, **overrides))
    report = run_benchmark(config, args.model)
    for result in report["models"]:
        imp = result["improvement"]
        print(
            f"{result['model_name']}: baseline mean {result['single_event']['baseline']['mean_ms']:.3f} ms -> "
            f"optimized {result['single_event']['optimized']['mean_ms']:.3f} ms "
            f"({imp['single_mean_latency_pct']:.1f}% lower); batch throughput "
            f"{result['batch']['optimized_batch']['throughput_per_s']:,.0f} predictions/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
