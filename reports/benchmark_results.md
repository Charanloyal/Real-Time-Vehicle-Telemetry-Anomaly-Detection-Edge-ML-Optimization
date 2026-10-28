# Inference benchmark results

Measured at 2026-10-28T05:30:27+00:00 on the machine below. Numbers are wall-clock measurements from `time.perf_counter_ns` in a single Python process; other workloads on the machine were not controlled, so expect run-to-run variation.

| Property | Value |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| CPU | Intel(R) Core(TM) Ultra 5 235 |
| Logical CPUs | 14 |
| RAM (GB) | 63.70 |
| Python | 3.14.7 |
| scikit-learn | 1.7.2 |
| xgboost | 3.4.1 |
| onnxruntime | 1.31.1 |
| numpy / pandas | 2.5.4 / 2.3.3 |
| In container | False |

Settings: 2,000 timed single-event calls per variant after 100 warm-up calls; batch size 1,000 with 20 timed batches (3 for the per-event loop). ONNX Runtime uses 1 intra-op thread. Load times are the median of 5 cold loads. Events come from the held-out test split (30,000 events).

## random_forest (RandomForestClassifier)

Single-event latency:

| Variant | Mean ms | p50 ms | p95 ms | p99 ms | Predictions/s |
|---|---|---|---|---|---|
| baseline | 35.22 | 36.15 | 42.30 | 47.29 | 28.40 |
| numpy_features | 30.33 | 31.42 | 36.29 | 38.78 | 32.97 |
| optimized | 3.8471 | 3.7433 | 6.3031 | 8.3680 | 259.93 |
| onnx_runtime | 0.0527 | 0.0416 | 0.1131 | 0.1987 | 18,964.4 |

- Mean latency improvement (baseline -> optimized): **89.1%**
- p50 latency improvement: **89.6%**
- Feature extraction change alone (baseline -> numpy_features): **13.9%** mean latency
- ONNX Runtime vs baseline: **99.9%** lower mean latency; vs optimized native: **98.6%** lower

Batch scoring (1,000 events per call):

| Variant | Mean ms / batch | Predictions/s | Timed batches |
|---|---|---|---|
| baseline_per_event_loop | 36,025.2 | 27.76 | 3 |
| baseline_vectorized | 37.00 | 27,029.3 | 20 |
| optimized_batch | 35.62 | 28,077.0 | 20 |
| onnx_runtime_batch | 7.7051 | 129,784.3 | 20 |

- Optimized batch vs per-event baseline loop: **1011.5x** throughput
- Optimized batch vs vectorized baseline: **1.04x** throughput
- ONNX Runtime batch vs optimized native batch: **4.62x** throughput
- Native score parity on 500 events: max |diff| = 2.22e-16 (single), 3.33e-16 (batch); decision agreement 100.00%
- ONNX parity on 30,000 events: max |diff| = 8.47e-07; decision agreement 100.0000%

Edge metrics:

| Artifact | Size (KB) | Cold load ms (median) |
|---|---|---|
| joblib (native) | 10,615.9 | 37.06 |
| ONNX (onnxruntime) | 5,128.9 | 364.48 |

## xgboost (XGBClassifier)

Single-event latency:

| Variant | Mean ms | p50 ms | p95 ms | p99 ms | Predictions/s |
|---|---|---|---|---|---|
| baseline | 2.9907 | 2.8757 | 4.0192 | 5.2110 | 334.38 |
| numpy_features | 0.4016 | 0.3136 | 0.7786 | 1.9373 | 2,489.9 |
| optimized | 0.2669 | 0.1997 | 0.4927 | 1.7096 | 3,746.2 |
| onnx_runtime | 0.0268 | 0.0271 | 0.0366 | 0.0674 | 37,366.8 |

- Mean latency improvement (baseline -> optimized): **91.1%**
- p50 latency improvement: **93.1%**
- Feature extraction change alone (baseline -> numpy_features): **86.6%** mean latency
- ONNX Runtime vs baseline: **99.1%** lower mean latency; vs optimized native: **90.0%** lower

Batch scoring (1,000 events per call):

| Variant | Mean ms / batch | Predictions/s | Timed batches |
|---|---|---|---|
| baseline_per_event_loop | 2,982.8 | 335.26 | 3 |
| baseline_vectorized | 4.0863 | 244,722.3 | 20 |
| optimized_batch | 1.6617 | 601,789.7 | 20 |
| onnx_runtime_batch | 2.2797 | 438,661.9 | 20 |

- Optimized batch vs per-event baseline loop: **1795.0x** throughput
- Optimized batch vs vectorized baseline: **2.46x** throughput
- ONNX Runtime batch vs optimized native batch: **0.73x** throughput
- Native score parity on 500 events: max |diff| = 0.00e+00 (single), 0.00e+00 (batch); decision agreement 100.00%
- ONNX parity on 30,000 events: max |diff| = 8.94e-08; decision agreement 100.0000%

Edge metrics:

| Artifact | Size (KB) | Cold load ms (median) |
|---|---|---|
| joblib (native) | 353.24 | 2.8934 |
| ONNX (onnxruntime) | 188.30 | 3.7682 |

![Benchmark](figures/benchmark_latency.png)
