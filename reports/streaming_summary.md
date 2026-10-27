# Streaming run summary

Generated 2026-10-27T11:45:58+00:00. Engine(s): local. Input: synthetic JSON Lines files.

| Metric | Value |
|---|---|
| Micro-batches | 10 |
| Input events | 100,500 |
| Valid events | 99,500 |
| Invalid events (quarantined) | 1,000 |
| Duplicates dropped | 500 |
| Events scored | 99,000 |
| Anomalies detected | 5,006 |
| Batch processing time, all batches (s) | 1.4660 |
| Rows/s inside batch processing | 68,572.6 |
| Batches processed by the last run | 10 |
| Wall-clock time of the last run (s) | 1.5500 |

Totals cover every committed micro-batch in the output directory; a re-run that resumes from the checkpoint only adds new batches.

## Invalid events by category

| Category | Events |
|---|---|
| missing_field | 335 |
| malformed_json | 183 |
| invalid_timestamp | 171 |
| out_of_range | 157 |
| non_numeric | 154 |

## Detection vs. synthetic ground truth

Matched 99,000 scored events to generator labels: precision 0.9742, recall 0.9704 (TP=4,877, FP=129, FN=149).

## Per-batch metrics

| Batch | Engine | Input | Valid | Invalid | Dupes | Anomalies | Seconds |
|---|---|---|---|---|---|---|---|
| 0 | local | 10,048 | 9,940 | 108 | 48 | 472 | 0.1656 |
| 1 | local | 10,057 | 9,965 | 92 | 57 | 477 | 0.1471 |
| 2 | local | 10,051 | 9,957 | 94 | 51 | 496 | 0.1344 |
| 3 | local | 10,052 | 9,951 | 101 | 52 | 473 | 0.1334 |
| 4 | local | 10,054 | 9,956 | 98 | 54 | 509 | 0.1329 |
| 5 | local | 10,042 | 9,949 | 93 | 42 | 491 | 0.1344 |
| 6 | local | 10,048 | 9,953 | 95 | 48 | 522 | 0.1377 |
| 7 | local | 10,050 | 9,956 | 94 | 50 | 559 | 0.1371 |
| 8 | local | 10,051 | 9,930 | 121 | 51 | 494 | 0.1580 |
| 9 | local | 10,047 | 9,943 | 104 | 47 | 513 | 0.1850 |

![Streaming batches](figures/streaming_batches.png)
