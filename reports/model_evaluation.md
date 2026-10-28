# Model evaluation (test split)

Data: **synthetic** telemetry from `vehicle_anomaly.data_generator`. Splits are grouped by vehicle; thresholds (max F1), early stopping and model selection use the validation split only. The test split is scored once for this report.

- Train: 140,000 events / 70 vehicles
- Validation: 30,000 events / 15 vehicles
- Test: 30,000 events / 15 vehicles (10.40% anomalies)
- Selection metric (validation): `average_precision`; selected point model `random_forest`, selected window model `random_forest_window`
- Feature sets: **point** = the event's own readings (+3 derived); **window** = point + per-vehicle history features (EWMA deviation and trend of engine temperature and battery voltage, history length)
- `isolation_forest` is unsupervised: it is fitted without labels; only its threshold uses validation labels

| Model | Features | Supervised | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| random_forest (selected) | point | yes | 0.7863 | 0.9139 | 0.8129 | 0.8604 | 0.9573 | 0.8849 |
| xgboost | point | yes | 0.9276 | 0.9500 | 0.7677 | 0.8492 | 0.9489 | 0.8731 |
| random_forest_window (selected) | window | yes | 0.4079 | 0.9537 | 0.9708 | 0.9622 | 0.9985 | 0.9924 |
| xgboost_window | window | yes | 0.7086 | 0.9710 | 0.9436 | 0.9571 | 0.9981 | 0.9897 |
| isolation_forest | window | no | 0.4705 | 0.4381 | 0.4072 | 0.4221 | 0.8478 | 0.4976 |

## Recall by anomaly type (false-positive rate for `normal`)

Drift types are gradual faults relative to the vehicle's own baseline; single readings often stay inside the fleet's normal range.

| Type | Test events | random_forest | xgboost | random_forest_window | xgboost_window | isolation_forest |
|---|---|---|---|---|---|---|
| normal | 26,879 | 0.0089 | 0.0047 | 0.0055 | 0.0033 | 0.0606 |
| overheating | 303 | 0.8878 | 0.8779 | 0.9505 | 0.9274 | 0.2211 |
| rpm_anomaly | 285 | 0.9860 | 0.9368 | 1.0000 | 0.9930 | 0.2772 |
| harsh_acceleration | 286 | 0.9685 | 0.9336 | 1.0000 | 0.9930 | 0.9895 |
| battery_voltage | 287 | 0.9477 | 0.8432 | 0.9895 | 0.9861 | 0.5261 |
| sensor_combination | 297 | 0.9966 | 0.9933 | 1.0000 | 1.0000 | 0.9865 |
| gradual_overheating (drift) | 643 | 0.8367 | 0.8134 | 0.9036 | 0.8196 | 0.0747 |
| battery_degradation (drift) | 1,020 | 0.5922 | 0.5255 | 0.9892 | 0.9706 | 0.3431 |

## ONNX export (edge runtime)

Parity is measured on the validation split: ONNX Runtime (float32) vs the native model (float64).

| Model | joblib size | ONNX size | Max score diff | Decision agreement |
|---|---|---|---|---|
| random_forest | 10,616 KB | 5,129 KB | 8.6e-07 | 99.9967% |
| xgboost | 353 KB | 188 KB | 8.9e-08 | 100.0000% |
| random_forest_window | 9,522 KB | 4,593 KB | 8.5e-07 | 100.0000% |
| xgboost_window | 380 KB | 218 KB | 8.8e-08 | 100.0000% |

## Confusion matrices

- **random_forest**: TN=26,640 FP=239 FN=584 TP=2,537
- **xgboost**: TN=26,753 FP=126 FN=725 TP=2,396
- **random_forest_window**: TN=26,732 FP=147 FN=91 TP=3,030
- **xgboost_window**: TN=26,791 FP=88 FN=176 TP=2,945
- **isolation_forest**: TN=25,249 FP=1,630 FN=1,850 TP=1,271

![Confusion matrices](figures/confusion_matrices.png)
![ROC and PR curves](figures/roc_pr_curves.png)
![Feature importance](figures/feature_importance.png)
