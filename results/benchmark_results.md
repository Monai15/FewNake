| Model | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |
|---|---|---|---|---|
| ComplementNB | 0.8809 ± 0.0083 | 0.8576 | 0.8557 | 0.29 |
| LinearSVC (calibrated) | 0.9026 ± 0.0111 | 0.8953 | 0.8941 | 0.31 |
| LogisticRegression (baseline) | 0.8987 ± 0.0084 | 0.8823 | 0.8810 | 0.29 |

**Best model:** LinearSVC (calibrated) (test macro F1 = 0.8941) → saved as `model_best.joblib`
