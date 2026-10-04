| Model | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |
|---|---|---|---|---|
| ComplementNB | 0.8754 ± 0.0086 | 0.8895 | 0.8884 | 0.33 |
| LinearSVC (calibrated) | 0.9021 ± 0.0081 | 0.9041 | 0.9029 | 0.34 |
| LogisticRegression (baseline) | 0.8960 ± 0.0084 | 0.9026 | 0.9018 | 0.38 |

**Best model:** LinearSVC (calibrated) (test macro F1 = 0.9029) → saved as `model_best.joblib`
