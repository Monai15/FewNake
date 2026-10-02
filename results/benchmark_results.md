| Model                         | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |
| ----------------------------- | ------------------ | -------- | ------------- | ------------------- |
| ComplementNB                  | 0.8775 ± 0.0120    | 0.8503   | 0.8479        | 0.42                |
| LinearSVC (calibrated)        | 0.9032 ± 0.0140    | 0.8881   | 0.8868        | 0.31                |
| LogisticRegression (baseline) | 0.8957 ± 0.0155    | 0.8735   | 0.8721        | 0.32                |

**Best model:** LinearSVC (calibrated) (test macro F1 = 0.8868) → saved as `model_best.joblib`
