| Model | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |
|---|---|---|---|---|
| Dummy (majority) | 0.3534 ± 0.0003 | 0.5465 | 0.3534 | 0.24 |
| ComplementNB | 0.8754 ± 0.0086 | 0.8895 | 0.8884 | 0.28 |
| LogisticRegression | 0.8960 ± 0.0084 | 0.9026 | 0.9018 | 0.26 |
| DecisionTree | 0.7919 ± 0.0052 | 0.8125 | 0.8097 | 0.25 |
| RandomForest (200) | 0.8873 ± 0.0070 | 0.8910 | 0.8895 | 0.40 |
| KNN (k=5, cosine) | 0.8579 ± 0.0204 | 0.8474 | 0.8455 | 0.39 |

**Best model:** LogisticRegression (test macro F1 = 0.9018) → saved as `model_best.joblib`
