# Thai Fake News Detection (FewNake)

Classify Thai health news as **Fact** or **Fake** using classical ML, served through a FastAPI backend with a single-file web UI. Built as an AI/ML mini-project (240-318).

- **Data:** [LimeSoda](https://github.com/byinth/LimeSoda) Thai health-news corpus (binary subset: 3,436 docs, Fact 1,558 / Fake 1,878)
- **Best model:** Calibrated LinearSVC + TF-IDF — test macro F1 **0.887**, accuracy 0.888, 0.31 ms/sample
- **UI:** paper/ink styled single-file HTML (works on projector + mobile)

## Pipeline

```
Raw Thai text (title + detail)
   → PyThaiNLP word_tokenize (newmm, pre-tokenized)
   → TF-IDF (max_features=10000, ngram 1–2)
   → classifier (ComplementNB / LogisticRegression / LinearSVC)
   → FastAPI POST /api/predict  (CORS *)
   → Web UI, 3s timeout → offline fallback with [Offline Demo Mode] badge
```

## Layout

```
FewNake/
├── src/                 # code (FastAPI, CLI, training, benchmark, demo picker, data download)
├── data/                # LimeSoda jsonl — NOT in repo, run src/download_data.py
├── models/              # model_baseline.joblib, model_best.joblib
├── results/             # evaluation_results.json, benchmark_results.md, demo_examples.json
├── frontend/            # fake-news-checker-mockup.html
└── docs/                # demo_summary.md
```

## Setup

```bash
python3 -m pip install --user -r requirements.txt

# download dataset into data/ (~50MB, not tracked in git)
python3 src/download_data.py
```

## Run

```bash
# 1. Start backend (from repo root)
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# 2. Open the web UI in a browser
open frontend/fake-news-checker-mockup.html

# 3. Or use the CLI
python3 src/predict.py --title "พาดหัวข่าว..." --detail "เนื้อหาข่าว..."
# → {"label": "Fake", "confidence": 0.98, "latency_ms": 8.25}
```

## Reproduce

```bash
# Baseline + evaluation (writes results/, models/) — needs data/ first
python3 src/train_baseline.py

# Multi-model benchmark (best model → model_best.joblib)
python3 src/benchmark.py

# Pick deterministic live-demo examples (2 Fact, 2 Fake)
python3 src/pick_demo_examples.py
```

## Results

| Model | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |
|---|---|---|---|---|
| ComplementNB | 0.8775 ± 0.0120 | 0.8503 | 0.8479 | 0.42 |
| LogisticRegression (baseline) | 0.8957 ± 0.0155 | 0.8735 | 0.8721 | 0.32 |
| **LinearSVC (calibrated)** ⭐ | **0.9032 ± 0.0140** | **0.8881** | **0.8868** | **0.31** |

Full details: `results/benchmark_results.md`, `results/evaluation_results.json`.

## Notes

- Binary classification (Fact/Fake only); `Undefined`-labeled docs in LimeSoda are excluded.
- Transformer route (WangchanBERTa) left as future work — CPU-first scope.
- Dataset © [byinth/LimeSoda](https://github.com/byinth/LimeSoda); see its license for data usage.
