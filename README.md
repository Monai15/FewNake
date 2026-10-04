# Thai Fake News Detection (FewNake)

Classify Thai health news as **Fact** or **Fake** using classical ML, served through a FastAPI backend with a single-file web UI. Built as an AI/ML mini-project (240-318).

- **Data:** [LimeSoda](https://github.com/byinth/LimeSoda) Thai health-news corpus (binary subset: 3,436 docs, Fact 1,558 / Fake 1,878)
- **Best model:** Calibrated LinearSVC + TF-IDF — test macro F1 **0.903**, accuracy 0.904, 0.34 ms/sample (seed 343).
  Training and serving feed the model the *same* PyThaiNLP `newmm` text, so this is also the
  number the live demo produces (100% prediction agreement between the two paths).
- **UI:** paper/ink styled single-file HTML (works on projector + mobile)

## Pipeline

```
Raw Thai text (title + detail)
   → PyThaiNLP word_tokenize (newmm)  ← same step at training time and at serving time
   → TF-IDF (max_features=10000, ngram 1–2, tokenizer=str.split — input is already segmented)
   → classifier (ComplementNB / LogisticRegression / LinearSVC)
   → FastAPI POST /api/predict  (CORS *)
   → Web UI: every check — preset buttons and typed text alike — goes live through the API;
     3s timeout → offline fallback to a fixture, badged [Offline Demo Mode] + a written notice
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
py -m pip install --user -r requirements.txt

# download dataset into data/ (~50MB, not tracked in git)
py src/download_data.py
```

## Run

```bash
# 1. Start backend (from repo root)
py -m uvicorn src.main:app --host localhost --port 8000
#    ข้อความที่ขึ้นบนจอจะคลิกได้: Uvicorn running on http://localhost:8000

# 2. Open the web UI — backend เสิร์ฟหน้าเว็บให้แล้ว ไม่ต้องเปิดไฟล์เอง
#    เปิด http://localhost:8000 ในเบราว์เซอร์

# 3. Or use the CLI
py src/predict.py --title "พาดหัวข่าว..." --detail "เนื้อหาข่าว..."
# → {"label": "Fake", "confidence": 0.98, "latency_ms": 8.25}
```

## Reproduce

```bash
# Baseline + evaluation (writes results/, models/) — needs data/ first
py src/train_baseline.py

# Multi-model benchmark (best model → model_best.joblib)
py src/benchmark.py

# Pick deterministic live-demo examples (2 Fact, 2 Fake)
py src/pick_demo_examples.py
```

## Results

| Model                         | 5-fold CV Macro F1  | Test Acc   | Test Macro F1 | Latency (ms/sample) |
| ----------------------------- | ------------------- | ---------- | ------------- | ------------------- |
| ComplementNB                  | 0.8754 ± 0.0086     | 0.8895     | 0.8884        | 0.33                |
| LogisticRegression (baseline) | 0.8960 ± 0.0084     | 0.9026     | 0.9018        | 0.38                |
| **LinearSVC (calibrated)** ⭐ | **0.9021 ± 0.0081** | **0.9041** | **0.9029**    | **0.34**            |

Every number above is produced on the same `newmm` text the API serves, so the reported score
is the live system's score. The train/test split uses `random_state=343` (3 last digits of the
student ID) everywhere, so `notebooks/learning_evidence.ipynb` reproduces these numbers exactly.
For reference, the pre-fix pipeline (trained on LimeSoda's own token list, default
`token_pattern`, then served `newmm` text) measured **0.8696** macro F1 on the identical test
split — the fix is worth **+3.3 macro F1 points** on the served system.

Full details: `results/benchmark_results.md`, `results/evaluation_results.json`.

## Notes

- Binary classification (Fact/Fake only); `Undefined`-labeled docs in LimeSoda are excluded.
- Text is tokenized with PyThaiNLP `newmm` **both** when training and when serving; the model
  never sees a token distribution at training time that differs from the one at inference.
- The TF-IDF vectorizer runs with `tokenizer=str.split, token_pattern=None`: the input is
  already tokenized, and sklearn's default `token_pattern` would re-split on every Thai
  combining mark (`แพทย์` → `แพทย`, `พร้อม` → `พร`+`อม`, `น้ำ` → dropped).
- Dataset packaging tokens (`""` padding and the `|||` title/body separator) are stripped in
  `load_soda()`; `|||` alone appears in 75% of Fact vs 57% of Fake docs, so leaving it in
  gives the model a class-correlated shortcut.
- UI preset buttons **and** typed text both go through the live API: a preset sends its own text
  to the model (the moringa example answers Fake 0.77). Nothing is served from a fixture while
  the server is up. If a request fails, the UI falls back to a fixture and says so out loud —
  `[Offline Demo Mode] ผลตัวอย่าง · ไม่ได้วิเคราะห์สด` plus the note *"โหมดสำรอง — ผลลัพธ์นี้มา
  จากข้อมูลตัวอย่าง ไม่ได้วิเคราะห์ข่าวจริง"*, and the meta line under the headline reads
  `ผลตัวอย่างจากโหมดสำรอง · ไม่ได้วิเคราะห์สด`. Live answers show `โมเดล model_best.joblib · API … ms`.
- Per-request latency shown by the UI is single-request (4–7 ms); the benchmark's
  ms/sample figures measure bulk prediction (0.33–0.38 ms for the three shipped models; 0.33–0.54 ms
  across every model measured in the audit run) and vary ~±0.05 ms between runs.
- Retrain with `train_baseline.py` → `benchmark.py` → `pick_demo_examples.py` in that order;
  pinned env is the `fewnake-310` kernel (Python 3.10 / scikit-learn 1.7.2) used by the
  notebook — pickles warn when loaded by a different scikit-learn version.
- Transformer route (WangchanBERTa) left as future work — CPU-first scope.
- Dataset © [byinth/LimeSoda](https://github.com/byinth/LimeSoda); see its license for data usage.
