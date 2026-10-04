# Thai Fake News Detection — Demo Summary

## Problem Statement
Fake health news spreads fast on Thai social media and misleads public decisions (miracle cures, vaccine myths, overstated herbal claims). This system classifies Thai news text as **Fact** or **Fake** using ML, giving readers a first-pass filter before sharing.

## Dataset
**LimeSoda** (byinth/LimeSoda) — Thai health-news corpus.
- Total docs: 5,763 (train 2,698 / val 300 / test 2,765)
- Binary subset (Fact + Fake only): **3,436 docs** (Fact 1,558 / Fake 1,878); `Undefined` dropped per binary scope
- Stratified split: **80% train / 20% test**, `random_state=343` (3 last digits of the student ID — same seed in `src/*.py` and the notebook)

## System Architecture
```
Raw Thai text (title + detail)
        │
        ▼
PyThaiNLP word_tokenize (engine='newmm')   ← the SAME step at training time and at serving time
        │
        ▼
TF-IDF vectorizer (max_features=10000, ngram 1–2, tokenizer=str.split, token_pattern=None)
        │
        ▼
Classifier model (LogisticRegression / ComplementNB / LinearSVC)
        │
        ▼
FastAPI POST /api/predict  ── CORS ──►  Single-file web UI (paper/ink design)
                                      ├─ preset buttons AND typed text → live API
                                      └─ 3s timeout → offline fallback to a fixture, badged
                                         "[Offline Demo Mode] ผลตัวอย่าง · ไม่ได้วิเคราะห์สด"
                                         + note "โหมดสำรอง — ไม่ได้วิเคราะห์ข่าวจริง"
```

## Model Comparison (same train/test split)
| Model | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |
|---|---|---|---|---|
| ComplementNB | 0.8754 ± 0.0086 | 0.8895 | 0.8884 | 0.33 |
| LogisticRegression (baseline) | 0.8960 ± 0.0084 | 0.9026 | 0.9018 | 0.38 |
| **LinearSVC (calibrated)** ⭐ | **0.9021 ± 0.0081** | **0.9041** | **0.9029** | **0.34** |

⭐ Best model (`model_best.joblib`) served by the API by default. Mapping to the LimeSoda paper's
classical baselines: LinearSVC here is the strongest of the three, and Macro F1 ≈ 0.89 is in the
same band as the paper's classical results; the transformer route (WangchanBERTa) is left as
future work per the CPU-first scope.

Training and serving use **the same PyThaiNLP `newmm` text** (`load_soda()` tokenizes with newmm,
the API tokenizes with newmm), so the number on the slide is the number the live system produces —
verified with 100% prediction agreement and 0 label flips across the 688 test docs. The pre-fix
pipeline (LimeSoda token list + default `token_pattern`) scored 0.8696 macro F1 on the same test
split: the fix is worth **+3.3 points**. `notebooks/learning_evidence.ipynb` (SEED = 343)
reproduces the served model's 0.9041 / 0.9029 exactly.

## Live Demo Examples (deterministic, `demo_examples.json`)
| # | Headline | Verdict | Confidence |
|---|---|---|---|
| 1 | อากาศร้อนอาหารบูดเสียง่ายแนะ!!รับประทานอาหาร“สุก… | Fact | 100.0% |
| 2 | สธ.เตรียมพร้อมรับคนไทย161คนกลับจากอู่ฮั่นพรุ่งนี้… | Fact | 99.9% |
| 3 | 5สูตรสมุนไพร..รักษาเก๊าท์ล้างกรดยูริก!!… | Fake | 99.9% |
| 4 | ใบทุเรียนเทศแห้งเอามาต้มกินมะเร็งหายค่ะตอนนี้ดัง… | Fake | 99.9% |

## Demo behaviour (what the UI does)
- **Preset buttons** (📝 ตัวอย่าง) and **text the presenter types** both go through the live API —
  a preset simply fills the two inputs and submits them. Answering the moringa preset = Fake 0.755
  from `model_best.joblib`, and the meta line under the headline reads `โมเดล model_best.joblib · API … ms`.
- If a request fails (server down / dropped), a 3-second timeout falls back to a fixture chosen
  deterministically from the text, and the UI states it plainly: badge
  `[Offline Demo Mode] ผลตัวอย่าง · ไม่ได้วิเคราะห์สด`, note *"โหมดสำรอง — ผลลัพธ์นี้มาจากข้อมูล
  ตัวอย่าง ไม่ได้วิเคราะห์ข่าวจริง (เซิร์ฟเวอร์ไม่ตอบ)"* and meta `ผลตัวอย่างจากโหมดสำรอง · ไม่ได้วิเคราะห์สด`.
  There is no silent demo path: a real model answer and a fixture answer always look different.

## Run
```bash
# backend (terminal 1) — run from the repo root
python3 -m uvicorn src.main:app --host localhost --port 8000
#   -> Uvicorn running on http://localhost:8000  (คลิกที่ข้อความใน terminal ได้เลย)

# frontend — ไม่ต้องเปิดไฟล์เอง เปิด http://localhost:8000 ในเบราว์เซอร์ (backend เสิร์ฟหน้าเว็บให้)

# CLI — run from src/
python3 predict.py --title "..." --detail "..."
```

## Files
`train_baseline.py` · `benchmark.py` · `pick_demo_examples.py` · `predict.py` · `main.py` · `model_baseline.joblib` · `model_best.joblib` · `evaluation_results.json` · `benchmark_results.md` · `demo_examples.json` · `fake-news-checker-mockup.html`
