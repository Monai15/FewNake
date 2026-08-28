# Thai Fake News Detection — Demo Summary

## Problem Statement
Fake health news spreads fast on Thai social media and misleads public decisions (miracle cures, vaccine myths, overstated herbal claims). This system classifies Thai news text as **Fact** or **Fake** using ML, giving readers a first-pass filter before sharing.

## Dataset
**LimeSoda** (byinth/LimeSoda) — Thai health-news corpus.
- Total docs: 5,763 (train 2,698 / val 300 / test 2,765)
- Binary subset (Fact + Fake only): **3,436 docs** (Fact 1,558 / Fake 1,878); `Undefined` dropped per binary scope
- Stratified split: **80% train / 20% test**, `random_state=42`

## System Architecture
```
Raw Thai text (title + detail)
        │
        ▼
PyThaiNLP word_tokenize (engine='newmm')   ← pre-tokenize to space-separated
        │
        ▼
TF-IDF vectorizer (max_features=10000, ngram 1–2)
        │
        ▼
Classifier model (LogisticRegression / ComplementNB / LinearSVC)
        │
        ▼
FastAPI POST /api/predict  ── CORS ──►  Single-file web UI (paper/ink design)
                                      └─ 3s timeout → offline fallback to canned fixtures
                                         + "[Offline Demo Mode]" badge (fail-safe demo)
```

## Model Comparison (same train/test split)
| Model | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |
|---|---|---|---|---|
| ComplementNB | 0.8775 ± 0.0120 | 0.8503 | 0.8479 | 0.42 |
| LogisticRegression (baseline) | 0.8957 ± 0.0155 | 0.8735 | 0.8721 | 0.32 |
| **LinearSVC (calibrated)** ⭐ | **0.9032 ± 0.0140** | **0.8881** | **0.8868** | **0.31** |

⭐ Best model (`model_best.joblib`) served by the API by default. Macro F1 ≈ 0.89 — on par with classical-baseline results in the LimeSoda paper domain; transformer (WangchanBERTa) left as future work per CPU-first scope.

## Live Demo Examples (deterministic, `demo_examples.json`)
| # | Headline | Verdict | Confidence |
|---|---|---|---|
| 1 | นายกฯ วอนประชาชนร่วมมือสู้โควิด-19 ย้ำประเทศไทยต้องชนะ | Fact | 100.0% |
| 2 | ไทยพบผู้ติดเชื้อโควิดเพิ่มอีก 3 ราย เตรียมลงโทษผู้ปกปิดประวัติการเดินทาง | Fact | 99.9% |
| 3 | ใบทุเรียนเทศต้มกิน มะเร็งหายได้จริง หมอไม่อยากให้ใครรู้! | Fake | 99.8% |
| 4 | สารสกัดใบทุเรียนเทศยับยั้งเซลล์มะเร็งตับ กรมวิทย์เผยผลวิจัย | Fake | 99.8% |

## Fail-Safe Demo
Web UI calls the FastAPI backend; if the server drops, a 3-second timeout falls back to canned fixture results with a visible `[Offline Demo Mode]` badge — the demo path never breaks.

## Run
```bash
# backend (terminal 1)
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# frontend (terminal 2) — open in browser
open fake-news-checker-mockup.html

# CLI
python3 predict.py --title "..." --detail "..."
```

## Files
`train_baseline.py` · `benchmark.py` · `pick_demo_examples.py` · `predict.py` · `main.py` · `model_baseline.joblib` · `model_best.joblib` · `evaluation_results.json` · `benchmark_results.md` · `demo_examples.json` · `fake-news-checker-mockup.html`
