"""Phase 3: FastAPI inference service for Thai fake-news detection.

Endpoints:
  POST /api/predict  {title?, detail?, text?} -> {label, confidence, latency_ms, model}
  GET  /health
"""
import os
import time
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from predict import preprocess
except ImportError:  # running as `uvicorn src.main:app` from repo root
    from src.predict import preprocess

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT = _ROOT / "models" / "model_best.joblib"
if not _DEFAULT.exists():
    _DEFAULT = _ROOT / "models" / "model_baseline.joblib"
MODEL_PATH = os.environ.get("MODEL_PATH", str(_DEFAULT))

app = FastAPI(title="Thai Fake News Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipe = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    title: Optional[str] = None
    detail: Optional[str] = None
    text: Optional[str] = None


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float
    model: str


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if req.text:
        raw = req.text
    elif req.title or req.detail:
        raw = (req.title or "") + "\n" + (req.detail or "")
    else:
        raise ValueError("provide text or title/detail")

    text = preprocess(raw)
    t0 = time.perf_counter()
    proba = _pipe.predict_proba([text])[0]
    label = "Fake" if _pipe.predict([text])[0] == 1 else "Fact"
    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        label=label,
        confidence=round(float(proba.max()), 4),
        latency_ms=round(float(latency_ms), 2),
        model=os.path.basename(MODEL_PATH),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}
