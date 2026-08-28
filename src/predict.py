"""CLI prediction against the trained baseline model.

Usage:
  python predict.py --text "ข้อความข่าว..."
  python predict.py --title "พาดหัว" --detail "เนื้อหา..."

Output: {label, confidence, latency_ms}
"""
import argparse
import json
import os
import time
from pathlib import Path

import joblib
from pythainlp.tokenize import word_tokenize

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT = _ROOT / "models" / "model_best.joblib"
if not _DEFAULT.exists():
    _DEFAULT = _ROOT / "models" / "model_baseline.joblib"
MODEL_PATH = os.environ.get("MODEL_PATH", str(_DEFAULT))


def preprocess(raw: str) -> str:
    """Tokenize with PyThaiNLP newmm, join tokens with whitespace."""
    return " ".join(word_tokenize(raw, engine="newmm")).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict Fact/Fake for Thai news text")
    ap.add_argument("--text", help="full news text (headline + body)")
    ap.add_argument("--title", help="headline")
    ap.add_argument("--detail", help="body")
    args = ap.parse_args()
    if args.text is None and args.title is None and args.detail is None:
        ap.error("provide --text OR --title/--detail")

    if args.title is not None and args.detail is not None:
        raw = args.title + "\n" + args.detail
    elif args.title is not None:
        raw = args.title
    elif args.detail is not None:
        raw = args.detail
    else:
        raw = args.text

    pipe = joblib.load(MODEL_PATH)
    text = preprocess(raw)

    t0 = time.perf_counter()
    proba = pipe.predict_proba([text])[0]
    label = "Fake" if pipe.predict([text])[0] == 1 else "Fact"
    latency_ms = (time.perf_counter() - t0) * 1000
    confidence = proba.max()

    print(json.dumps({"label": label, "confidence": round(float(confidence), 4), "latency_ms": round(float(latency_ms), 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
