"""Phase 1: Load LimeSoda -> pre-tokenized text -> TF-IDF + LogisticRegression baseline.

Filter to binary Fact/Fake only. Stratified 80/20 split (random_state=343).
Saves evaluation_results.json and model_baseline.joblib.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from predict import preprocess  # running as `python train_baseline.py`
except ImportError:  # imported as `src.train_baseline` (notebook / other scripts)
    from src.predict import preprocess

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
FILES = [DATA_DIR / "train_v1.jsonl", DATA_DIR / "val_v1.jsonl", DATA_DIR / "test_v1.jsonl"]
LABEL_MAP = {"Fact News": 0, "Fake News": 1}
# Dataset packaging, not language: "" (padding) and "|||" (title/body separator).
# "|||" sits in 75.4% of Fact docs but 56.9% of Fake docs, so leaving it in hands the
# model a class-correlated shortcut that has nothing to do with the text.
ARTIFACTS = {"", "|||"}


def build_vectorizer() -> TfidfVectorizer:
    """TF-IDF over text that is ALREADY tokenized.

    token_pattern must be disabled here. The default r"(?u)\\b\\w\\w+\\b" re-splits the
    input, and Thai combining marks (U+0E31-U+0E4E) are not \\w, so it turns
    "แพทย์" into "แพทย", splits "พร้อม" into "พร"+"อม", and drops "น้ำ" entirely --
    the vocabulary fills up with fragments. tokenizer=str.split keeps the
    segmentation that was handed to us (LimeSoda tokens when training,
    PyThaiNLP newmm tokens when serving).
    """
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
    )


def load_soda(style: str = "serve") -> pd.DataFrame:
    """Load the LimeSoda binary Fact/Fake subset.

    style="serve" (default)
        Feed the text through the same PyThaiNLP newmm step the API uses, so the
        model is trained on exactly the token distribution it gets served.
        Re-joining LimeSoda's token list restores the original Thai text (Thai
        has no inter-word spaces), minus the packaging artifacts.
    style="dataset"
        Keep LimeSoda's own token list as the text -- the pre-fix behaviour.
        Kept so the "train tokens vs serve tokens" comparison stays reproducible
        (notebook part 7); it is NOT what the shipped model uses.
    """
    if style not in ("serve", "dataset"):
        raise ValueError(f"style must be 'serve' or 'dataset', got {style!r}")

    rows = []
    for path in FILES:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                tag = rec.get("Document Tag")
                if tag not in LABEL_MAP:
                    continue  # drop Undefined / non-binary
                toks = [t for t in (rec.get("Text") or []) if t not in ARTIFACTS]
                if style == "dataset":
                    text = " ".join(toks)
                else:
                    text = preprocess("".join(toks))
                rows.append({"text": text, "label": LABEL_MAP[tag]})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    df = load_soda()
    print(f"loaded {len(df)} binary rows (Fact={int((df.label==0).sum())}, Fake={int((df.label==1).sum())})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=343, stratify=df["label"]
    )
    print(f"split -> train {len(X_train)} / test {len(X_test)}")

    pipe = Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    latency = (time.time() - t0) * 1000

    print(classification_report(y_test, y_pred, target_names=["Fact", "Fake"]))
    print(f"confusion matrix:\n{cm}")

    results = {
        "dataset": "LimeSoda (binary Fact/Fake only)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": "PyThaiNLP newmm (train and serve identical) + TfidfVectorizer(max_features=10000, ngram_range=(1,2), tokenizer=str.split, token_pattern=None) + LogisticRegression(max_iter=1000, class_weight='balanced')",
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "train_time_sec": time.time() - t0,
        "predict_latency_ms_per_sample": latency / len(X_test),
    }
    (_ROOT / "results").mkdir(exist_ok=True)
    (_ROOT / "models").mkdir(exist_ok=True)
    with open(_ROOT / "results" / "evaluation_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    joblib.dump(pipe, _ROOT / "models" / "model_baseline.joblib")
    print("wrote results/evaluation_results.json + models/model_baseline.joblib")


if __name__ == "__main__":
    main()
