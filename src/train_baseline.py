"""Phase 1: Load LimeSoda -> pre-tokenized text -> TF-IDF + LogisticRegression baseline.

Filter to binary Fact/Fake only. Stratified 80/20 split (random_state=42).
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

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
FILES = [DATA_DIR / "train_v1.jsonl", DATA_DIR / "val_v1.jsonl", DATA_DIR / "test_v1.jsonl"]
LABEL_MAP = {"Fact News": 0, "Fake News": 1}


def load_soda() -> pd.DataFrame:
    rows = []
    for path in FILES:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                tag = rec.get("Document Tag")
                if tag not in LABEL_MAP:
                    continue  # drop Undefined / non-binary
                # pre-tokenization: join the provided token list with spaces
                text = " ".join(rec.get("Text") or [])
                rows.append({"text": text, "label": LABEL_MAP[tag]})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    df = load_soda()
    print(f"loaded {len(df)} binary rows (Fact={int((df.label==0).sum())}, Fake={int((df.label==1).sum())})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"split -> train {len(X_train)} / test {len(X_test)}")

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
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
        "model": "TfidfVectorizer(max_features=10000, ngram_range=(1,2)) + LogisticRegression(max_iter=1000, class_weight='balanced')",
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
