"""Phase 4: Multi-model benchmark on the same train/test split.

Models: ComplementNB, Calibrated LinearSVC, LogisticRegression (baseline).
Outputs: benchmark_results.md, evaluation_results.json update, model_best.joblib.
"""
import json
import time
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from train_baseline import load_soda

MODELS = {
    "ComplementNB": ComplementNB(),
    "LinearSVC (calibrated)": CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3),
    "LogisticRegression (baseline)": LogisticRegression(max_iter=1000, class_weight="balanced"),
}


def build_pipe(clf) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ("clf", clf),
        ]
    )


def main() -> None:
    df = load_soda()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    best = None
    best_f1 = -1.0

    for name, clf in MODELS.items():
        pipe = build_pipe(clf)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_s = time.time() - t0

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro")
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

        t1 = time.time()
        pipe.predict(X_test)
        latency_ms = (time.time() - t1) / len(X_test) * 1000

        rows.append(
            {
                "model": name,
                "cv_f1_macro_mean": float(cv_scores.mean()),
                "cv_f1_macro_std": float(cv_scores.std()),
                "test_accuracy": acc,
                "test_precision_macro": prec,
                "test_recall_macro": rec,
                "test_f1_macro": f1,
                "latency_ms_per_sample": latency_ms,
                "train_time_sec": fit_s,
            }
        )
        print(f"{name}: cv_f1={cv_scores.mean():.4f} acc={acc:.4f} f1={f1:.4f} lat={latency_ms:.2f}ms")
        if f1 > best_f1:
            best_f1 = f1
            best = (name, pipe)

    _root = Path(__file__).resolve().parent.parent
    (_root / "results").mkdir(exist_ok=True)
    (_root / "models").mkdir(exist_ok=True)
    # update evaluation_results.json
    with open(_root / "results" / "evaluation_results.json", encoding="utf-8") as fh:
        results = json.load(fh)
    results["models"] = rows
    results["best_model"] = best[0]
    with open(_root / "results" / "evaluation_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    joblib.dump(best[1], _root / "models" / "model_best.joblib")

    lines = [
        "| Model | 5-fold CV Macro F1 | Test Acc | Test Macro F1 | Latency (ms/sample) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['cv_f1_macro_mean']:.4f} ± {r['cv_f1_macro_std']:.4f} "
            f"| {r['test_accuracy']:.4f} | {r['test_f1_macro']:.4f} | {r['latency_ms_per_sample']:.2f} |"
        )
    lines.append(f"\n**Best model:** {best[0]} (test macro F1 = {best_f1:.4f}) → saved as `model_best.joblib`")
    md = "\n".join(lines) + "\n"
    with open(_root / "results" / "benchmark_results.md", "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
