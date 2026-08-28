"""Phase 5: pick 4 deterministic live-demo examples (2 Fact, 2 Fake).

Reuses the exact train/test split (random_state=42). Chooses correctly-predicted
test docs with highest confidence from each class, titles truncated from the text.
Writes demo_examples.json.
"""
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from pythainlp.tokenize import word_tokenize

from train_baseline import load_soda

_ROOT = Path(__file__).resolve().parent.parent
N_PER_CLASS = 2
TITLE_WORDS = 12


def main() -> None:
    df = load_soda()
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    pipe = joblib.load(_ROOT / "models" / "model_best.joblib")
    proba = pipe.predict_proba(X_test)

    picks = {}
    for idx, text in enumerate(X_test):
        true_label = int(y_test.iloc[idx])
        pred = int(pipe.predict([text])[0])
        conf = float(proba[idx].max())
        if pred != true_label or conf < 0.9:
            continue
        picks.setdefault(true_label, []).append((conf, idx, text))

    out = []
    for label, key in ((0, "Fact"), (1, "Fake")):
        for conf, idx, text in sorted(picks[label], reverse=True)[:N_PER_CLASS]:
            words = word_tokenize(text, engine="newmm")
            title = "".join(words[:TITLE_WORDS]) + ("…" if len(words) > TITLE_WORDS else "")
            out.append(
                {
                    "label": key,
                    "title": title,
                    "detail": text,
                    "confidence": round(conf, 4),
                    "test_index": int(idx),
                }
            )

    (_ROOT / "results").mkdir(exist_ok=True)
    with open(_ROOT / "results" / "demo_examples.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    for r in out:
        print(f"[{r['label']}] {r['confidence']:.2f} :: {r['title'][:60]}")


if __name__ == "__main__":
    main()
