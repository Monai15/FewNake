"""Download LimeSoda dataset into data/ (train/val/test jsonl).

Source: https://github.com/byinth/LimeSoda
Usage:  python3 src/download_data.py
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/byinth/LimeSoda/master/dataset_train_wo_tokentags_v1"
FILES = ["train_v1.jsonl", "val_v1.jsonl", "test_v1.jsonl"]
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        dest = DATA_DIR / name
        if dest.exists():
            print(f"skip  {name} (exists)")
            continue
        url = f"{BASE}/{name}"
        print(f"fetch {name} ...")
        urllib.request.urlretrieve(url, dest)
    print("done")


if __name__ == "__main__":
    main()
