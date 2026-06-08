"""Download raw dataset files into data/raw."""

import json
from pathlib import Path
import sys
from config import TRAIN_PATH, TEST_PATH, VAL_PATH
from datasets import load_dataset

def main():
    if TRAIN_PATH.exists() and TEST_PATH.exists() and VAL_PATH.exists():
        print("Dataset already prepared. Skipping download.")
    else:
        print("Downloading dataset...")
        dataset = load_dataset("SetFit/amazon_reviews_multi_en")

        TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

        with TRAIN_PATH.open("w", encoding="utf-8") as f:
            for x in dataset["train"]:
                f.write(json.dumps(x) + "\n")

        with TEST_PATH.open("w", encoding="utf-8") as f:
            for x in dataset["test"]:
                f.write(json.dumps(x) + "\n")

        with VAL_PATH.open("w", encoding="utf-8") as f:
            for x in dataset["validation"]:
                f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
