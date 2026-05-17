"""Download raw dataset files into data/raw."""

import json
from pathlib import Path
import sys

from datasets import load_dataset

from config import RAW_DATA_DIR


def main():
    dataset = load_dataset("SetFit/amazon_reviews_multi_en")
    output_path_train = RAW_DATA_DIR / "amazon_reviews_multi_en_train.jsonl"
    output_path_test = RAW_DATA_DIR / "amazon_reviews_multi_en_test.jsonl"
    output_path_validation = RAW_DATA_DIR / "amazon_reviews_multi_en_validation.jsonl"
    output_path_train.parent.mkdir(parents=True, exist_ok=True)
    output_path_test.parent.mkdir(parents=True, exist_ok=True)
    output_path_validation.parent.mkdir(parents=True, exist_ok=True)

    with output_path_train.open("w", encoding="utf-8") as file:
        for item in dataset["train"]:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    with output_path_test.open("w", encoding="utf-8") as file:
        for item in dataset["test"]:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    with output_path_validation.open("w", encoding="utf-8") as file:
        for item in dataset["validation"]:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved raw data to {output_path_train}, {output_path_test}, and {output_path_validation}")


if __name__ == "__main__":
    main()
