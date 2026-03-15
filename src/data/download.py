from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import DatasetDict, load_dataset


RAW_DIR = Path("data/raw")
TRAIN_SPLIT_PATH = RAW_DIR / "train.jsonl"
EVAL_SPLIT_PATH = RAW_DIR / "eval.jsonl"
FULL_TRAIN_PATH = RAW_DIR / "pubmed_qa_train.jsonl"
OFFICIAL_EVAL_PATH = RAW_DIR / "pubmed_qa_eval_official.jsonl"
DEFAULT_SPLIT_SEED = 42


def _write_split(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_local_train_eval_split(
    records: list[dict],
    *,
    train_ratio: float = 0.8,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[int, int]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    train_records = shuffled[:split_idx]
    eval_records = shuffled[split_idx:]
    _write_split(train_records, TRAIN_SPLIT_PATH)
    _write_split(eval_records, EVAL_SPLIT_PATH)
    return len(train_records), len(eval_records)


def download_pubmed_qa() -> DatasetDict:
    dataset = load_dataset("pubmed_qa", "pqa_labeled")

    train_records = list(dataset["train"])
    _write_split(train_records, FULL_TRAIN_PATH)

    if "eval" in dataset:
        _write_split(list(dataset["eval"]), OFFICIAL_EVAL_PATH)

    train_count, eval_count = write_local_train_eval_split(train_records)
    print(
        "Saved deterministic local split from pubmed_qa_train.jsonl: "
        f"train={train_count}, eval={eval_count}"
    )

    return dataset


def main() -> None:
    dataset = download_pubmed_qa()
    for split_name, split_ds in dataset.items():
        print(f"Downloaded {split_name}: {len(split_ds)} rows")


if __name__ == "__main__":
    main()
