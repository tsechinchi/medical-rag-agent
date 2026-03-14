from __future__ import annotations

import json
from pathlib import Path

from datasets import DatasetDict, load_dataset


RAW_DIR = Path("data/raw")


def _write_split(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def download_pubmed_qa() -> DatasetDict:
    dataset = load_dataset("pubmed_qa", "pqa_labeled")

    for split_name, split_ds in dataset.items():
        _write_split(list(split_ds), RAW_DIR / f"{split_name}.jsonl")

    return dataset


def main() -> None:
    dataset = download_pubmed_qa()
    for split_name, split_ds in dataset.items():
        print(f"Saved {split_name}: {len(split_ds)} rows")


if __name__ == "__main__":
    main()
