from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_PATH = Path("data/raw/eval.jsonl")
OUT_PATH = Path("data/eval/test_set.json")


def build_test_set(limit: int | None = None, expected_min: int | None = None) -> list[dict]:
    rows = []
    with SOURCE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            rows.append(
                {
                    "question": row["question"],
                    "ground_truth": row.get("long_answer") or row.get("ground_truth") or row.get("final_decision", ""),
                    "context_docs": row.get("context") or row.get("context_docs") or [],
                }
            )

    if expected_min is not None and len(rows) < expected_min:
        print(
            f"Warning: built test set has {len(rows)} rows, below expected_min={expected_min}. "
            "If this run is for smoke-testing, this is fine; otherwise increase source coverage."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--expected-min",
        type=int,
        default=200,
        metavar="N",
        help="Warn if fewer than N rows were written.",
    )
    args = parser.parse_args()
    built = build_test_set(limit=args.limit, expected_min=args.expected_min)
    print(f"Wrote {len(built)} rows to {OUT_PATH}")
