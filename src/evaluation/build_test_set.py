from __future__ import annotations

import json
from pathlib import Path


SOURCE_PATH = Path("data/raw/eval.jsonl")
OUT_PATH = Path("data/eval/test_set.json")


def build_test_set(limit: int = 50) -> list[dict]:
    rows = []
    with SOURCE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            row = json.loads(line)
            rows.append(
                {
                    "question": row["question"],
                    "ground_truth": row.get("long_answer") or row.get("ground_truth") or row.get("final_decision", ""),
                    "context_docs": row.get("context") or row.get("context_docs") or [],
                }
            )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    built = build_test_set()
    print(f"Wrote {len(built)} rows to {OUT_PATH}")
