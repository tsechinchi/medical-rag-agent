from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.graph.graph import compile_graph


TEST_SET_PATH = Path("data/eval/test_set.json")
OUT_PATH = Path("experiments/model_free_eval_results.csv")


def _load_test_set(test_set: list[dict] | None = None) -> list[dict]:
    if test_set is not None:
        return test_set
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


def _jaccard(a: str, b: str) -> float:
    lhs = set(a.lower().split())
    rhs = set(b.lower().split())
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def run_model_free_evaluation(llm=None, test_set: list[dict] | None = None) -> pd.DataFrame:
    rows = _load_test_set(test_set)
    app = compile_graph()
    outputs = []
    for row in rows:
        start = time.perf_counter()
        result = app.invoke({"query": row["question"], "retry_count": 0})
        latency = time.perf_counter() - start
        answer = result.get("final_answer", result.get("draft_answer", ""))
        contexts = [node.get_content() for node in result.get("retrieved_docs", [])]
        gt = row["ground_truth"]
        outputs.append(
            {
                "question": row["question"],
                "answer": answer,
                "ground_truth": gt,
                "faithfulness": round(result.get("faithfulness_score", 0.0), 4),
                "answer_relevancy": round(_jaccard(answer, gt), 4),
                "context_precision": round(_jaccard(" ".join(contexts[:3]), gt), 4),
                "context_recall": round(_jaccard(gt, " ".join(contexts)), 4),
                "avg_retries": result.get("retry_count", 0),
                "latency_per_query_s": round(latency, 4),
            }
        )
    df = pd.DataFrame(outputs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    legacy_path = Path("experiments/ragas_results.csv")
    df.to_csv(legacy_path, index=False)
    return df
