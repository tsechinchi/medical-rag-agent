from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from bert_score import score as bert_score

from src.graph.graph import compile_graph


TEST_SET_PATH = Path("data/eval/test_set.json")
OUT_PATH = Path("experiments/bertscore_results.csv")
MODEL_TYPE = "microsoft/deberta-xlarge-mnli"


def _load_test_set(test_set: list[dict] | None = None) -> list[dict]:
    if test_set is not None:
        return test_set
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


def run_bertscore_evaluation(test_set: list[dict] | None = None) -> pd.DataFrame:
    rows = _load_test_set(test_set)
    app = compile_graph()
    preds = []
    refs = []
    questions = []
    for row in rows:
        result = app.invoke({"query": row["question"], "retry_count": 0})
        preds.append(result.get("final_answer", result.get("draft_answer", "")))
        refs.append(row["ground_truth"])
        questions.append(row["question"])

    _, _, f1 = bert_score(preds, refs, model_type=MODEL_TYPE, verbose=False, device="cpu")
    df = pd.DataFrame(
        {
            "question": questions,
            "prediction": preds,
            "reference": refs,
            "bertscore_f1": f1.tolist(),
        }
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    return df
