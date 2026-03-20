from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as bert_score

from config import config as app_config
from src.graph.graph import compile_graph


import re as _re
def _clean_answer(text):
    text = _re.sub(r"\[\d+\]", "", text)
    text = _re.sub(r"\[Evidence[^\]]*\]", "", text)
    text = _re.sub(r"\[Partially[^\]]*\]", "", text)
    text = _re.sub(r"Medical disclaimer:.*", "", text, flags=_re.IGNORECASE)
    return text.strip()


TEST_SET_PATH = Path("data/eval/test_set.json")
OUT_PATH = Path("experiments/bertscore_results.csv")
MODEL_FREE_PATH = Path("experiments/model_free_eval_results.csv")
MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
BERTSCORE_BATCH_SIZE = int(os.getenv("BERTSCORE_BATCH_SIZE", "8"))


def _resolve_bertscore_device() -> str:
    configured = str(getattr(app_config, "BERTSCORE_DEVICE", "auto")).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured in {"cuda", "cpu"}:
        if configured == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_test_set(test_set: list[dict] | None = None) -> list[dict]:
    if test_set is not None:
        return test_set
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


def _load_predictions_from_model_free(rows: list[dict]) -> tuple[list[str], list[str], list[str]] | None:
    if not MODEL_FREE_PATH.exists():
        return None

    df = pd.read_csv(MODEL_FREE_PATH)
    required_cols = {"question", "answer", "ground_truth"}
    if not required_cols.issubset(df.columns):
        return None

    if rows is None:
        selected = df
    else:
        indexed = df.drop_duplicates(subset=["question"]).set_index("question")
        questions = [row["question"] for row in rows]
        if any(question not in indexed.index for question in questions):
            return None
        selected = indexed.loc[questions].reset_index()

    preds = selected["answer"].fillna("").astype(str).tolist()
    refs = selected["ground_truth"].fillna("").astype(str).tolist()
    questions = selected["question"].astype(str).tolist()
    return preds, refs, questions


def run_bertscore_evaluation(test_set: list[dict] | None = None) -> pd.DataFrame:
    rows = _load_test_set(test_set)
    cached = _load_predictions_from_model_free(rows)
    if cached is not None:
        preds, refs, questions = cached
    else:
        app = compile_graph()
        preds = []
        refs = []
        questions = []
        for row in rows:
            result = app.invoke({"query": row["question"], "retry_count": 0})
            preds.append(result.get("final_answer", result.get("draft_answer", "")))
            refs.append(row["ground_truth"])
            questions.append(row["question"])

    _, _, f1 = bert_score(
        preds,
        refs,
        model_type=MODEL_TYPE,
        verbose=False,
        device=_resolve_bertscore_device(),
        batch_size=BERTSCORE_BATCH_SIZE,
    )
    df = pd.DataFrame(
        {
            "question": questions,
            "prediction": preds,
            "reference": refs,
            "bertscore_f1": f1.tolist(),   # type: ignore
        }
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    return df


def main() -> None:
    df = run_bertscore_evaluation()
    print(f"BERTScore results saved to {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
