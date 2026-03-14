from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.model_free_eval import run_model_free_evaluation


OUT_PATH = Path("experiments/ragas_results.csv")


def run_ragas_evaluation(llm=None, test_set: list[dict] | None = None) -> pd.DataFrame:
    df = run_model_free_evaluation(llm=llm, test_set=test_set)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    return df
