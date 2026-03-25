from __future__ import annotations

from pathlib import Path

from src.evaluation.model_free_eval import run_model_free_evaluation
from src.evaluation.runtime import EvalRuntime

OUT_PATH = Path("experiments/ragas_results.csv")


def run_ragas_evaluation(
    llm=None,
    test_set: list[dict] | None = None,
    *,
    runtime: EvalRuntime | None = None,
):
    return run_model_free_evaluation(llm=llm, test_set=test_set, runtime=runtime, output_path=OUT_PATH)
