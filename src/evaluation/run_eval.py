"""Unified evaluation entrypoint.

Runs:
  1. model-free evaluation
  2. BERTScore evaluation
  3. merged summary generation

Run:
    python -m src.evaluation.run_eval
    python -m src.evaluation.run_eval --n_samples 10
    python -m src.evaluation.run_eval --skip-bertscore
    python -m src.evaluation.run_eval --profile auto --budget-seconds 10800 --with-ragas-judge
"""

from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path

from config import config as app_config
from src.evaluation.bertscore_eval import run_bertscore_evaluation
from src.evaluation.merge_results import merge_results
from src.evaluation.model_free_eval import run_model_free_evaluation
from src.evaluation.runtime import resolve_eval_runtime
from src.model.llm_wrapper import QuantizedHFLLM, register_llm
from src.model.loader import load_model_and_tokenizer
from src.utils.seed import set_seed
from src.utils.memory import flush_gpu

TEST_SET_PATH = Path("data/eval/test_set.json")


def _load_test_set(n_samples: int | None) -> list[dict] | None:
    if n_samples is None:
        return None
    return json.loads(TEST_SET_PATH.read_text())[:n_samples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None, metavar="N")
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=getattr(app_config, "EVAL_TIME_BUDGET_SECONDS", 10800),
        metavar="SECONDS",
        help="Wall-clock budget used to choose the evaluation profile and judge settings.",
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "fast", "t4-safe", "t4-tight"],
        default="auto",
        help="Evaluation profile: auto, fast, t4-safe, or t4-tight.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Override random seed for this run (defaults to config.SEED).",
    )
    parser.add_argument("--skip-model-free", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument(
        "--with-ragas-judge",
        action="store_true",
        help=(
            "Enable the structured single-pass judge metrics (answer_relevancy/context_*). "
            "Useful for the full 3-hour T4 budgeted eval run."
        ),
    )
    args = parser.parse_args()

    set_seed(app_config.SEED if args.seed is None else args.seed)
    runtime = resolve_eval_runtime(
        profile=args.profile,
        budget_seconds=args.budget_seconds,
        judge_requested=bool(args.with_ragas_judge),
    )

    test_set = _load_test_set(args.n_samples)
    if test_set is not None:
        print(f"Using {len(test_set)} questions (--n_samples {args.n_samples})")
    print(
        "Evaluation profile: "
        f"{runtime.profile} | budget={runtime.budget_seconds}s | judge={'on' if runtime.judge_enabled else 'off'}"
    )

    if not args.skip_model_free:
        loaded = load_model_and_tokenizer()
        register_llm(
            QuantizedHFLLM(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                max_new_tokens=runtime.generation_max_new_tokens,
                min_new_tokens=runtime.generation_min_new_tokens,
                temperature=getattr(app_config, "GENERATION_TEMPERATURE", 0.0),
            context_window=getattr(app_config, "INFERENCE_CONTEXT_WINDOW", 2048),
            do_sample=False,
            repetition_penalty=1.05,
            top_p=1.0,
                top_k=1,
                num_beams=1,
            )
        )

    if not args.skip_model_free:
        print("Running model-free evaluation...")
        run_model_free_evaluation(test_set=test_set, runtime=runtime)
        flush_gpu()

    if not args.skip_bertscore:
        print("Running BERTScore evaluation...")
        run_bertscore_evaluation(test_set=test_set, runtime=runtime)
        flush_gpu()

    print("Merging evaluation outputs...")
    merge_results()
    print("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
