"""Ablation studies to measure component contributions.

Ablation A: MAX_RETRIES=0 (no critic loop)
Ablation B: Dense-only retrieval (remove BM25)
Ablation C: No reranker
Ablation D: Full evaluation with the fine-tuned QLoRA adapter

Run:
    python -m src.evaluation.ablations
    python -m src.evaluation.ablations --only A
    python -m src.evaluation.ablations --only D --profile t4-tight --with-ragas-judge
"""

from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path
from unittest.mock import patch

import pandas as pd
import torch
from bert_score import score as bert_score_fn
from tqdm import tqdm

# Ensure project root is on sys.path when executed as a plain script.
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from config import config as app_config
from src.evaluation.runtime import EvalRuntime, resolve_eval_runtime
from src.model.loader import load_model_and_tokenizer
from src.model.llm_wrapper import QuantizedHFLLM, register_llm
from src.utils.answer_cleaning import clean_for_scoring
from src.utils.seed import set_seed

TEST_SET_PATH = Path("data/eval/test_set.json")
DEFAULT_ABLATION_RUNTIME = resolve_eval_runtime(profile="fast", budget_seconds=None, judge_requested=False)


def _resolve_bertscore_device() -> str:
    configured = str(getattr(app_config, "BERTSCORE_DEVICE", "auto")).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured in {"cuda", "cpu"}:
        if configured == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_test_set() -> list[dict]:
    return json.loads(TEST_SET_PATH.read_text())


def _write_ablation_results(df: pd.DataFrame, filename: str, label: str) -> pd.DataFrame:
    out = Path("experiments") / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{label} saved to {out}")
    return df


def _run_pipeline(
    graph_app,
    test_set: list[dict],
    desc: str = "Running pipeline",
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Run graph on test set; return (questions, preds, refs, raw_results)."""
    questions, preds, refs, raw = [], [], [], []
    for entry in tqdm(test_set, desc=desc, unit="q", dynamic_ncols=True):
        q = entry["question"]
        result = graph_app.invoke({"query": q, "retry_count": 0})
        answer = result.get("final_answer", result.get("draft_answer", ""))
        questions.append(q)
        preds.append(answer)
        refs.append(entry["ground_truth"])
        raw.append(result)
    return questions, preds, refs, raw


def _compute_bertscore(
    preds: list[str],
    refs: list[str],
    *,
    runtime: EvalRuntime | None = None,
) -> list[float]:
    runtime = runtime or DEFAULT_ABLATION_RUNTIME
    cleaned_preds = [clean_for_scoring(pred) for pred in preds]
    cleaned_refs = [clean_for_scoring(ref) for ref in refs]
    _, _, f1 = bert_score_fn(
        cleaned_preds,
        cleaned_refs,
        model_type=runtime.bertscore_model_type,
        verbose=False,
        device=_resolve_bertscore_device(),
        batch_size=runtime.bertscore_batch_size,
    )
    return f1.tolist()


def _ablation_frame(
    questions: list[str],
    preds: list[str],
    refs: list[str],
    *,
    runtime: EvalRuntime | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": questions,
            "prediction": preds,
            "reference": refs,
            "bertscore_f1": _compute_bertscore(preds, refs, runtime=runtime),
        }
    )


def _build_and_register_llm(
    *,
    finetuned: bool = False,
    runtime: EvalRuntime | None = None,
) -> QuantizedHFLLM:
    runtime = runtime or DEFAULT_ABLATION_RUNTIME
    loaded = load_model_and_tokenizer(finetuned=finetuned)
    llm = QuantizedHFLLM(
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
    register_llm(llm)
    return llm


def _merge_qlora_results(ragas_df: pd.DataFrame, bert_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [col for col in ["question", "run_id", "bertscore_f1"] if col in bert_df.columns]
    if "run_id" in ragas_df.columns and "run_id" in bert_df.columns and len(merge_cols) >= 3:
        return ragas_df.merge(bert_df[merge_cols], on=["question", "run_id"], how="left")
    fallback_cols = [col for col in ["question", "bertscore_f1"] if col in bert_df.columns]
    return ragas_df.merge(bert_df[fallback_cols], on="question", how="left")


def ablation_no_loop(
    test_set: list[dict] | None = None,
    *,
    runtime: EvalRuntime | None = None,
) -> pd.DataFrame:
    """Run with MAX_RETRIES=0 to disable the critic retry loop."""
    if test_set is None:
        test_set = _load_test_set()
    runtime = runtime or DEFAULT_ABLATION_RUNTIME

    with patch("src.graph.graph.MAX_RETRIES", 0):
        from src.graph.graph import compile_graph

        app = compile_graph()

    questions, preds, refs, _ = _run_pipeline(app, test_set, desc="Ablation A (no loop)")
    return _write_ablation_results(
        _ablation_frame(questions, preds, refs, runtime=runtime),
        "ablation_no_loop.csv",
        "Ablation A",
    )


def ablation_dense_only(
    test_set: list[dict] | None = None,
    *,
    runtime: EvalRuntime | None = None,
) -> pd.DataFrame:
    """Run with BM25 removed and keep vector retrieval only."""
    if test_set is None:
        test_set = _load_test_set()
    runtime = runtime or DEFAULT_ABLATION_RUNTIME

    from llama_index.core.schema import NodeWithScore, QueryBundle
    from src.data.build_indices import load_index

    index = load_index()
    vector_retriever = index.as_retriever(similarity_top_k=10)

    def _dense_only_retrieve(state):
        sub_queries = state.get("sub_queries", [state["query"]])
        from src.graph.nodes.retriever import _deduplicate, _get_reranker

        reranker = _get_reranker()
        all_nodes: list[NodeWithScore] = []
        for sq in sub_queries:
            qb = QueryBundle(query_str=sq)
            candidates = vector_retriever.retrieve(qb)
            reranked = reranker.postprocess_nodes(candidates, query_bundle=qb)
            all_nodes.extend(reranked)
        return {"retrieved_docs": _deduplicate(all_nodes)}

    with patch("src.graph.graph.retriever", _dense_only_retrieve):
        from src.graph.graph import compile_graph

        app = compile_graph()

    questions, preds, refs, _ = _run_pipeline(app, test_set, desc="Ablation B (dense only)")
    return _write_ablation_results(
        _ablation_frame(questions, preds, refs, runtime=runtime),
        "ablation_dense_only.csv",
        "Ablation B",
    )


def ablation_no_rerank(
    test_set: list[dict] | None = None,
    *,
    runtime: EvalRuntime | None = None,
) -> pd.DataFrame:
    """Run with the reranker removed."""
    if test_set is None:
        test_set = _load_test_set()
    runtime = runtime or DEFAULT_ABLATION_RUNTIME

    from llama_index.core.schema import NodeWithScore, QueryBundle

    def _no_rerank_retrieve(state):
        sub_queries = state.get("sub_queries", [state["query"]])
        from src.graph.nodes.retriever import _deduplicate, _get_retriever

        hybrid = _get_retriever()
        all_nodes: list[NodeWithScore] = []
        for sq in sub_queries:
            qb = QueryBundle(query_str=sq)
            candidates = hybrid.retrieve(qb)
            candidates.sort(key=lambda n: n.score or 0.0, reverse=True)
            all_nodes.extend(candidates[:3])
        return {"retrieved_docs": _deduplicate(all_nodes)}

    with patch("src.graph.graph.retriever", _no_rerank_retrieve):
        from src.graph.graph import compile_graph

        app = compile_graph()

    questions, preds, refs, _ = _run_pipeline(app, test_set, desc="Ablation C (no rerank)")
    return _write_ablation_results(
        _ablation_frame(questions, preds, refs, runtime=runtime),
        "ablation_no_rerank.csv",
        "Ablation C",
    )


def ablation_qlora(
    test_set: list[dict] | None = None,
    *,
    runtime: EvalRuntime | None = None,
) -> pd.DataFrame:
    """Run the full evaluation stack with the fine-tuned adapter enabled."""
    if test_set is None:
        test_set = _load_test_set()
    runtime = runtime or DEFAULT_ABLATION_RUNTIME
    llm = _build_and_register_llm(finetuned=True, runtime=runtime)

    try:
        from src.evaluation.bertscore_eval import run_bertscore_evaluation
        from src.evaluation.model_free_eval import run_model_free_evaluation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ablation D requires src.evaluation.model_free_eval and "
            "src.evaluation.bertscore_eval to be available in the workspace."
        ) from exc

    ragas_df = run_model_free_evaluation(llm=llm, test_set=test_set, runtime=runtime)
    bert_df = run_bertscore_evaluation(test_set=test_set, runtime=runtime)
    return _write_ablation_results(_merge_qlora_results(ragas_df, bert_df), "ablation_qlora.csv", "Ablation D")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["A", "B", "C", "D"], default=None)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N questions (e.g. 10 for a quick smoke-test).",
    )
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=getattr(app_config, "EVAL_TIME_BUDGET_SECONDS", 10800),
        metavar="SECONDS",
        help="Wall-clock budget used when resolving the finetuned ablation runtime.",
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "fast", "t4-safe", "t4-tight"],
        default="auto",
        help="Runtime profile for finetuned ablation D.",
    )
    parser.add_argument(
        "--with-ragas-judge",
        action="store_true",
        help="Enable structured judge metrics for finetuned ablation D.",
    )
    args = parser.parse_args()

    set_seed(app_config.SEED)
    qlora_runtime = resolve_eval_runtime(
        profile=args.profile,
        budget_seconds=args.budget_seconds,
        judge_requested=bool(args.with_ragas_judge),
    )
    if args.only != "D":
        _build_and_register_llm(finetuned=False, runtime=DEFAULT_ABLATION_RUNTIME)

    test_set = _load_test_set()
    total_questions = len(test_set)
    if args.n_samples is not None:
        test_set = test_set[: args.n_samples]
        print(f"Using {len(test_set)} of {total_questions} questions (--n_samples {args.n_samples})")

    if args.only is None or args.only == "A":
        print("\nAblation A: No retry loop")
        ablation_no_loop(test_set, runtime=DEFAULT_ABLATION_RUNTIME)

    if args.only is None or args.only == "B":
        print("\nAblation B: Dense-only retrieval")
        ablation_dense_only(test_set, runtime=DEFAULT_ABLATION_RUNTIME)

    if args.only is None or args.only == "C":
        print("\nAblation C: No reranker")
        ablation_no_rerank(test_set, runtime=DEFAULT_ABLATION_RUNTIME)

    if args.only is None or args.only == "D":
        print("\nAblation D: Fine-tuned QLoRA")
        print(
            "Finetuned runtime: "
            f"{qlora_runtime.profile} | budget={qlora_runtime.budget_seconds}s | "
            f"judge={'on' if qlora_runtime.judge_enabled else 'off'}"
        )
        ablation_qlora(test_set, runtime=qlora_runtime)

    print("\nAll ablations complete.")


if __name__ == "__main__":
    main()
