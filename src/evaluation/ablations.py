"""Ablation studies — evaluate pipeline variants to measure component contributions.

Ablation A: MAX_RETRIES=0 (no critic loop)
Ablation B: Dense-only retrieval (remove BM25)
Ablation C: No reranker
Ablation D: Full evaluation with the fine-tuned QLoRA adapter

Run:
    python -m src.evaluation.ablations           # all four
    python -m src.evaluation.ablations --only A   # just one
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# Ensure project root is on sys.path when executed as a plain script
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from bert_score import score as bert_score_fn
from tqdm import tqdm

from config import config as app_config
from src.model.llm_wrapper import QuantizedHFLLM, register_llm
from src.model.loader import load_model_and_tokenizer
from src.utils.seed import set_seed

TEST_SET_PATH = Path("data/eval/test_set.json")
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


def _load_test_set() -> list[dict]:
    return json.loads(TEST_SET_PATH.read_text())


def _run_pipeline(graph_app, test_set: list[dict], desc: str = "Running pipeline") -> tuple[list[str], list[str], list[str], list[dict]]:
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


def _compute_bertscore(preds: list[str], refs: list[str]) -> list[float]:
    _, _, F1 = bert_score_fn(preds, refs, model_type=BERTSCORE_MODEL, verbose=False, device="cpu")
    return F1.tolist()


def _build_and_register_llm(*, finetuned: bool = False) -> QuantizedHFLLM:
    loaded = load_model_and_tokenizer(finetuned=finetuned)
    llm = QuantizedHFLLM(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        max_new_tokens=getattr(app_config, "GENERATION_MAX_NEW_TOKENS", None),
        min_new_tokens=getattr(app_config, "GENERATION_MIN_NEW_TOKENS", 16),
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


def _summary_row(
    variant: str,
    questions: list[str],
    preds: list[str],
    refs: list[str],
    raw: list[dict],
    elapsed: float,
) -> dict:
    f1_scores = _compute_bertscore(preds, refs)
    avg_retries = sum(r.get("retry_count", 0) for r in raw) / max(len(raw), 1)
    avg_chunks = sum(len(r.get("retrieved_docs", [])) for r in raw) / max(len(raw), 1)
    return {
        "variant": variant,
        "n_questions": len(questions),
        "bertscore_f1_mean": round(sum(f1_scores) / max(len(f1_scores), 1), 4),
        "avg_retries": round(avg_retries, 2),
        "avg_chunks": round(avg_chunks, 2),
        "latency_per_query_s": round(elapsed / max(len(questions), 1), 2),
        "total_time_s": round(elapsed, 1),
    }


# ── Ablation A: No retry loop ───────────────────────────────────────────

def ablation_no_loop(test_set: list[dict] | None = None) -> pd.DataFrame:
    """Run with MAX_RETRIES=0 to disable the critic retry loop."""
    if test_set is None:
        test_set = _load_test_set()

    with patch("src.graph.graph.MAX_RETRIES", 0):
        # Re-import to pick up patched value
        from src.graph.graph import compile_graph
        app = compile_graph()

    t0 = time.time()
    questions, preds, refs, raw = _run_pipeline(app, test_set, desc="Ablation A (no loop)")
    elapsed = time.time() - t0

    df = pd.DataFrame({
        "question": questions,
        "prediction": preds,
        "reference": refs,
        "bertscore_f1": _compute_bertscore(preds, refs),
    })
    out = Path("experiments/ablation_no_loop.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Ablation A saved to {out}")
    return df


# ── Ablation B: Dense-only retrieval (no BM25) ──────────────────────────

def ablation_dense_only(test_set: list[dict] | None = None) -> pd.DataFrame:
    """Run with BM25 removed — vector retrieval only."""
    if test_set is None:
        test_set = _load_test_set()

    from src.data.build_indices import load_index
    from llama_index.core.schema import QueryBundle, NodeWithScore

    index = load_index()
    vector_retriever = index.as_retriever(similarity_top_k=10)

    def _dense_only_retrieve(state):
        sub_queries = state.get("sub_queries", [state["query"]])
        from src.graph.nodes.retriever import _get_reranker, _deduplicate
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

    t0 = time.time()
    questions, preds, refs, raw = _run_pipeline(app, test_set, desc="Ablation B (dense only)")
    elapsed = time.time() - t0

    df = pd.DataFrame({
        "question": questions,
        "prediction": preds,
        "reference": refs,
        "bertscore_f1": _compute_bertscore(preds, refs),
    })
    out = Path("experiments/ablation_dense_only.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Ablation B saved to {out}")
    return df


# ── Ablation C: No reranker ─────────────────────────────────────────────

def ablation_no_rerank(test_set: list[dict] | None = None) -> pd.DataFrame:
    """Run with the SentenceTransformerRerank removed."""
    if test_set is None:
        test_set = _load_test_set()

    from llama_index.core.schema import QueryBundle, NodeWithScore

    def _no_rerank_retrieve(state):
        sub_queries = state.get("sub_queries", [state["query"]])
        from src.graph.nodes.retriever import _get_retriever, _deduplicate
        hybrid = _get_retriever()
        all_nodes: list[NodeWithScore] = []
        for sq in sub_queries:
            qb = QueryBundle(query_str=sq)
            candidates = hybrid.retrieve(qb)
            # Keep top-3 by score without reranking
            candidates.sort(key=lambda n: n.score or 0.0, reverse=True)
            all_nodes.extend(candidates[:3])
        return {"retrieved_docs": _deduplicate(all_nodes)}

    with patch("src.graph.graph.retriever", _no_rerank_retrieve):
        from src.graph.graph import compile_graph
        app = compile_graph()

    t0 = time.time()
    questions, preds, refs, raw = _run_pipeline(app, test_set, desc="Ablation C (no rerank)")
    elapsed = time.time() - t0

    df = pd.DataFrame({
        "question": questions,
        "prediction": preds,
        "reference": refs,
        "bertscore_f1": _compute_bertscore(preds, refs),
    })
    out = Path("experiments/ablation_no_rerank.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Ablation C saved to {out}")
    return df


# ── Ablation D: Fine-tuned QLoRA model ──────────────────────────────────

def ablation_qlora(test_set: list[dict] | None = None) -> pd.DataFrame:
    """Run the full evaluation stack with the fine-tuned adapter enabled."""
    if test_set is None:
        test_set = _load_test_set()

    llm = _build_and_register_llm(finetuned=True)

    try:
        from src.evaluation.bertscore_eval import run_bertscore_evaluation
        from src.evaluation.model_free_eval import run_model_free_evaluation
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ablation D requires src.evaluation.model_free_eval and "
            "src.evaluation.bertscore_eval to be available in the workspace."
        ) from exc

    ragas_df = run_model_free_evaluation(llm=llm, test_set=test_set)
    bert_df = run_bertscore_evaluation(test_set=test_set)

    if "prediction" in bert_df.columns and "answer" not in bert_df.columns:
        bert_df = bert_df.rename(columns={"prediction": "answer"})
    if "reference" in bert_df.columns and "ground_truth" not in bert_df.columns:
        bert_df = bert_df.rename(columns={"reference": "ground_truth"})

    merge_cols = [col for col in ["question", "bertscore_f1"] if col in bert_df.columns]
    merged = ragas_df.merge(bert_df[merge_cols], on="question", how="left")

    out = Path("experiments/ablation_qlora.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Ablation D saved to {out}")
    return merged


# ── Entry point ──────────────────────────────────────────────────────────

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
    args = parser.parse_args()

    set_seed(app_config.SEED)
    if args.only != "D":
        _build_and_register_llm(finetuned=False)

    test_set = _load_test_set()
    if args.n_samples is not None:
        test_set = test_set[: args.n_samples]
        print(f"Using {len(test_set)} of 50 questions (--n_samples {args.n_samples})")

    if args.only is None or args.only == "A":
        print("\n── Ablation A: No retry loop ──")
        ablation_no_loop(test_set)

    if args.only is None or args.only == "B":
        print("\n── Ablation B: Dense-only retrieval ──")
        ablation_dense_only(test_set)

    if args.only is None or args.only == "C":
        print("\n── Ablation C: No reranker ──")
        ablation_no_rerank(test_set)

    if args.only is None or args.only == "D":
        print("\n── Ablation D: Fine-tuned QLoRA ──")
        ablation_qlora(test_set)

    print("\nAll ablations complete.")


if __name__ == "__main__":
    main()
