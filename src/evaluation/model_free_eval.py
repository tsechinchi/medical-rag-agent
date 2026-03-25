from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from llama_index.core import Settings

from src.evaluation.result_utils import load_csv_if_exists
from src.evaluation.runtime import EvalRuntime, resolve_eval_runtime
from src.evaluation.run_metadata import annotate_with_run_metadata, compute_run_id, resolve_run_id
from src.evaluation.structured_judge import JudgeRow, build_structured_judge
from src.graph.graph import compile_graph
from src.graph.nodes.critic import clear_critic_cache
from src.graph.nodes.retriever import clear_retriever_cache
from src.model.loader import clear_model_cache
from src.utils.answer_cleaning import clean_for_scoring, is_abstention, is_corrupted_output
from src.utils.memory import flush_gpu


TEST_SET_PATH = Path("data/eval/test_set.json")
OUT_PATH = Path("experiments/model_free_eval_results.csv")
LEGACY_OUT_PATH = Path("experiments/ragas_results.csv")
_REQUIRED_BASE_COLUMNS = {
    "question",
    "answer",
    "raw_answer",
    "ground_truth",
    "retrieved_contexts_json",
    "faithfulness_nli",
    "latency_per_query_s",
    "abstention_detected",
}


def _load_test_set(test_set: list[dict] | None = None) -> list[dict]:
    if test_set is not None:
        return test_set
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


def _iter_chunks(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _row_has_base_columns(row: dict[str, Any]) -> bool:
    return _REQUIRED_BASE_COLUMNS.issubset(row.keys())


def _load_existing_rows(path: Path, expected_run_id: str) -> dict[str, dict[str, Any]]:
    df = load_csv_if_exists(path)
    if df.empty or "question" not in df.columns:
        return {}

    resolved_run_id = resolve_run_id(df, "question")
    if resolved_run_id != expected_run_id:
        return {}

    deduped = df.drop_duplicates(subset=["question"], keep="last")
    rows: dict[str, dict[str, Any]] = {}
    for record in deduped.to_dict(orient="records"):
        rows[str(record.get("question", ""))] = record
    return rows


def _build_base_record(
    *,
    question: str,
    ground_truth: str,
    result: dict[str, Any],
    latency_s: float,
    max_context_docs: int,
) -> dict[str, Any]:
    raw_answer = str(result.get("final_answer", result.get("draft_answer", "")) or "")
    answer = clean_for_scoring(raw_answer)
    retrieved_docs = result.get("retrieved_docs", []) or []
    retrieved_contexts = [
        str(node.get_content())
        for node in retrieved_docs[:max_context_docs]
        if hasattr(node, "get_content")
    ]
    top_score = 0.0
    if retrieved_docs:
        top_doc = retrieved_docs[0]
        top_score = float(getattr(top_doc, "score", 0.0) or 0.0)

    faithfulness = round(float(result.get("faithfulness_score", 0.0) or 0.0), 4)
    retry_count = int(result.get("retry_count", 0) or 0)
    unsupported_claims = int(result.get("unsupported_claims_count", 0) or 0)

    return {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "ground_truth": ground_truth,
        "retrieved_contexts_json": json.dumps(retrieved_contexts, ensure_ascii=False),
        "faithfulness": faithfulness,
        "faithfulness_nli": faithfulness,
        "faithfulness_ragas": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
        "latency_per_query_s": round(float(latency_s), 4),
        "avg_retries": retry_count,
        "abstention_detected": 1 if is_abstention(answer) else 0,
        "unsupported_claims_count": unsupported_claims,
        "citation_count": raw_answer.count("[") if "[" in raw_answer else 0,
        "retry_count": retry_count,
        "evidence_score": top_score,
        "corrupted_output_detected": 1 if is_corrupted_output(raw_answer) else 0,
    }


def _release_graph_resources() -> None:
    Settings.llm = None
    clear_model_cache()
    clear_retriever_cache()
    clear_critic_cache()
    flush_gpu()


def _write_results(
    df: pd.DataFrame,
    *,
    questions: list[str],
    metadata: dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    annotated = annotate_with_run_metadata(df, questions, metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_path, index=False)
    annotated.to_csv(LEGACY_OUT_PATH, index=False)
    return annotated


def _score_with_judge(
    df: pd.DataFrame,
    runtime: EvalRuntime,
    *,
    questions: list[str],
    metadata: dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    if df.empty or not runtime.judge_enabled:
        return df

    judge = build_structured_judge(
        model_id=runtime.judge_model_id,
        max_new_tokens=runtime.judge_max_new_tokens,
        max_context_docs=runtime.max_context_docs,
        timeout_seconds=runtime.judge_timeout_seconds,
    )

    try:
        pending_indices = [
            idx
            for idx, row in df.iterrows()
            if pd.isna(row.get("faithfulness_ragas"))
            or pd.isna(row.get("answer_relevancy"))
            or pd.isna(row.get("context_precision"))
            or pd.isna(row.get("context_recall"))
        ]
        if not pending_indices:
            return df

        for chunk_indices in tqdm(
            _iter_chunks(pending_indices, runtime.checkpoint_every_rows),
            desc="Judge scoring",
            unit="chunk",
            dynamic_ncols=True,
        ):
            judge_rows = []
            for idx in chunk_indices:
                row = df.loc[idx]
                contexts: list[str] = []
                raw_contexts = row.get("retrieved_contexts_json", "[]")
                if isinstance(raw_contexts, str):
                    try:
                        parsed_contexts = json.loads(raw_contexts)
                        if isinstance(parsed_contexts, list):
                            contexts = [str(ctx) for ctx in parsed_contexts]
                    except Exception:
                        contexts = []
                judge_rows.append(
                    JudgeRow(
                        question=str(row["question"]),
                        answer=str(row.get("answer", "")),
                        reference=str(row.get("ground_truth", "")),
                        contexts=contexts[: runtime.max_context_docs],
                        abstention_detected=bool(int(row.get("abstention_detected", 0) or 0)),
                    )
                )

            scored_rows = judge.score_rows(judge_rows, batch_size=runtime.judge_batch_size)
            for idx, scored in zip(chunk_indices, scored_rows):
                df.loc[idx, "faithfulness_ragas"] = round(float(scored["faithfulness"]), 4)
                df.loc[idx, "answer_relevancy"] = round(float(scored["answer_relevancy"]), 4)
                df.loc[idx, "context_precision"] = round(float(scored["context_precision"]), 4)
                df.loc[idx, "context_recall"] = round(float(scored["context_recall"]), 4)
                df.loc[idx, "judge_raw_output"] = scored.get("judge_raw_output", "")
                df.loc[idx, "judge_used_fallback"] = bool(scored.get("judge_used_fallback", False))

            df = _write_results(df, questions=questions, metadata=metadata, output_path=output_path)

        return df
    finally:
        judge.close()


def run_model_free_evaluation(
    test_set: list[dict] | None = None,
    *,
    llm: Any | None = None,
    runtime: EvalRuntime | None = None,
    output_path: Path = OUT_PATH,
) -> pd.DataFrame:
    rows = _load_test_set(test_set)
    runtime = runtime or resolve_eval_runtime(profile=None, budget_seconds=None, judge_requested=False)
    if llm is not None:
        Settings.llm = llm

    questions = [str(row["question"]) for row in rows]
    metadata = runtime.metadata()
    expected_run_id = compute_run_id(questions, metadata=metadata)

    existing_rows = _load_existing_rows(output_path, expected_run_id)
    needs_generation = any(
        question not in existing_rows or not _row_has_base_columns(existing_rows[question])
        for question in questions
    )

    if needs_generation:
        app = compile_graph()
        records: list[dict[str, Any]] = []
        for row in tqdm(rows, desc="Graph eval", unit="q", dynamic_ncols=True):
            question = str(row["question"])
            existing = existing_rows.get(question)
            if existing is not None and _row_has_base_columns(existing):
                records.append(existing)
                continue

            start = time.perf_counter()
            result = app.invoke({"query": question, "retry_count": 0})
            latency = time.perf_counter() - start
            records.append(
                _build_base_record(
                    question=question,
                    ground_truth=str(row["ground_truth"]),
                    result=result,
                    latency_s=latency,
                    max_context_docs=runtime.max_context_docs,
                )
            )

            if len(records) % runtime.checkpoint_every_rows == 0:
                _write_results(pd.DataFrame(records), questions=questions, metadata=metadata, output_path=output_path)

        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame([existing_rows[question] for question in questions])

    df = _write_results(df, questions=questions, metadata=metadata, output_path=output_path)

    _release_graph_resources()

    if runtime.judge_enabled:
        df = _score_with_judge(
            df,
            runtime,
            questions=questions,
            metadata=metadata,
            output_path=output_path,
        )
        df = _write_results(df, questions=questions, metadata=metadata, output_path=output_path)

    return df
