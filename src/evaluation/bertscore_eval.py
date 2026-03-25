from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from llama_index.core import Settings
from bert_score import score as bert_score
from tqdm import tqdm

from config import config as app_config
from src.evaluation.result_utils import load_csv_if_exists
from src.evaluation.runtime import EvalRuntime, resolve_eval_runtime
from src.evaluation.run_metadata import annotate_with_run_metadata, compute_run_id, resolve_run_id
from src.graph.graph import compile_graph
from src.graph.nodes.critic import clear_critic_cache
from src.graph.nodes.retriever import clear_retriever_cache
from src.model.llm_wrapper import QuantizedHFLLM, register_llm
from src.model.loader import clear_model_cache, load_model_and_tokenizer
from src.utils.answer_cleaning import clean_for_scoring
from src.utils.memory import flush_gpu


TEST_SET_PATH = Path("data/eval/test_set.json")
OUT_PATH = Path("experiments/bertscore_results.csv")
MODEL_FREE_PATH = Path("experiments/model_free_eval_results.csv")
LEGACY_MODEL_FREE_PATH = Path("experiments/ragas_results.csv")


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


def _prediction_record(
    *,
    question: str,
    prediction_raw: str,
    reference_raw: str,
    prediction_cleaned: str | None = None,
) -> dict[str, str]:
    cleaned_prediction = clean_for_scoring(prediction_cleaned if prediction_cleaned is not None else prediction_raw)
    cleaned_reference = clean_for_scoring(reference_raw)
    return {
        "question": question,
        "prediction_raw": prediction_raw,
        "prediction_cleaned": cleaned_prediction,
        "reference_raw": reference_raw,
        "reference_cleaned": cleaned_reference,
    }


def _load_model_free_predictions(
    rows: list[dict],
    expected_run_id: str,
) -> list[dict[str, Any]] | None:
    questions = [str(row["question"]) for row in rows]
    for candidate in (MODEL_FREE_PATH, LEGACY_MODEL_FREE_PATH):
        if not candidate.exists():
            continue

        df = load_csv_if_exists(candidate)
        if df.empty or "question" not in df.columns:
            continue

        resolved_run_id = resolve_run_id(df, "question")
        if resolved_run_id != expected_run_id:
            continue

        indexed = df.drop_duplicates(subset=["question"], keep="last").set_index("question")
        if any(question not in indexed.index for question in questions):
            continue

        selected = indexed.loc[questions].reset_index()
        prediction_raw_col = "raw_answer" if "raw_answer" in selected.columns else "answer"
        prediction_clean_col = "answer" if "answer" in selected.columns else prediction_raw_col
        reference_raw_col = "ground_truth"

        return [
            _prediction_record(
                question=str(selected.loc[idx, "question"]),
                prediction_raw=str(selected.loc[idx, prediction_raw_col]),
                prediction_cleaned=str(selected.loc[idx, prediction_clean_col]),
                reference_raw=str(selected.loc[idx, reference_raw_col]),
            )
            for idx in range(len(selected))
        ]

    return None


def _build_predictions_from_graph(
    rows: list[dict],
) -> list[dict[str, Any]]:
    app = compile_graph()
    predictions: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Graph for BERTScore", unit="q", dynamic_ncols=True):
        result = app.invoke({"query": row["question"], "retry_count": 0})
        raw_answer = str(result.get("final_answer", result.get("draft_answer", "")) or "")
        predictions.append(
            _prediction_record(
                question=str(row["question"]),
                prediction_raw=raw_answer,
                reference_raw=str(row["ground_truth"]),
            )
        )
    return predictions


def _ensure_llm(runtime: EvalRuntime) -> None:
    if Settings.llm is not None:
        return

    loaded = load_model_and_tokenizer()
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


def _release_graph_resources() -> None:
    from llama_index.core import Settings

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
    return annotated


def _load_existing_results(path: Path, expected_run_id: str) -> pd.DataFrame:
    df = load_csv_if_exists(path)
    if df.empty or "question" not in df.columns:
        return pd.DataFrame()

    resolved_run_id = resolve_run_id(df, "question")
    if resolved_run_id != expected_run_id:
        return pd.DataFrame()
    return df.drop_duplicates(subset=["question"], keep="last")


def _is_complete_score(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    score = record.get("bertscore_f1")
    if score is None:
        return False
    try:
        return pd.notna(score)
    except Exception:
        return False


def _scored_record(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "question": row["question"],
        "prediction": row["prediction_cleaned"],
        "prediction_raw": row["prediction_raw"],
        "prediction_cleaned": row["prediction_cleaned"],
        "reference": row["reference_cleaned"],
        "reference_raw": row["reference_raw"],
        "reference_cleaned": row["reference_cleaned"],
        "bertscore_f1": float(score),
    }


def _has_complete_scores(df: pd.DataFrame, expected_rows: int) -> bool:
    if df.empty or "bertscore_f1" not in df.columns or len(df) != expected_rows:
        return False
    return pd.to_numeric(df["bertscore_f1"], errors="coerce").notna().all()


def _score_chunks(
    rows: list[dict[str, Any]],
    *,
    runtime: EvalRuntime,
    metadata: dict[str, Any],
    existing: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    existing_map = (
        existing.set_index("question").to_dict(orient="index")
        if not existing.empty and "question" in existing.columns
        else {}
    )
    row_map = {str(row["question"]): row for row in rows}
    all_questions = [str(row["question"]) for row in rows]

    scored_rows: list[dict[str, Any]] = []
    pending_rows = [row for row in rows if not _is_complete_score(existing_map.get(str(row["question"])))]

    if pending_rows:
        chunk_size = max(runtime.checkpoint_every_rows * 4, runtime.bertscore_batch_size * 4, 8)
        for chunk in tqdm(
            _iter_chunks(pending_rows, chunk_size),
            desc="BERTScore chunks",
            unit="chunk",
            dynamic_ncols=True,
        ):
            preds = [row["prediction_cleaned"] for row in chunk]
            refs = [row["reference_cleaned"] for row in chunk]
            _, _, f1 = bert_score(
                preds,
                refs,
                model_type=runtime.bertscore_model_type,
                verbose=False,
                device=_resolve_bertscore_device(),
                batch_size=runtime.bertscore_batch_size,
            )

            scored_chunk = [_scored_record(row, score) for row, score in zip(chunk, f1.tolist())]
            for scored_row in scored_chunk:
                existing_map[scored_row["question"]] = scored_row
            _write_results(
                pd.DataFrame([existing_map[q] for q in all_questions if q in existing_map]),
                questions=all_questions,
                metadata=metadata,
                output_path=output_path,
            )

    for question in all_questions:
        if question in existing_map:
            scored_rows.append(existing_map[question])
        else:
            # This can only happen if the existing file was incomplete and the
            # current run was interrupted before scoring the remaining rows.
            scored_rows.append(row_map[question])

    return _write_results(
        pd.DataFrame(scored_rows),
        questions=all_questions,
        metadata=metadata,
        output_path=output_path,
    )


def _iter_chunks(items: list[dict[str, Any]], chunk_size: int):
    size = max(1, int(chunk_size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_bertscore_evaluation(
    test_set: list[dict] | None = None,
    *,
    runtime: EvalRuntime | None = None,
    output_path: Path = OUT_PATH,
) -> pd.DataFrame:
    rows = _load_test_set(test_set)
    runtime = runtime or resolve_eval_runtime(profile=None, budget_seconds=None, judge_requested=False)
    questions = [str(row["question"]) for row in rows]
    metadata = runtime.metadata()
    expected_run_id = compute_run_id(questions, metadata=metadata)

    existing = _load_existing_results(output_path, expected_run_id)
    if _has_complete_scores(existing, len(rows)):
        return _write_results(existing, questions=questions, metadata=metadata, output_path=output_path)

    predictions = _load_model_free_predictions(rows, expected_run_id)
    if predictions is None:
        _ensure_llm(runtime)
        predictions = _build_predictions_from_graph(rows)
        _release_graph_resources()

    return _score_chunks(
        predictions,
        runtime=runtime,
        metadata=metadata,
        existing=existing,
        output_path=output_path,
    )


def main() -> None:
    df = run_bertscore_evaluation()
    print(f"BERTScore results saved to {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
