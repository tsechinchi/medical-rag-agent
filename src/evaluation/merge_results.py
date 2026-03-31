"""Merge all evaluation CSVs into experiments/all_results.csv.

Run:
    python -m src.evaluation.merge_results
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path when executed as a plain script.
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from src.evaluation.result_utils import align_eval_frames, load_csv_if_exists, safe_mean

EXPERIMENTS = Path("experiments")
OUT_PATH = EXPERIMENTS / "all_results.csv"
MODEL_FREE_PATH = EXPERIMENTS / "model_free_eval_results.csv"
LEGACY_MODEL_FREE_PATH = EXPERIMENTS / "ragas_results.csv"
BERT_PATH = EXPERIMENTS / "bertscore_results.csv"
ABLATION_MAP = {
    "ablation_no_loop": "ablation_no_loop.csv",
    "ablation_dense_only": "ablation_dense_only.csv",
    "ablation_no_rerank": "ablation_no_rerank.csv",
    "finetuned_pipeline": "ablation_qlora.csv",
}


def _load_model_free_df() -> pd.DataFrame:
    for path in (MODEL_FREE_PATH, LEGACY_MODEL_FREE_PATH):
        df = load_csv_if_exists(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def _load_bert_df() -> pd.DataFrame:
    return load_csv_if_exists(BERT_PATH)


def _existing_source_paths() -> list[Path]:
    paths = [
        MODEL_FREE_PATH,
        LEGACY_MODEL_FREE_PATH,
        BERT_PATH,
        *(EXPERIMENTS / fname for fname in ABLATION_MAP.values()),
    ]
    return [path for path in paths if path.exists()]


def _warn_if_summary_stale() -> None:
    if not OUT_PATH.exists():
        return

    summary_mtime = OUT_PATH.stat().st_mtime
    newer_sources = [path.name for path in _existing_source_paths() if path.stat().st_mtime > summary_mtime]
    if newer_sources:
        print(
            "Warning: experiments/all_results.csv is older than these source files: "
            + ", ".join(newer_sources)
        )


def _resolve_full_pipeline_frames(model_free_df: pd.DataFrame, bert_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = model_free_df if not model_free_df.empty else bert_df
    if model_free_df.empty or bert_df.empty:
        return source_df, bert_df

    aligned_bert_df = align_eval_frames(model_free_df, bert_df)
    if aligned_bert_df.empty:
        print(
            "Warning: full_pipeline summary is skipping BERTScore because the model-free "
            "and BERTScore runs do not share the same run fingerprint."
        )
    return source_df, aligned_bert_df


def _abstention_precision(df: pd.DataFrame) -> float | None:
    if "abstention_detected" not in df.columns or "faithfulness_nli" not in df.columns:
        return None
    abstentions = df[df["abstention_detected"].fillna(0).astype(int) == 1]
    if abstentions.empty:
        return None
    correct = (abstentions["faithfulness_nli"].fillna(0).astype(float) == 1.0).mean()
    return round(float(correct), 4)


def _unsupported_claims_rate(df: pd.DataFrame) -> float | None:
    if "unsupported_claims_count" not in df.columns:
        return None
    series = pd.to_numeric(df["unsupported_claims_count"], errors="coerce").fillna(0.0)
    return round(float((series > 0).mean()), 4)


def _corrupted_output_rate(df: pd.DataFrame) -> float | None:
    if "corrupted_output_detected" not in df.columns:
        return None
    series = pd.to_numeric(df["corrupted_output_detected"], errors="coerce").fillna(0.0)
    return round(float(series.mean()), 4)


def _summarize_variant(variant: str, source_df: pd.DataFrame, bert_df: pd.DataFrame | None = None) -> dict:
    bert_source_df = bert_df if bert_df is not None else source_df
    return {
        "variant": variant,
        "n_questions": len(source_df),
        "faithfulness": safe_mean(source_df, "faithfulness"),
        "faithfulness_nli": safe_mean(source_df, "faithfulness_nli"),
        "faithfulness_ragas": safe_mean(source_df, "faithfulness_ragas"),
        "answer_relevancy": safe_mean(source_df, "answer_relevancy", non_abstention_only=True),
        "context_precision": safe_mean(source_df, "context_precision", non_abstention_only=True),
        "context_recall": safe_mean(source_df, "context_recall", non_abstention_only=True),
        "bertscore_f1_mean": safe_mean(bert_source_df, "bertscore_f1", non_abstention_only=True),
        "bertscore_f1_all_mean": safe_mean(bert_source_df, "bertscore_f1"),
        "bertscore_f1_abstention_mean": safe_mean(
            bert_source_df[bert_source_df["abstention_detected"].fillna(0).astype(int) == 1]
            if not bert_source_df.empty and "abstention_detected" in bert_source_df.columns
            else pd.DataFrame(),
            "bertscore_f1",
        ),
        "avg_retries": safe_mean(source_df, "avg_retries"),
        "latency_per_query_s": safe_mean(source_df, "latency_per_query_s"),
        "abstention_precision": _abstention_precision(source_df),
        "unsupported_claims_rate": _unsupported_claims_rate(source_df),
        "corrupted_output_rate": _corrupted_output_rate(source_df),
    }


def merge_results() -> pd.DataFrame:
    _warn_if_summary_stale()
    rows: list[dict] = []

    model_free_df = _load_model_free_df()
    bert_df = _load_bert_df()

    if not model_free_df.empty or not bert_df.empty:
        source_df, aligned_bert_df = _resolve_full_pipeline_frames(model_free_df, bert_df)
        rows.append(_summarize_variant("full_pipeline", source_df, aligned_bert_df))

    for variant, fname in tqdm(ABLATION_MAP.items(), desc="Merging ablations", unit="file"):
        path = EXPERIMENTS / fname
        if path.exists():
            df = pd.read_csv(path)
            rows.append(_summarize_variant(variant, df))

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Unified results saved to {OUT_PATH}  ({len(df)} variants)")
    return df


def main() -> None:
    df = merge_results()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
