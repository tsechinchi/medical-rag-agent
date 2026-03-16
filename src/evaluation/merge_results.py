"""Merge all evaluation CSVs into experiments/all_results.csv.

Run:
    python -m src.evaluation.merge_results
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# Ensure project root is on sys.path when executed as a plain script
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path

import pandas as pd
from tqdm import tqdm

EXPERIMENTS = Path("experiments")
OUT_PATH = EXPERIMENTS / "all_results.csv"


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if col in df.columns:
        # For metrics that should be masked for abstention rows, filter them out
        if col in ["answer_relevancy", "context_precision", "context_recall"]:
            # Only compute mean for non-abstention rows
            if "abstention_detected" in df.columns:
                filtered_df = df[df["abstention_detected"] == 0]
                if len(filtered_df) > 0:
                    return round(filtered_df[col].mean(), 4)
                else:
                    return None  # All rows were abstentions
        return round(df[col].mean(), 4)
    return None


def _summarize_variant(variant: str, df: pd.DataFrame) -> dict:
    # Calculate abstention precision if we have the necessary columns
    abstention_precision = None
    if "abstention_detected" in df.columns and "faithfulness_nli" in df.columns:
        abstentions = df[df["abstention_detected"] == 1]
        if len(abstentions) > 0:
            # Correct abstention = faithfulness_nli == 1.0 (system correctly abstained)
            correct_abstentions = len(abstentions[abstentions["faithfulness_nli"] == 1.0])
            abstention_precision = round(correct_abstentions / len(abstentions), 4)

    return {
        "variant": variant,
        "n_questions": len(df),
        "faithfulness": _safe_mean(df, "faithfulness"),
        "faithfulness_nli": _safe_mean(df, "faithfulness_nli"),
        "faithfulness_ragas": _safe_mean(df, "faithfulness_ragas"),
        "answer_relevancy": _safe_mean(df, "answer_relevancy"),
        "context_precision": _safe_mean(df, "context_precision"),
        "context_recall": _safe_mean(df, "context_recall"),
        "bertscore_f1_mean": _safe_mean(df, "bertscore_f1"),
        "avg_retries": _safe_mean(df, "avg_retries"),
        "latency_per_query_s": _safe_mean(df, "latency_per_query_s"),
        "abstention_precision": abstention_precision,
        "unsupported_claims_rate": _safe_mean(df, "unsupported_claims_count") if "unsupported_claims_count" in df.columns else None,
    }


def merge_results() -> pd.DataFrame:
    rows: list[dict] = []

    # Full pipeline — model-free evaluation
    model_free_path = EXPERIMENTS / "model_free_eval_results.csv"
    legacy_ragas_path = EXPERIMENTS / "ragas_results.csv"
    bert_path = EXPERIMENTS / "bertscore_results.csv"

    bdf = pd.read_csv(bert_path) if bert_path.exists() else pd.DataFrame()

    eval_path = model_free_path if model_free_path.exists() else legacy_ragas_path

    if eval_path.exists():
        rdf = pd.read_csv(eval_path)
        n_q = len(rdf)
        faith = _safe_mean(rdf, "faithfulness")
        ar    = _safe_mean(rdf, "answer_relevancy")
        cp    = _safe_mean(rdf, "context_precision")
        cr    = _safe_mean(rdf, "context_recall")
        has_valid = any(v is not None and pd.notna(v) for v in [faith, ar, cp, cr])
    else:
        n_q, faith, ar, cp, cr, has_valid = 0, None, None, None, None, False

    # Always add a full_pipeline row if we have either RAGAS or BERTScore data
    if has_valid or len(bdf) > 0:
        # Calculate abstention metrics for full pipeline
        abstention_precision = None
        unsupported_claims_rate = None
        if has_valid and "abstention_detected" in rdf.columns:
            abstentions = rdf[rdf["abstention_detected"] == 1]
            if len(abstentions) > 0:
                correct_abstentions = len(abstentions[abstentions["faithfulness_nli"] == 1.0])
                abstention_precision = round(correct_abstentions / len(abstentions), 4)
            if "unsupported_claims_count" in rdf.columns:
                unsupported_claims_rate = round(rdf["unsupported_claims_count"].mean(), 4)

        rows.append({
            "variant": "full_pipeline",
            "n_questions": n_q if has_valid else (len(bdf) if len(bdf) else 0),
            "faithfulness": faith if has_valid else None,
            "faithfulness_nli": _safe_mean(rdf, "faithfulness_nli") if has_valid else None,
            "faithfulness_ragas": _safe_mean(rdf, "faithfulness_ragas") if has_valid else None,
            "answer_relevancy": _safe_mean(rdf, "answer_relevancy") if has_valid else None,
            "context_precision": _safe_mean(rdf, "context_precision") if has_valid else None,
            "context_recall": _safe_mean(rdf, "context_recall") if has_valid else None,
            "bertscore_f1_mean": _safe_mean(bdf, "bertscore_f1") if len(bdf) else None,
            "avg_retries": _safe_mean(rdf, "avg_retries") if has_valid else None,
            "latency_per_query_s": _safe_mean(rdf, "latency_per_query_s") if has_valid else None,
            "abstention_precision": abstention_precision,
            "unsupported_claims_rate": unsupported_claims_rate,
        })

    # Ablations
    ablation_map = {
        "ablation_no_loop": "ablation_no_loop.csv",
        "ablation_dense_only": "ablation_dense_only.csv",
        "ablation_no_rerank": "ablation_no_rerank.csv",
        "finetuned_pipeline": "ablation_qlora.csv",
    }

    for variant, fname in tqdm(ablation_map.items(), desc="Merging ablations", unit="file"):
        p = EXPERIMENTS / fname
        if p.exists():
            adf = pd.read_csv(p)
            rows.append(_summarize_variant(variant, adf))

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
