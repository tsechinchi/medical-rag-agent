from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.run_metadata import resolve_run_id


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def align_eval_frames(model_free_df: pd.DataFrame, bert_df: pd.DataFrame) -> pd.DataFrame:
    if model_free_df.empty or bert_df.empty:
        return pd.DataFrame()

    model_run_id = resolve_run_id(model_free_df, "question")
    bert_run_id = resolve_run_id(bert_df, "question")
    if not model_run_id or not bert_run_id or model_run_id != bert_run_id:
        return pd.DataFrame()

    merge_keys = ["question"]
    if "run_id" in model_free_df.columns and "run_id" in bert_df.columns:
        merge_keys.append("run_id")

    return model_free_df.merge(
        bert_df,
        on=merge_keys,
        how="inner",
        suffixes=("_model_free", "_bertscore"),
    )


def abstention_mask(df: pd.DataFrame, column: str = "abstention_detected") -> pd.Series:
    if column not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    return df[column].fillna(0).astype(int) == 0


def safe_mean(
    df: pd.DataFrame,
    column: str,
    *,
    non_abstention_only: bool = False,
    abstention_column: str = "abstention_detected",
) -> float | None:
    if column not in df.columns:
        return None

    series = pd.to_numeric(df[column], errors="coerce")
    if non_abstention_only and abstention_column in df.columns:
        series = series[abstention_mask(df, abstention_column)]

    series = series.dropna()
    if series.empty:
        return None
    return round(float(series.mean()), 4)
