from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


def compute_run_id(questions: Iterable[str]) -> str:
    normalized = [str(question) for question in questions]
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def annotate_with_run_metadata(df: pd.DataFrame, questions: Iterable[str]) -> pd.DataFrame:
    question_list = [str(question) for question in questions]
    annotated = df.copy()
    annotated["run_id"] = compute_run_id(question_list)
    annotated["run_question_count"] = len(question_list)
    return annotated


def resolve_run_id(df: pd.DataFrame, question_col: str) -> str | None:
    if df.empty:
        return None
    if "run_id" in df.columns:
        run_ids = df["run_id"].dropna().astype(str).unique().tolist()
        if len(run_ids) == 1:
            return run_ids[0]
    if question_col in df.columns:
        return compute_run_id(df[question_col].fillna("").astype(str).tolist())
    return None
