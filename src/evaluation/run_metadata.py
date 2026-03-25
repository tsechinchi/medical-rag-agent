from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import pandas as pd


def _question_list(questions: Iterable[str]) -> list[str]:
    return [str(question) for question in questions]


def _normalize_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return "{}"
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)


def _run_id_from_questions(questions: Iterable[str], metadata: dict[str, Any] | None = None) -> str:
    normalized = _question_list(questions)
    payload = "\n".join(normalized).encode("utf-8")
    meta_blob = _normalize_metadata(metadata).encode("utf-8")
    return hashlib.sha256(payload + b"\n" + meta_blob).hexdigest()[:16]


def compute_run_id(questions: Iterable[str], metadata: dict[str, Any] | None = None) -> str:
    return _run_id_from_questions(questions, metadata)


def annotate_with_run_metadata(
    df: pd.DataFrame,
    questions: Iterable[str],
    *,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    question_list = _question_list(questions)
    annotated = df.copy()
    annotated["run_id"] = _run_id_from_questions(question_list, metadata)
    annotated["run_question_count"] = len(question_list)
    if metadata is not None:
        annotated["run_metadata_json"] = _normalize_metadata(metadata)
        for key, value in metadata.items():
            annotated[f"run_{key}"] = value
    return annotated


def _single_value(df: pd.DataFrame, column: str) -> str | None:
    values = df[column].dropna().astype(str).unique().tolist()
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return None
    return None


def _unique_count(df: pd.DataFrame, column: str) -> int:
    return len(df[column].dropna().astype(str).unique().tolist())


def _question_series(df: pd.DataFrame, question_col: str) -> list[str] | None:
    if question_col not in df.columns:
        return None
    return df[question_col].fillna("").astype(str).tolist()


def resolve_run_id(df: pd.DataFrame, question_col: str) -> str | None:
    if df.empty:
        return None
    if "run_id" in df.columns:
        run_id = _single_value(df, "run_id")
        if run_id is not None:
            return run_id
        if _unique_count(df, "run_id") > 1:
            return None
    questions = _question_series(df, question_col)
    if questions is not None and "run_metadata_json" in df.columns:
        metadata_json = _single_value(df, "run_metadata_json")
        if metadata_json is not None:
            try:
                metadata = json.loads(metadata_json)
            except Exception:
                return None
            return _run_id_from_questions(questions, metadata)
        if _unique_count(df, "run_metadata_json") > 1:
            return None
    if questions is not None:
        return _run_id_from_questions(questions)
    return None
