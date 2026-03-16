from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    query: str
    sub_queries: list[str]
    retrieved_docs: list[Any]
    draft_answer: str
    faithfulness_score: float
    unsupported_claims_count: int
    critic_feedback: str
    retry_count: int
    final_answer: str
    citations: list[str]
    safety_filter_triggered: bool
    disclaimer: str
    confidence_level: float  # Confidence score (0.0-1.0) of the answer
