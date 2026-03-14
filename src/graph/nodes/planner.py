from __future__ import annotations

from config import config as app_config
from src.graph.state import AgentState


def _fallback_decompose(query: str) -> list[str]:
    tokens = query.strip().rstrip("?")
    return [
        tokens,
        f"What evidence in the corpus supports the answer to: {tokens}?",
        f"What limitations or caveats are reported for: {tokens}?",
    ]


def planner(state: AgentState) -> AgentState:
    query = state["query"]
    sub_queries = _fallback_decompose(query)[:getattr(app_config, "PLANNER_MAX_SUBQUERIES", 1)]
    return {**state, "sub_queries": sub_queries}
