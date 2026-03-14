from __future__ import annotations

import re

from config import config as app_config
from src.graph.state import AgentState


_DRUG_TOKEN_RE = re.compile(r"\b([A-Z][a-z]{3,})\b")
_NON_DRUG_TOKENS = {
    "according",
    "package",
    "insert",
    "clinical",
    "trials",
    "what",
    "which",
    "first",
    "day",
    "days",
    "schedule",
}


def _extract_drug_name(query: str) -> str:
    # Heuristic fallback: first title-cased token that is not a generic word.
    matches = _DRUG_TOKEN_RE.findall(query)
    for token in matches:
        if token.lower() not in _NON_DRUG_TOKENS:
            return token
    return "the drug"


def drug_lookup(state: AgentState) -> AgentState:
    query = state.get("query", "")
    drug = _extract_drug_name(query)
    max_subqueries = int(getattr(app_config, "DRUG_LOOKUP_MAX_SUBQUERIES", 3))

    candidates = [
        query,
        f"{drug} dosage and administration titration schedule",
        f"{drug} day 1 day 2 day 3 day 4 day 5 day 6 maintenance dose",
        f"{drug} FDA label dosing table gastrointestinal side effects",
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for sq in candidates:
        key = sq.strip().lower()
        if key and key not in seen:
            deduped.append(sq)
            seen.add(key)

    return {**state, "sub_queries": deduped[:max_subqueries]}
