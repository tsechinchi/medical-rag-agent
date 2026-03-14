from __future__ import annotations

import re

from src.graph.state import AgentState


DISCLAIMER = (
    "Medical disclaimer: This output is informational only and must not be used "
    "as a substitute for licensed clinical judgment.\n\n"
)
SAFETY_RE = re.compile(r"\b(dosage|prescription|diagnosis|treat(?:ment|ing)?|medication)\b", re.IGNORECASE)


def safety_filter(text: str) -> tuple[str, bool]:
    if SAFETY_RE.search(text):
        return DISCLAIMER + text, True
    return text, False


def synthesizer(state: AgentState) -> AgentState:
    docs = state.get("retrieved_docs", [])
    citations = [
        getattr(node, "node_id", None) or node.metadata.get("chunk_id") or "unknown_chunk"
        for node in docs
    ]
    answer = state.get("draft_answer", "")
    filtered_answer, triggered = safety_filter(answer)
    if citations:
        filtered_answer = filtered_answer.rstrip() + "\n\nSources: " + ", ".join(f"[{cid}]" for cid in citations)
    return {
        **state,
        "final_answer": filtered_answer,
        "citations": citations,
        "safety_filter_triggered": triggered,
        "disclaimer": DISCLAIMER if triggered else "",
    }
