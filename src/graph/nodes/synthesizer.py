from __future__ import annotations

import re
from collections import OrderedDict

from src.graph.state import AgentState

_MAX_REF_CHARS = 90


DISCLAIMER = (
    "Medical disclaimer: This output is informational only and must not be used "
    "as a substitute for licensed clinical judgment.\n\n"
)
SAFETY_RE = re.compile(r"\b(dosage|prescription|diagnosis|treat(?:ment|ing)?|medication)\b", re.IGNORECASE)
INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")


def safety_filter(text: str) -> tuple[str, bool]:
    if SAFETY_RE.search(text):
        return DISCLAIMER + text, True
    return text, False


def _ref_label(text: str) -> str:
    """Truncate a reference title to _MAX_REF_CHARS with ellipsis."""
    text = text.strip()
    if len(text) <= _MAX_REF_CHARS:
        return text
    return text[:_MAX_REF_CHARS].rstrip() + "…"


def synthesizer(state: AgentState) -> AgentState:
    docs = state.get("retrieved_docs", [])

    # Build numbered reference list deduped by pubmed_id.
    seen: dict[str, int] = {}  # pubmed_id -> ref number
    ref_lines: list[str] = []
    chunk_ref_nums: list[int] = []
    for node in docs:
        meta = node.metadata
        pid = str(meta.get("pubmed_id") or meta.get("chunk_id") or "")
        if pid not in seen:
            ref_num = len(ref_lines) + 1
            seen[pid] = ref_num
            question = str(meta.get("question") or meta.get("chunk_id") or pid)
            label = _ref_label(question)
            ref_lines.append(f"[{ref_num}] {label} (PubMed: {pid})")
        chunk_ref_nums.append(seen[pid])

    citations = list(OrderedDict.fromkeys(str(n) for n in chunk_ref_nums))

    answer = state.get("draft_answer", "")
    # Strip a disclaimer the model may have echoed verbatim from the prompt.
    clean_answer = answer.replace(DISCLAIMER.strip(), "").strip()
    _, triggered = safety_filter(clean_answer)

    if ref_lines and not INLINE_CITATION_RE.search(clean_answer):
        clean_answer = clean_answer.rstrip() + " " + " ".join(f"[{citation}]" for citation in citations)

    if ref_lines:
        clean_answer = clean_answer.rstrip() + "\n\n---\n**References**\n" + "\n".join(ref_lines)

    return {
        **state,
        "final_answer": clean_answer,
        "citations": citations,
        "safety_filter_triggered": triggered,
        "disclaimer": DISCLAIMER if triggered else "",
    }
