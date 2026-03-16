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
INSUFFICIENT_EVIDENCE = "The available evidence does not directly address this question."


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


def _clean_answer_text(text: str) -> str:
    """Remove common artifact sections and keep a compact final answer."""
    cleaned = text.replace(DISCLAIMER.strip(), "").strip()

    lines: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line:
            lines.append("")
            continue
        if lower.startswith("medical disclaimer:"):
            continue
        if lower.startswith("sources:"):
            continue
        if lower.startswith("**references**"):
            continue
        if line == "---":
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _limit_sentences(text: str, max_sentences: int = 3) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        kept.append(piece)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept) if kept else text.strip()


def _get_confidence_label(confidence_level: float) -> str:
    """Map confidence score (0.0-1.0) to human-readable label."""
    if confidence_level >= 0.9:
        return "Strongly Supported by Evidence"
    elif confidence_level >= 0.7:
        return "Well-Supported by Evidence"
    elif confidence_level >= 0.5:
        return "Partially Supported; Context Limitations Noted"
    elif confidence_level > 0.0:
        return "Weakly Supported; Treat as Provisional"
    return None


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
    clean_answer = _clean_answer_text(answer)
    clean_answer = _limit_sentences(clean_answer, max_sentences=3)
    _, triggered = safety_filter(clean_answer)

    add_inline_citations = clean_answer != INSUFFICIENT_EVIDENCE
    if add_inline_citations and ref_lines and not INLINE_CITATION_RE.search(clean_answer):
        clean_answer = clean_answer.rstrip() + " " + " ".join(f"[{citation}]" for citation in citations)

    # Add confidence label if present
    confidence_level = state.get("confidence_level", 0.0)
    confidence_label = _get_confidence_label(confidence_level)
    if confidence_label:
        clean_answer = clean_answer.rstrip() + f"\n[Evidence Confidence: {confidence_label}]"

    return {
        **state,
        "final_answer": clean_answer,
        "citations": citations,
        "safety_filter_triggered": triggered,
        "disclaimer": DISCLAIMER if triggered else "",
    }
