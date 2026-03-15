"""Generator node — format retrieved docs as context, call BioMistral, store draft answer."""

from __future__ import annotations

import re

from config import config as app_config
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore

from src.graph.state import AgentState
from src.model.prompts import build_generation_prompt, classify_query_mode


_DOSING_EVIDENCE_RE = re.compile(
    r"\b(titrat(?:e|ion)|day\s*1|day\s*2|maintenance\s*dose|dosage\s+and\s+administration|package\s+insert|fda\s+label)\b",
    re.IGNORECASE,
)


def _docs_to_context_blocks(
    docs: list[NodeWithScore],
) -> list[tuple[str, str]]:
    """Convert retrieved nodes to (chunk_id, content) pairs."""
    blocks: list[tuple[str, str]] = []
    for node in docs:
        chunk_id = (
            getattr(node, "node_id", None)
            or node.metadata.get("chunk_id")
            or "unknown_chunk"
        )
        blocks.append((chunk_id, node.get_content()))
    return blocks


def _has_dosing_evidence(docs: list[NodeWithScore]) -> bool:
    for node in docs:
        text = node.get_content() or ""
        if _DOSING_EVIDENCE_RE.search(text):
            return True
        meta = node.metadata or {}
        question = str(meta.get("question") or "")
        if _DOSING_EVIDENCE_RE.search(question):
            return True
    return False


def generator(state: AgentState) -> AgentState:
    """Build a grounded prompt from retrieved docs and generate a draft answer."""
    query: str = state["query"]
    docs: list[NodeWithScore] = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", "")
    query_mode = classify_query_mode(query)

    if not docs:
        return {
            **state,
            "draft_answer": "The available evidence does not directly address this question.",
            "critic_feedback": "",
        }

    top_score = float(docs[0].score or 0.0)
    low_evidence_floor = float(getattr(app_config, "LOW_EVIDENCE_SCORE_FLOOR", 0.0))
    if top_score < low_evidence_floor:
        return {
            **state,
            "draft_answer": "The available evidence does not directly address this question.",
            "critic_feedback": "",
        }

    if query_mode == "dosing":
        if not _has_dosing_evidence(docs):
            return {
                **state,
                "draft_answer": "The available evidence does not directly address the exact dosing schedule.",
                "critic_feedback": "",
            }

    context_blocks = _docs_to_context_blocks(docs)
    prompt = build_generation_prompt(
        query=query,
        context_blocks=context_blocks,
        critic_feedback=critic_feedback,
    )

    llm = Settings.llm
    response = llm.complete(prompt)

    return {**state, "draft_answer": response.text}
