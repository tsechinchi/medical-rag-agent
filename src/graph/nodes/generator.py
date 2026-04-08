"""Generator node."""
from __future__ import annotations
from llama_index.core import Settings
from src.graph.state import AgentState
from src.model.prompts import build_generation_prompt
from src.utils.answer_cleaning import INSUFFICIENT_EVIDENCE, is_corrupted_output

def _docs_to_context_blocks(docs):
    blocks = []
    for node in docs:
        chunk_id = (
            getattr(node, "node_id", None)
            or node.metadata.get("chunk_id")
            or "unknown"
        )
        blocks.append((chunk_id, node.get_content()))
    return blocks

def generator(state: AgentState) -> AgentState:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", "")

    if not docs:
        return {
            **state,
            "draft_answer": "The available evidence does not directly address this question.",
            "critic_feedback": "",
            "confidence_level": 0.0,
        }

    context_blocks = _docs_to_context_blocks(docs)
    prompt = build_generation_prompt(
        query=query,
        context_blocks=context_blocks,
        critic_feedback=critic_feedback,
        is_partial=bool(critic_feedback.strip()),
    )

    llm = Settings.llm
    response = llm.complete(prompt)
    draft_answer = response.text.strip()
    if is_corrupted_output(draft_answer):
        draft_answer = INSUFFICIENT_EVIDENCE

    return {
        **state,
        "draft_answer": draft_answer,
        "confidence_level": 0.9,
        "critic_feedback": "",
    }
