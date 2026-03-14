"""Generator node — format retrieved docs as context, call BioMistral, store draft answer."""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore

from src.graph.state import AgentState
from src.model.prompts import build_generation_prompt


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


def generator(state: AgentState) -> AgentState:
    """Build a grounded prompt from retrieved docs and generate a draft answer."""
    query: str = state["query"]
    docs: list[NodeWithScore] = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", "")

    context_blocks = _docs_to_context_blocks(docs)
    prompt = build_generation_prompt(
        query=query,
        context_blocks=context_blocks,
        critic_feedback=critic_feedback,
    )

    llm = Settings.llm
    response = llm.complete(prompt)

    return {**state, "draft_answer": response.text}
