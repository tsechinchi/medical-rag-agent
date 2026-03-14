"""Retriever node — run hybrid retrieval + reranking per sub-query, merge & deduplicate."""

from __future__ import annotations

from config import config as app_config
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.graph.state import AgentState
from src.retrieval.hybrid import load_hybrid_retriever
from src.retrieval.reranker import build_reranker

# Module-level singletons (lazy-loaded once)
_retriever = None
_reranker = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = load_hybrid_retriever(
            similarity_top_k=getattr(app_config, "RETRIEVAL_SIMILARITY_TOP_K", 6)
        )
    return _retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = build_reranker(top_n=getattr(app_config, "RERANK_TOP_N", 2))
    return _reranker


def _deduplicate(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Keep the highest-scoring occurrence of each node_id."""
    seen: dict[str, NodeWithScore] = {}
    for node in nodes:
        nid = node.node_id or id(node)
        if nid not in seen or (node.score or 0.0) > (seen[nid].score or 0.0):
            seen[nid] = node
    return sorted(seen.values(), key=lambda n: n.score or 0.0, reverse=True)


def retriever(state: AgentState) -> AgentState:
    """Retrieve and rerank documents for each sub-query, then merge."""
    sub_queries: list[str] = state.get("sub_queries", [state["query"]])
    hybrid = _get_retriever()
    reranker = _get_reranker()

    all_nodes: list[NodeWithScore] = []
    for sq in sub_queries:
        qb = QueryBundle(query_str=sq)
        candidates = hybrid.retrieve(qb)
        reranked = reranker.postprocess_nodes(candidates, query_bundle=qb)
        all_nodes.extend(reranked)

    merged = _deduplicate(all_nodes)

    # Drop chunks whose rerank score is below the configured relevance gate.
    # This prevents off-topic corpus hits from being passed to the generator.
    min_score: float = getattr(app_config, "RERANK_MIN_SCORE", 0.0)
    filtered = [n for n in merged if (n.score or 0.0) >= min_score]

    return {**state, "retrieved_docs": filtered}
