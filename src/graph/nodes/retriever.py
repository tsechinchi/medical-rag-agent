"""Retriever node — run hybrid retrieval + reranking per sub-query, merge & deduplicate."""

from __future__ import annotations

import re

from config import config as app_config
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.graph.state import AgentState
from src.retrieval.hybrid import load_hybrid_retriever
from src.retrieval.reranker import build_reranker

# Module-level singletons (lazy-loaded once)
_retriever = None
_reranker = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


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


def _keywords(text: str) -> set[str]:
    tokens = {tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= 3}
    return {tok for tok in tokens if tok not in _STOPWORDS}


def _overlap_ratio(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / max(1, len(query_terms))


def _is_domain_relevant(node: NodeWithScore, query_terms: set[str], min_overlap: float) -> bool:
    metadata = getattr(node, "metadata", {}) or {}
    meta_question = str(metadata.get("question") or "")
    content = node.get_content() if hasattr(node, "get_content") else ""
    content = str(content or "")

    question_overlap = _overlap_ratio(query_terms, _keywords(meta_question))
    content_overlap = _overlap_ratio(query_terms, _keywords(content))
    best_overlap = max(question_overlap, content_overlap)
    return best_overlap >= min_overlap


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

    # Early abstention gate for weak retrieval evidence.
    if merged:
        top_score = float(merged[0].score or 0.0)
        evidence_floor = float(getattr(app_config, "LOW_EVIDENCE_SCORE_FLOOR", 0.0))
        if top_score < evidence_floor:
            return {**state, "retrieved_docs": []}

    # Drop chunks whose rerank score is below the configured relevance gate.
    # This prevents off-topic corpus hits from being passed to the generator.
    min_score: float = getattr(app_config, "RERANK_MIN_SCORE", 0.0)
    filtered = [n for n in merged if (n.score or 0.0) >= min_score]

    # Additional lexical domain gate to suppress high-score but off-topic
    # candidates (for example, when generic biomedical wording overlaps).
    if bool(getattr(app_config, "ENABLE_DOMAIN_RELEVANCE_GATE", False)) and filtered:
        full_query = " ".join(sub_queries).strip()
        query_terms = _keywords(full_query)
        min_overlap = float(getattr(app_config, "DOMAIN_RELEVANCE_MIN_OVERLAP", 0.0))
        domain_filtered = [
            node
            for node in filtered
            if _is_domain_relevant(node, query_terms, min_overlap)
        ]
        if domain_filtered:
            filtered = domain_filtered

    # Keep only nodes near the top rerank score to reduce noisy context.
    margin: float = float(getattr(app_config, "RERANK_SCORE_MARGIN", 1.0))
    if filtered:
        best_score = float(filtered[0].score or 0.0)
        filtered = [n for n in filtered if float(n.score or 0.0) >= best_score - margin]

    # Safety floor: if score thresholds are too strict, backfill from merged
    # to preserve enough evidence for generation.
    min_docs: int = int(getattr(app_config, "MIN_CONTEXT_DOCS", 1))
    if len(filtered) < max(1, min_docs):
        used_ids = {
            (node.node_id if node.node_id is not None else str(id(node)))
            for node in filtered
        }
        full_query = " ".join(sub_queries).strip()
        query_terms = _keywords(full_query)
        use_domain_gate = bool(getattr(app_config, "ENABLE_DOMAIN_RELEVANCE_GATE", False))
        min_overlap = float(getattr(app_config, "DOMAIN_RELEVANCE_MIN_OVERLAP", 0.0))

        preferred_pool: list[NodeWithScore] = []
        fallback_pool: list[NodeWithScore] = []
        for node in merged:
            if use_domain_gate and not _is_domain_relevant(node, query_terms, min_overlap):
                fallback_pool.append(node)
            else:
                preferred_pool.append(node)

        for node in preferred_pool + fallback_pool:
            node_key = node.node_id if node.node_id is not None else str(id(node))
            if node_key in used_ids:
                continue
            filtered.append(node)
            used_ids.add(node_key)
            if len(filtered) >= max(1, min_docs):
                break

    # Fail-safe: avoid empty context by keeping the best available node.
    if not filtered and merged:
        filtered = [merged[0]]

    max_docs: int = int(getattr(app_config, "MAX_CONTEXT_DOCS", getattr(app_config, "RERANK_TOP_N", 3)))
    filtered = filtered[:max(1, max_docs)]

    return {**state, "retrieved_docs": filtered}
