from __future__ import annotations

from typing import Any

from config import config as app_config
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.llms.mock import MockLLM

import src.data.build_indices as build_indices_mod


def load_hybrid_retriever(
    similarity_top_k: int = getattr(app_config, "RETRIEVAL_SIMILARITY_TOP_K", 6),
) -> QueryFusionRetriever:
    index = build_indices_mod.load_index()
    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    bm25_retriever = build_indices_mod.load_bm25_retriever()
    bm25_retriever.similarity_top_k = getattr(app_config, "BM25_SIMILARITY_TOP_K", 3)
    fusion_mode: Any = "reciprocal_rerank"

    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        # Ensure retriever initialization is fully offline and does not depend
        # on Settings.llm (which may default to OpenAI in fresh environments).
        llm=MockLLM(max_tokens=1),
        mode=fusion_mode,
        similarity_top_k=similarity_top_k,
        num_queries=int(getattr(app_config, "RETRIEVAL_FUSION_NUM_QUERIES", 1)),
        use_async=False,
        verbose=False,
    )
