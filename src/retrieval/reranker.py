from __future__ import annotations

from dataclasses import dataclass

import torch
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

from config import config as app_config


def _default_device() -> str:
    configured_device = getattr(app_config, "RERANK_DEVICE", "cpu")
    if configured_device != "auto":
        return configured_device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FallbackReranker:
    """Offline-safe fallback when cross-encoder cannot be initialized."""

    top_n: int

    def postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        ranked = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
        return ranked[: self.top_n]


def build_reranker(
    top_n: int = getattr(app_config, "RERANK_TOP_N", 2),
    device: str | None = None,
) -> SentenceTransformerRerank | FallbackReranker:
    resolved_device = device or _default_device()
    try:
        return SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=top_n,
            device=resolved_device,
        )
    except Exception as exc:
        print(
            "Warning: failed to initialize cross-encoder reranker; "
            f"falling back to score-only reranking. Details: {exc}"
        )
        return FallbackReranker(top_n=top_n)
