from __future__ import annotations

import torch
from llama_index.core.postprocessor import SentenceTransformerRerank

from config import config as app_config


def _default_device() -> str:
    configured_device = getattr(app_config, "RERANK_DEVICE", "cpu")
    if configured_device != "auto":
        return configured_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_reranker(
    top_n: int = getattr(app_config, "RERANK_TOP_N", 2),
    device: str | None = None,
) -> SentenceTransformerRerank:
    resolved_device = device or _default_device()
    return SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=top_n,
        device=resolved_device,
    )
