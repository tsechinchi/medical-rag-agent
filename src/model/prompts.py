from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a medical information assistant. "
    "Using ONLY the provided context passages, write a concise direct answer to the question. "
    "Cite sources inline using their numeric reference number, e.g. [1] or [2]. "
    "If the context does not contain enough information to answer the question, "
    "say exactly: 'The available evidence does not directly address this question.' "
    "Do NOT copy or repeat raw context text verbatim."
)

_MAX_CHUNK_CHARS = 400


def build_generation_prompt(query: str, context_blocks: list[tuple[str, str]]) -> str:
    """context_blocks is a list of (chunk_id, text); passages are numbered [1], [2], ..."""
    context = "\n\n".join(
        f"[{idx}] {text.strip()[:_MAX_CHUNK_CHARS]}"
        for idx, (_, text) in enumerate(context_blocks, start=1)
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {query.strip()}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
