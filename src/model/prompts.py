from __future__ import annotations


SYSTEM_PROMPT = (
    "Answer using ONLY the provided context. Cite [chunk_id] inline. "
    "If unsupported, say so."
)


def build_generation_prompt(query: str, context_blocks: list[tuple[str, str]]) -> str:
    context = "\n\n".join(
        f"[{chunk_id}] {text.strip()}" for chunk_id, text in context_blocks
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
