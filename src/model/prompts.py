from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a medical information assistant. "
    "Using ONLY the provided context passages, write a concise direct answer to the question. "
    "Every factual claim must be directly supported by at least one provided passage. "
    "Do not add background medical knowledge, mechanisms, diagnoses, treatments, or next steps unless they are explicitly stated in the context. "
    "If the context supports only part of the answer, answer only that supported part and explicitly say what the context does not establish. "
    "When the question asks for a mechanism, diagnosis, or definitive management, do not infer across similar conditions or drugs. "
    "Cite sources inline using their numeric reference number, e.g. [1] or [2]. "
    "If the context does not contain enough information to answer the question, "
    "say exactly: 'The available evidence does not directly address this question.' "
    "Do NOT copy or repeat raw context text verbatim."
)

_MAX_CHUNK_CHARS = 400


def build_generation_prompt(
    query: str,
    context_blocks: list[tuple[str, str]],
    critic_feedback: str = "",
) -> str:
    """context_blocks is a list of (chunk_id, text); passages are numbered [1], [2], ..."""
    context = "\n\n".join(
        f"[{idx}] {text.strip()[:_MAX_CHUNK_CHARS]}"
        for idx, (_, text) in enumerate(context_blocks, start=1)
    )
    retry_block = ""
    if critic_feedback.strip():
        retry_block = (
            "Previous draft problems to avoid:\n"
            f"{critic_feedback.strip()}\n\n"
            "Revise the answer so that every sentence is directly supported by the context.\n\n"
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {query.strip()}\n\n"
        f"{retry_block}"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
