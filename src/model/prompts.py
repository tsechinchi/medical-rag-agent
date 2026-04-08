from __future__ import annotations

import re


SYSTEM_PROMPT = (
    "You are a medical information assistant. "
    "Using ONLY the provided context passages, write a concise direct answer to the question. "
    "Every factual claim must be directly supported by at least one provided passage. "
    "Do not add background medical knowledge, mechanisms, diagnoses, treatments, or next steps unless they are explicitly stated in the context. "
    "If the context supports only part of the answer, answer only that supported part and explicitly say what the context does not establish. "
    "When the question asks for a mechanism, diagnosis, or definitive management, do not infer across similar conditions or drugs. "
    "Cite sources inline using their numeric reference number, e.g. [1] or [2]. "
    "If no provided passage supports any part of the answer, "
    "say exactly: 'The available evidence does not directly address this question.' "
    "Do NOT copy or repeat raw context text verbatim. "
    "Keep the response concise (1-3 sentences). "
    "Do not include section headers, reference lists, or disclaimers in the final answer."
    "\n\n"
    "CRITICAL SAFETY ANCHOR: Every claim must be directly traceable to the provided context. "
    "If uncertain about supporting evidence, state the limitation explicitly: "
    "'The context suggests X, but does not clearly establish Y.' "
    "Prefer a partial evidence-backed answer over a full abstention when at least one claim is directly supported. "
    "Abstention is better than speculation in medical contexts."
)

_MAX_CHUNK_CHARS = 1200
_CALCULATION_QUERY_RE = re.compile(
    r"\b(calculate|calculation|compute|estimated|estimate|formula|cockcroft|gault|crcl|creatinine\s+clearance)\b",
    re.IGNORECASE,
)
_DOSING_QUERY_RE = re.compile(
    r"\b(titrat(?:e|ion)|dose|dosage|dosing|schedule|escalat(?:e|ion)|day\s*1|day\s*2|mg)\b",
    re.IGNORECASE,
)
_BINARY_QUERY_RE = re.compile(
    r"^\s*(does|do|did|is|are|was|were|can|could|should|would|will|has|have|had)\b",
    re.IGNORECASE,
)


def classify_query_mode(query: str) -> str:
    if _CALCULATION_QUERY_RE.search(query):
        return "calculation"
    if _DOSING_QUERY_RE.search(query):
        return "dosing"
    if _BINARY_QUERY_RE.search(query.strip()):
        return "binary"
    return "default"


def _mode_instructions(query: str, has_context: bool, is_partial: bool = False) -> str:
    mode = classify_query_mode(query)
    base_instruction = ""
    if is_partial:
        base_instruction = (
            "This answer is based on limited evidence (confidence level: PARTIALLY SUPPORTED). "
            "Be clear about what the context supports and what it does not: "
            "'The evidence suggests X, but the context does not establish Y.' "
            "Explicitly state context limitations. "
        )

    if mode == "calculation":
        return (
            f"{base_instruction}"
            "For this question, the patient-specific values written in the Question are authoritative input data. "
            "Copy them exactly as written and do not replace them with typical or normal values. "
            "First restate the extracted variables exactly, then show the formula substitution, then give the computed result. "
            "If a required variable is missing, state what is missing instead of guessing. "
            "If provided context includes downstream dosing or contraindication rules, use it for those claims; otherwise keep the answer limited to the calculation implied by the Question. "
        )
    if mode == "dosing":
        if has_context:
            return (
                f"{base_instruction}"
                "If the context contains a titration schedule, escalation sequence, or day-by-day dosing table, reproduce every step in order. "
                "Do not compress a multi-step schedule into a simplified summary. "
                "If the exact schedule is not present in the context, state any directly supported dosing details first and then say what schedule details the context does not establish. "
            )
        return (
            f"{base_instruction}"
            "This is a dosing-style question, but no supporting context is available. "
            "Do not invent a schedule or regimen. "
        )
    if mode == "binary":
        return (
            f"{base_instruction}"
            "This is a yes/no question. Start the answer with 'Yes,' or 'No,' when the context clearly supports one side. "
            "If the evidence is mixed, answer with the supported nuance instead of forcing a hard yes or no. "
            "Only use the exact abstention sentence when no provided passage supports either side. "
            "After the yes/no decision, give one concise evidence-backed sentence only. "
        )
    return base_instruction


def build_generation_prompt(
    query: str,
    context_blocks: list[tuple[str, str]],
    critic_feedback: str = "",
    is_partial: bool = False,
) -> str:
    """context_blocks is a list of (chunk_id, text); passages are numbered [1], [2], ..."""
    has_context = bool(context_blocks)
    context = "\n\n".join(
        f"[{idx}] {text.strip()[:_MAX_CHUNK_CHARS]}"
        for idx, (_, text) in enumerate(context_blocks, start=1)
    )
    mode_block = _mode_instructions(query, has_context, is_partial=is_partial)
    retry_block = ""
    if critic_feedback.strip():
        retry_block = (
            "Previous draft problems to avoid:\n"
            f"{critic_feedback.strip()}\n\n"
            "Revise the answer so that every sentence is directly supported by the context.\n\n"
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{mode_block}\n\n"
        f"Question: {query.strip()}\n\n"
        f"{retry_block}"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
