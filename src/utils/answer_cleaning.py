from __future__ import annotations

import re


INSUFFICIENT_EVIDENCE = "The available evidence does not directly address this question."
INSUFFICIENT_DOSING = "The available evidence does not directly address the exact dosing schedule."

_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_PARTIAL_PREFIX_RE = re.compile(
    r"^(?:\[(?:partially supported|weakly supported|strongly supported)\]\s*)+",
    re.IGNORECASE,
)
_EVIDENCE_LABEL_RE = re.compile(r"\[Evidence Confidence:[^\]]*\]", re.IGNORECASE)
_DISCLAIMER_RE = re.compile(
    r"^medical disclaimer:.*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOURCE_LINE_RE = re.compile(r"^sources:\s*", re.IGNORECASE)
_REFERENCE_HEADER_RE = re.compile(r"^\*\*references\*\*\s*$", re.IGNORECASE)
_METADATA_LEAK_RE = re.compile(
    r"('relations'\s*:|'meshes'\s*:|'keywords'\s*:|'labels'\s*:|'reasoning'\s*:|'evidence'\s*:)",
    re.IGNORECASE,
)
_DICTISH_RE = re.compile(r"\{\s*'[^']+'\s*:\s*", re.IGNORECASE)


def is_abstention(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered.startswith(INSUFFICIENT_EVIDENCE.lower()) or lowered.startswith(
        INSUFFICIENT_DOSING.lower()
    )


def is_corrupted_output(text: str) -> bool:
    candidate = (text or "").strip()
    if not candidate:
        return False
    if _METADATA_LEAK_RE.search(candidate):
        return True
    if _DICTISH_RE.search(candidate):
        return True
    if candidate.count("'") > 20 and candidate.count(":") > 4:
        return True
    return False


def _remove_noise_lines(text: str) -> str:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if _SOURCE_LINE_RE.match(line):
            continue
        if _REFERENCE_HEADER_RE.match(line):
            continue
        if line == "---":
            continue
        if line.lower().startswith("references:"):
            continue
        lines.append(line)
    return "\n".join(lines)


def limit_sentences(text: str, max_sentences: int = 10) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept: list[str] = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        kept.append(piece)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept) if kept else (text or "").strip()


def clean_answer_text(text: str, *, max_sentences: int | None = 3) -> str:
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    cleaned = _PARTIAL_PREFIX_RE.sub("", cleaned)
    cleaned = _DISCLAIMER_RE.sub("", cleaned)
    cleaned = _remove_noise_lines(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if max_sentences is not None:
        cleaned = limit_sentences(cleaned, max_sentences=max_sentences)
    return cleaned.strip()


def clean_for_scoring(text: str) -> str:
    cleaned = clean_answer_text(text, max_sentences=None)
    cleaned = _INLINE_CITATION_RE.sub("", cleaned)
    cleaned = _EVIDENCE_LABEL_RE.sub("", cleaned)
    cleaned = _PARTIAL_PREFIX_RE.sub("", cleaned)
    cleaned = _DISCLAIMER_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
