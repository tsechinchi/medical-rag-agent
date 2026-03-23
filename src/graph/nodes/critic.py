from __future__ import annotations

import re
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import config as app_config
from src.graph.state import AgentState
from src.model.prompts import classify_query_mode
from src.utils.answer_cleaning import is_abstention


NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
def _resolve_critic_device() -> str:
    configured = str(getattr(app_config, "CRITIC_DEVICE", "auto")).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured in {"cuda", "cpu"}:
        if configured == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_nli_components():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.to(_resolve_critic_device())
    model.eval()
    return tokenizer, model


def _entailment_score(premise: str, hypothesis: str) -> float:
    tokenizer, model = _load_nli_components()
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    if probs.shape[0] >= 3:
        return float(probs[2].item())
    return float(probs.max().item())


def _split_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: list[str] = []
    for line in lines:
        if line.startswith("---") or line.lower().startswith("**references**"):
            break
        parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(line) if part.strip()]
        sentences.extend(parts or [line])
    return sentences


def _normalize_sentence_for_nli(sentence: str) -> str:
    cleaned = _INLINE_CITATION_RE.sub("", sentence)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _sentence_support(sentences: list[str], docs) -> tuple[list[float], list[str]]:
    scores: list[float] = []
    feedback: list[str] = []
    support_threshold = float(getattr(app_config, "CRITIC_SENTENCE_SUPPORT_THRESHOLD", 0.6))
    for sentence in sentences:
        normalized = _normalize_sentence_for_nli(sentence)
        if len(normalized) < 12:
            continue
        best_score = 0.0
        for node in docs:
            best_score = max(best_score, _entailment_score(node.get_content(), normalized))
        scores.append(best_score)
        if best_score < support_threshold:
            feedback.append(f"Unsupported or weakly supported claim: {normalized}")
    return scores, feedback


def _numeric_anchor_feedback(query: str, draft: str) -> list[str]:
    if classify_query_mode(query) != "calculation":
        return []

    feedback: list[str] = []
    query_numbers = list(dict.fromkeys(_NUMBER_RE.findall(query)))
    missing_numbers = [value for value in query_numbers if value not in draft]
    if missing_numbers:
        feedback.append(
            "The draft did not preserve all numeric inputs from the question: "
            + ", ".join(missing_numbers)
        )

    lowered_query = query.lower()
    lowered_draft = draft.lower()
    for token in ("male", "female"):
        if token in lowered_query and token not in lowered_draft:
            feedback.append(f"The draft did not preserve the patient sex from the question: {token}.")

    return feedback


def critic(state: AgentState) -> AgentState:
    query = state.get("query", "")
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_docs", [])

    # If the generator intentionally abstained due to low/no evidence, do not
    # force retries; this is a valid grounded response.
    if is_abstention(draft):
        return {
            **state,
            "faithfulness_score": 1.0,
            "confidence_level": 0.0,
            "critic_feedback": "",
            "unsupported_claims_count": 0,
        }

    if not draft or not docs:
        anchor_feedback = _numeric_anchor_feedback(query, draft)
        if classify_query_mode(query) == "calculation":
            feedback = "\n".join(anchor_feedback)
            return {
                **state,
                "faithfulness_score": 1.0 if not anchor_feedback else 0.0,
                "confidence_level": 0.0,
                "critic_feedback": feedback,
                "unsupported_claims_count": len(anchor_feedback),
            }
        feedback = "\n".join(anchor_feedback) if anchor_feedback else "No grounded answer could be verified."
        return {
            **state,
            "faithfulness_score": 0.0,
            "confidence_level": 0.0,
            "critic_feedback": feedback,
            "unsupported_claims_count": len(anchor_feedback),
        }

    doc_scores = [_entailment_score(node.get_content(), draft) for node in docs]
    doc_mean = sum(doc_scores) / len(doc_scores)

    sentences = _split_sentences(draft)
    sentence_scores, feedback_items = _sentence_support(sentences, docs)
    sentence_mean = sum(sentence_scores) / max(len(sentence_scores), 1)
    min_sentence_score = min(sentence_scores) if sentence_scores else 0.0
    anchor_feedback = _numeric_anchor_feedback(query, draft)
    feedback_items.extend(anchor_feedback)

    # Penalize mixed true/false answers: one unsupported sentence should block synthesis.
    faithfulness_score = min(doc_mean, sentence_mean, min_sentence_score)
    if anchor_feedback:
        faithfulness_score = min(faithfulness_score, 0.0)
    critic_feedback = "\n".join(feedback_items)

    return {
        **state,
        "faithfulness_score": faithfulness_score,
        "confidence_level": max(0.0, min(1.0, float(faithfulness_score))),
        "critic_feedback": critic_feedback,
        "unsupported_claims_count": len(feedback_items),
    }
