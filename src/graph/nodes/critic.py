from __future__ import annotations

import re
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.graph.state import AgentState


NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _load_nli_components():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.to("cpu")
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


def _sentence_support(sentences: list[str], docs) -> tuple[list[float], list[str]]:
    scores: list[float] = []
    feedback: list[str] = []
    for sentence in sentences:
        best_score = 0.0
        for node in docs:
            best_score = max(best_score, _entailment_score(node.get_content(), sentence))
        scores.append(best_score)
        if best_score < 0.6:
            feedback.append(f"Unsupported or weakly supported claim: {sentence}")
    return scores, feedback


def critic(state: AgentState) -> AgentState:
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_docs", [])
    if not draft or not docs:
        return {**state, "faithfulness_score": 0.0, "critic_feedback": "No grounded answer could be verified."}

    doc_scores = [_entailment_score(node.get_content(), draft) for node in docs]
    doc_mean = sum(doc_scores) / len(doc_scores)

    sentences = _split_sentences(draft)
    sentence_scores, feedback_items = _sentence_support(sentences, docs)
    sentence_mean = sum(sentence_scores) / max(len(sentence_scores), 1)
    min_sentence_score = min(sentence_scores) if sentence_scores else 0.0

    # Penalize mixed true/false answers: one unsupported sentence should block synthesis.
    faithfulness_score = min(doc_mean, sentence_mean, min_sentence_score)
    critic_feedback = "\n".join(feedback_items)

    return {
        **state,
        "faithfulness_score": faithfulness_score,
        "critic_feedback": critic_feedback,
    }
