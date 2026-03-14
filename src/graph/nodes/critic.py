from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.graph.state import AgentState


NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


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


def critic(state: AgentState) -> AgentState:
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_docs", [])
    if not draft or not docs:
        return {**state, "faithfulness_score": 0.0}

    scores = [_entailment_score(node.get_content(), draft) for node in docs]
    mean_score = sum(scores) / len(scores)
    return {**state, "faithfulness_score": mean_score}
