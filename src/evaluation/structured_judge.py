from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from config import config as app_config
from src.utils.answer_cleaning import clean_for_scoring, is_abstention


_JUDGE_SYSTEM = (
    "You are a JSON-only evaluation assistant. "
    "Return one valid JSON object and nothing else. "
    "Do not include markdown fences, prose, or comments."
)
_JSON_CODE_FENCE_RE = re.compile(r"```json|```", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class JudgeRow:
    question: str
    answer: str
    reference: str
    contexts: list[str]
    abstention_detected: bool = False


def _judge_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _token_set(text: str) -> set[str]:
    tokens = {token for token in _TOKEN_RE.findall(text.lower()) if len(token) >= 3}
    return {token for token in tokens if token not in _STOPWORDS}


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def _heuristic_scores(row: JudgeRow) -> dict[str, float]:
    answer = clean_for_scoring(row.answer)
    reference = clean_for_scoring(row.reference)
    context_text = " ".join(clean_for_scoring(ctx) for ctx in row.contexts if ctx).strip()

    if row.abstention_detected or is_abstention(answer):
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    answer_question = _overlap_ratio(answer, row.question)
    answer_reference = _overlap_ratio(answer, reference)
    context_question = _overlap_ratio(context_text, row.question)
    context_reference = _overlap_ratio(context_text, reference)
    answer_context = _overlap_ratio(answer, context_text)

    return {
        "faithfulness": round(max(0.0, min(1.0, 0.55 * answer_context + 0.45 * context_reference)), 4),
        "answer_relevancy": round(max(0.0, min(1.0, 0.6 * answer_question + 0.4 * answer_reference)), 4),
        "context_precision": round(max(0.0, min(1.0, 0.7 * context_question + 0.3 * context_reference)), 4),
        "context_recall": round(max(0.0, min(1.0, context_reference)), 4),
    }


def _normalize_scores(scores: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        value = scores.get(key, 0.0)
        try:
            normalized[key] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return normalized


def _extract_json(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return ""
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        pass

    for fence in ("```json", "```"):
        if fence in candidate:
            inner = candidate.split(fence, 1)[1].split("```", 1)[0].strip()
            try:
                json.loads(inner)
                return inner
            except Exception:
                candidate = inner
                break

    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = candidate.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for idx, ch in enumerate(candidate[start:], start=start):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    snippet = candidate[start : idx + 1]
                    try:
                        json.loads(snippet)
                        return snippet
                    except Exception:
                        break
    return ""


def _build_prompt(row: JudgeRow, max_context_docs: int) -> str:
    contexts = row.contexts[:max_context_docs]
    context_block = "\n\n".join(
        f"[{idx}] {clean_for_scoring(ctx)[:800]}" for idx, ctx in enumerate(contexts, start=1)
    )
    abstention_hint = ""
    if row.abstention_detected or is_abstention(row.answer):
        abstention_hint = (
            "The answer is an explicit abstention because the evidence was insufficient. "
            "In that case, score faithfulness as 1.0 and the other metrics as 0.0. "
        )

    return (
        f"{_JUDGE_SYSTEM}\n\n"
        "Score the answer against the question, reference answer, and retrieved context.\n"
        "Return a JSON object with exactly these numeric keys: "
        "faithfulness, answer_relevancy, context_precision, context_recall.\n"
        "Use values from 0.0 to 1.0. Higher is better.\n"
        f"{abstention_hint}"
        "Guidance:\n"
        "- faithfulness: whether the answer claims are supported by the retrieved context.\n"
        "- answer_relevancy: whether the answer directly addresses the question and matches the reference answer.\n"
        "- context_precision: whether the retrieved context is focused on the needed evidence.\n"
        "- context_recall: whether the retrieved context contains the key evidence needed by the reference answer.\n\n"
        f"Question: {row.question.strip()}\n\n"
        f"Answer: {clean_for_scoring(row.answer).strip()}\n\n"
        f"Reference answer: {clean_for_scoring(row.reference).strip()}\n\n"
        f"Retrieved contexts:\n{context_block or '[1]'}\n\n"
        "JSON:"
    )


def _load_model_components(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the structured judge model path.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=2)
def _cached_model_components(model_id: str):
    return _load_model_components(model_id)


class StructuredJudge:
    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        max_context_docs: int,
        timeout_seconds: int,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_context_docs = max_context_docs
        self.timeout_seconds = timeout_seconds
        self._use_model = False
        self._load_error: str | None = None
        self._tokenizer = None
        self._model = None

        try:
            self._tokenizer, self._model = _cached_model_components(model_id)
            self._use_model = True
        except Exception as exc:
            self._load_error = str(exc)
            self._use_model = False

    @property
    def is_model_backed(self) -> bool:
        return self._use_model and self._tokenizer is not None and self._model is not None

    def _generate_batch(self, prompts: list[str], batch_size: int) -> list[str]:
        if not self.is_model_backed:
            return []

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None
        assert model is not None

        outputs: list[str] = []
        effective_batch_size = max(1, int(batch_size))
        timeout_s = max(1, int(self.timeout_seconds))
        for start in range(0, len(prompts), effective_batch_size):
            batch = prompts[start : start + effective_batch_size]
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_time=float(timeout_s),
                )
            prompt_len = inputs["input_ids"].shape[1]
            outputs.extend(
                tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip()
                for sequence in generated
            )
        return outputs

    def score_rows(self, rows: list[JudgeRow], *, batch_size: int | None = None) -> list[dict[str, Any]]:
        if not rows:
            return []

        if not self.is_model_backed:
            fallback_rows: list[dict[str, Any]] = []
            for row in rows:
                scores = _heuristic_scores(row)
                fallback_rows.append(
                    {
                        **scores,
                        "judge_raw_output": json.dumps(scores, separators=(",", ":")),
                        "judge_used_fallback": True,
                    }
                )
            return fallback_rows

        prompts = [_build_prompt(row, self.max_context_docs) for row in rows]
        raw_outputs = self._generate_batch(prompts, batch_size or int(getattr(app_config, "EVAL_JUDGE_BATCH_SIZE", 4)))
        scored_rows: list[dict[str, Any]] = []
        for row, raw_output in zip(rows, raw_outputs):
            extracted = _extract_json(raw_output)
            parsed: dict[str, Any] = {}
            if extracted:
                try:
                    candidate = json.loads(extracted)
                    if isinstance(candidate, dict):
                        parsed = candidate
                except Exception:
                    parsed = {}
            if parsed:
                scores = _normalize_scores(parsed)
                used_fallback = False
            else:
                scores = _heuristic_scores(row)
                used_fallback = True
            scored_rows.append(
                {
                    **scores,
                    "judge_raw_output": raw_output,
                    "judge_used_fallback": used_fallback,
                }
            )
        return scored_rows

    def close(self) -> None:
        clear_judge_cache()


def build_structured_judge(
    *,
    model_id: str | None = None,
    max_new_tokens: int | None = None,
    max_context_docs: int | None = None,
    timeout_seconds: int | None = None,
) -> StructuredJudge:
    return StructuredJudge(
        model_id=model_id or str(getattr(app_config, "EVAL_JUDGE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")),
        max_new_tokens=max_new_tokens or int(getattr(app_config, "EVAL_JUDGE_MAX_NEW_TOKENS", 128)),
        max_context_docs=max_context_docs or int(getattr(app_config, "MAX_CONTEXT_DOCS", 3)),
        timeout_seconds=timeout_seconds or int(getattr(app_config, "EVAL_JUDGE_TIMEOUT_SECONDS", 180)),
    )


def clear_judge_cache() -> None:
    _cached_model_components.cache_clear()
