from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import config as app_config


DEFAULT_BUDGET_SECONDS = int(getattr(app_config, "EVAL_TIME_BUDGET_SECONDS", 10800))


@dataclass(frozen=True)
class EvalRuntime:
    profile: str
    budget_seconds: int
    judge_enabled: bool
    judge_model_id: str | None
    judge_batch_size: int
    judge_max_new_tokens: int | None
    judge_timeout_seconds: int
    bertscore_model_type: str
    bertscore_batch_size: int
    checkpoint_every_rows: int
    max_context_docs: int
    generation_max_new_tokens: int
    generation_min_new_tokens: int

    def metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "budget_seconds": self.budget_seconds,
            "judge_enabled": self.judge_enabled,
            "judge_model_id": self.judge_model_id or "",
            "judge_batch_size": self.judge_batch_size,
            "judge_max_new_tokens": self.judge_max_new_tokens or 0,
            "judge_timeout_seconds": self.judge_timeout_seconds,
            "bertscore_model_type": self.bertscore_model_type,
            "bertscore_batch_size": self.bertscore_batch_size,
            "generation_max_new_tokens": self.generation_max_new_tokens,
            "generation_min_new_tokens": self.generation_min_new_tokens,
            "generation_model_id": str(getattr(app_config, "INFERENCE_MODEL_ID", app_config.MODEL_ID)),
            "generation_revision": str(getattr(app_config, "INFERENCE_REVISION", app_config.REVISION)),
            "use_finetuned": bool(getattr(app_config, "USE_FINETUNED", False)),
            "finetuned_adapter_path": str(getattr(app_config, "FINETUNED_ADAPTER_PATH", "")),
            "max_context_docs": self.max_context_docs,
            "faithfulness_threshold": float(getattr(app_config, "FAITHFULNESS_THRESHOLD", 0.0)),
            "critic_sentence_support_threshold": float(
                getattr(app_config, "CRITIC_SENTENCE_SUPPORT_THRESHOLD", 0.0)
            ),
            "max_retries": int(getattr(app_config, "MAX_RETRIES", 0)),
        }


def _base_runtime(*, profile: str, budget_seconds: int, judge_enabled: bool) -> EvalRuntime:
    generation_max = int(getattr(app_config, "GENERATION_MAX_NEW_TOKENS", 64) or 64)
    generation_min = int(getattr(app_config, "GENERATION_MIN_NEW_TOKENS", 16) or 16)
    bertscore_model_type = str(getattr(app_config, "BERTSCORE_MODEL_TYPE", "microsoft/deberta-v3-base"))
    max_context_docs = max(3, int(getattr(app_config, "MAX_CONTEXT_DOCS", 3)))

    if profile == "fast":
        return EvalRuntime(
            profile="fast",
            budget_seconds=budget_seconds,
            judge_enabled=False,
            judge_model_id=None,
            judge_batch_size=0,
            judge_max_new_tokens=None,
            judge_timeout_seconds=0,
            bertscore_model_type=bertscore_model_type,
            bertscore_batch_size=8,
            checkpoint_every_rows=10,
            max_context_docs=max_context_docs,
            generation_max_new_tokens=min(48, generation_max),
            generation_min_new_tokens=min(12, generation_min),
        )

    if profile == "t4-tight":
        return EvalRuntime(
            profile="t4-tight",
            budget_seconds=budget_seconds,
            judge_enabled=judge_enabled,
            judge_model_id=str(getattr(app_config, "EVAL_JUDGE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")),
            judge_batch_size=2,
            judge_max_new_tokens=max(64, int(getattr(app_config, "EVAL_JUDGE_MAX_NEW_TOKENS", 128)) - 32),
            judge_timeout_seconds=max(120, int(getattr(app_config, "EVAL_JUDGE_TIMEOUT_SECONDS", 180)) - 60),
            bertscore_model_type=bertscore_model_type,
            bertscore_batch_size=4,
            checkpoint_every_rows=3,
            max_context_docs=max_context_docs,
            generation_max_new_tokens=min(32, generation_max),
            generation_min_new_tokens=min(8, generation_min),
        )

    if profile == "t4-safe":
        return EvalRuntime(
            profile="t4-safe",
            budget_seconds=budget_seconds,
            judge_enabled=judge_enabled,
            judge_model_id=str(getattr(app_config, "EVAL_JUDGE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")),
            judge_batch_size=int(getattr(app_config, "EVAL_JUDGE_BATCH_SIZE", 4)),
            judge_max_new_tokens=int(getattr(app_config, "EVAL_JUDGE_MAX_NEW_TOKENS", 128)),
            judge_timeout_seconds=int(getattr(app_config, "EVAL_JUDGE_TIMEOUT_SECONDS", 180)),
            bertscore_model_type=bertscore_model_type,
            bertscore_batch_size=8,
            checkpoint_every_rows=5,
            max_context_docs=max_context_docs,
            generation_max_new_tokens=min(48, generation_max),
            generation_min_new_tokens=min(12, generation_min),
        )

    raise ValueError(f"Unknown evaluation profile: {profile}")


def resolve_eval_runtime(
    *,
    profile: str | None = None,
    budget_seconds: int | None = None,
    judge_requested: bool = False,
) -> EvalRuntime:
    resolved_budget = int(budget_seconds or DEFAULT_BUDGET_SECONDS)
    requested_profile = str(profile or "auto").strip().lower()

    if requested_profile == "auto":
        if judge_requested:
            requested_profile = "t4-safe" if resolved_budget >= DEFAULT_BUDGET_SECONDS else "t4-tight"
        else:
            requested_profile = "fast"

    return _base_runtime(
        profile=requested_profile,
        budget_seconds=resolved_budget,
        judge_enabled=judge_requested and requested_profile != "fast",
    )
