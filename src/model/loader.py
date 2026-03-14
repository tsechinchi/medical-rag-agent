from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import time
from typing import cast

import torch
from huggingface_hub.errors import RevisionNotFoundError
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import config as app_config


os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128",
)


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def _resolve_device_map() -> dict[str, int] | str:
    if torch.cuda.is_available():
        return {"": 0}
    return "cpu"


def _load_tokenizer_with_revision_fallback(
    model_id: str,
    revision: str,
) -> PreTrainedTokenizerBase:
    try:
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
            ),
        )
    except (RevisionNotFoundError, ValueError):
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
        )


def _load_model_with_revision_fallback(
    model_id: str,
    revision: str,
    bnb_config: BitsAndBytesConfig,
) -> PreTrainedModel:
    load_kwargs = {
        "quantization_config": bnb_config,
        "device_map": _resolve_device_map(),
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    try:
        return cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                **load_kwargs,
            ),
        )
    except (RevisionNotFoundError, ValueError):
        # Revision may not exist on Hub; load from local cache to avoid the
        # "Unrecognized model" ValueError that newer transformers raises when
        # it can't fetch the config.json at the given revision.
        return cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=True,
                **load_kwargs,
            ),
        )


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # Allow bitsandbytes to offload fp32 layers to CPU when GPU VRAM is
        # insufficient to hold the full quantised model.  Without this flag
        # device_map="auto" raises a ValueError whenever any module is mapped
        # to CPU or disk.
        llm_int8_enable_fp32_cpu_offload=True,
    )


def load_model_and_tokenizer(
    model_id: str = getattr(app_config, "INFERENCE_MODEL_ID", app_config.MODEL_ID),
    revision: str = getattr(app_config, "INFERENCE_REVISION", app_config.REVISION),
    finetuned: bool = getattr(app_config, "USE_FINETUNED", False),
) -> LoadedModel:
    return _load_model_and_tokenizer_cached(model_id, revision, finetuned)


@lru_cache(maxsize=4)
def _load_model_and_tokenizer_cached(
    model_id: str = getattr(app_config, "INFERENCE_MODEL_ID", app_config.MODEL_ID),
    revision: str = getattr(app_config, "INFERENCE_REVISION", app_config.REVISION),
    finetuned: bool = getattr(app_config, "USE_FINETUNED", False),
) -> LoadedModel:
    start_time = time.perf_counter()
    bnb_config = build_bnb_config()

    if finetuned and model_id != app_config.MODEL_ID:
        model_id = app_config.MODEL_ID
        revision = app_config.REVISION

    tokenizer = _load_tokenizer_with_revision_fallback(model_id, revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_with_revision_fallback(model_id, revision, bnb_config)

    if finetuned:
        adapter_path = Path(app_config.FINETUNED_ADAPTER_PATH)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned adapter not found at {adapter_path}. "
                "Train the adapter first or set USE_FINETUNED=False."
            )
        model = cast(
            PreTrainedModel,
            PeftModel.from_pretrained(model, str(adapter_path)),
        )

    elapsed = time.perf_counter() - start_time
    timeout_s = getattr(app_config, "MODEL_LOAD_TIMEOUT_SECONDS", 300)
    if elapsed > timeout_s:
        raise TimeoutError(
            f"Model load exceeded budget: {elapsed:.1f}s > {timeout_s}s. "
            "Use a smaller INFERENCE_MODEL_ID or prewarm cache."
        )

    model.eval()

    return LoadedModel(model=model, tokenizer=tokenizer)
