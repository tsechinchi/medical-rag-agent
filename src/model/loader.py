from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import time
from typing import cast

import torch
from huggingface_hub import snapshot_download
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


def _use_local_files_only() -> bool:
    return str(os.getenv("TRANSFORMERS_OFFLINE", "")).strip().lower() in {"1", "true", "yes"}


def _model_cache_path() -> Path:
    return Path(getattr(app_config, "MODEL_CACHE_DIR", "models/biomistral-7b"))


def _has_complete_model_snapshot(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    has_config = (path / "config.json").exists()
    has_tokenizer = (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and has_tokenizer and has_weights


def _model_source_path(model_id: str) -> str:
    cache_path = _model_cache_path()
    if _has_complete_model_snapshot(cache_path):
        return str(cache_path)
    return model_id


def _download_model_snapshot(model_id: str, revision: str) -> Path:
    cache_path = _model_cache_path()
    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(cache_path),
            local_dir_use_symlinks=False,
        )
    except RevisionNotFoundError:
        fallback_revision = "main"
        if revision != fallback_revision:
            print(
                f"Warning: revision '{revision}' was not found for {model_id}. "
                f"Falling back to '{fallback_revision}'."
            )
            snapshot_download(
                repo_id=model_id,
                revision=fallback_revision,
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
            )
        else:
            raise
    return cache_path


def _load_tokenizer_with_revision_fallback(
    model_id: str,
    revision: str,
) -> PreTrainedTokenizerBase:
    local_files_only = _use_local_files_only()
    source = _model_source_path(model_id)
    try:
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                source,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=True,
            ),
        )
    except (RevisionNotFoundError, ValueError, OSError, TypeError):
        if local_files_only:
            raise
        _download_model_snapshot(model_id, revision)
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(str(_model_cache_path()), local_files_only=True, trust_remote_code=True),
        )


def _load_model_with_revision_fallback(
    model_id: str,
    revision: str,
    bnb_config: BitsAndBytesConfig,
) -> PreTrainedModel:
    local_files_only = _use_local_files_only()
    source = _model_source_path(model_id)
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
                source,
                revision=revision,
                local_files_only=local_files_only,
                **load_kwargs,
            ),
        )
    except (RevisionNotFoundError, ValueError, OSError, TypeError):
        if local_files_only:
            raise
        # Revision may not exist on Hub; load from local cache to avoid the
        # "Unrecognized model" ValueError that newer transformers raises when
        # it can't fetch the config.json at the given revision.
        _download_model_snapshot(model_id, revision)
        return cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                str(_model_cache_path()),
                local_files_only=True,
                **load_kwargs,
            ),
        )


def _is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )


def _latest_local_checkpoint_adapter(base_adapter_path: Path) -> Path | None:
    checkpoints_root = base_adapter_path.parent
    if not checkpoints_root.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in checkpoints_root.glob("checkpoint-*"):
        if not child.is_dir():
            continue
        suffix = child.name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        if _is_adapter_dir(child):
            candidates.append((int(suffix), child))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _download_adapter_snapshot(target_dir: Path) -> Path | None:
    repo_id = str(getattr(app_config, "FINETUNED_ADAPTER_REPO_ID", "") or "").strip()
    if not repo_id:
        return None

    revision = str(getattr(app_config, "FINETUNED_ADAPTER_REVISION", "main") or "main")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ],
    )
    if _is_adapter_dir(target_dir):
        return target_dir
    return None


def _resolve_adapter_path() -> Path:
    adapter_path = Path(app_config.FINETUNED_ADAPTER_PATH)
    if _is_adapter_dir(adapter_path):
        return adapter_path

    if bool(getattr(app_config, "AUTO_RESUME_LATEST_CHECKPOINT_ADAPTER", False)):
        latest_checkpoint = _latest_local_checkpoint_adapter(adapter_path)
        if latest_checkpoint is not None:
            print(f"Using latest local checkpoint adapter: {latest_checkpoint}")
            return latest_checkpoint

    if bool(getattr(app_config, "AUTO_DOWNLOAD_FINETUNED_ADAPTER", False)):
        try:
            downloaded = _download_adapter_snapshot(adapter_path)
            if downloaded is not None:
                print(f"Downloaded finetuned adapter into: {downloaded}")
                return downloaded
        except Exception as exc:
            print(f"Warning: adapter auto-download failed: {exc}")

    raise FileNotFoundError(
        f"Fine-tuned adapter not found at {adapter_path}. "
        "Set FINETUNED_ADAPTER_REPO_ID for auto-download, "
        "or disable USE_FINETUNED."
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

    cache_path = _model_cache_path()
    cache_ready = _has_complete_model_snapshot(cache_path)
    if not cache_ready and not _use_local_files_only():
        print(f"Downloading model snapshot into {cache_path} ...")
        _download_model_snapshot(model_id, revision)
        cache_ready = True
    elif cache_path.exists() and not cache_ready and _use_local_files_only():
        raise FileNotFoundError(
            f"Model cache at {cache_path} is incomplete. "
            "Download the snapshot once, or delete the partial directory and retry."
        )

    tokenizer = _load_tokenizer_with_revision_fallback(model_id, revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_with_revision_fallback(model_id, revision, bnb_config)

    if finetuned:
        adapter_path = _resolve_adapter_path()
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
