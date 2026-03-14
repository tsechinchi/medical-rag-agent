from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training

from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from config.config import FINETUNED_ADAPTER_PATH, SEED
from src.model.loader import load_model_and_tokenizer
from src.utils.seed import set_seed

RAW_PATH = _ROOT / "data/raw/pubmed_qa_train.jsonl"
PROCESSED_PATH = _ROOT / "data/processed/pubmed_qa_train.jsonl"
OUTPUT_DIR = _ROOT / "data/qlora_checkpoints"
FINAL_ADAPTER_DIR = _ROOT / FINETUNED_ADAPTER_PATH


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_sample(sample: dict[str, object]) -> str:
    question = _as_text(sample.get("question"))
    context = _as_text(sample.get("text")) or "N/A"
    answer = _as_text(sample.get("final_decision")) or _as_text(sample.get("long_answer")) or "N/A"
    return (
        "### Question\n"
        f"{question}\n\n"
        "### Context\n"
        f"{context}\n\n"
        "### Answer\n"
        f"{answer}"
    )


def load_training_dataset(limit: int | None = None) -> Dataset:
    data_path = PROCESSED_PATH if PROCESSED_PATH.exists() else RAW_PATH
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.map(
        lambda sample: {"text": format_sample(sample)},
        remove_columns=dataset.column_names,
    )


def train(limit: int | None = None, max_steps: int = -1) -> None:
    set_seed(SEED)

    loaded = load_model_and_tokenizer(finetuned=False)
    tokenizer = loaded.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = loaded.model
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    train_dataset = load_training_dataset(limit=limit)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_length=512,
        bf16=True,
        report_to="none",
        num_train_epochs=1,
        max_steps=max_steps,
        logging_steps=10,
        save_strategy="epoch",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()

    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trained_model = trainer.model
    if trained_model is None:
        raise RuntimeError("Training completed without a model instance to save.")
    cast(Any, trained_model).save_pretrained(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    print(f"Saved final adapter to {FINAL_ADAPTER_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        metavar="N",
        help="Override training steps (e.g. 1 for a quick adapter smoke run).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA not detected. QLoRA training is expected to run on GPU.")

    train(limit=args.limit, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
