# QLoRA Checkpoints

This directory stores local PEFT adapter artifacts produced by:

```bash
uv run python -m src.finetune.qlora_train
```

Contents:

- `checkpoint-*`: intermediate trainer checkpoints saved during training
- `final/`: the adapter directory the repo uses by default via `FINETUNED_ADAPTER_PATH`

These files are local training artifacts, not published standalone models. They must be loaded together with the base model `BioMistral/BioMistral-7B`.

Typical usage:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "BioMistral/BioMistral-7B"
adapter_path = "data/qlora_checkpoints/final"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
```

If you only need the current adapter for inference or ablation D, `final/` is the directory that matters.
