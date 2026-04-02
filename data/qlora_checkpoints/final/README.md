# Final QLoRA Adapter

This directory contains the adapter that the repo loads when the finetuned path is enabled.

- Default path source: `config/config.py -> FINETUNED_ADAPTER_PATH`
- Used by: `src/model/loader.py` when `finetuned=True`
- Typical evaluation entrypoint: `python -m src.evaluation.ablations --only D`

Load it with the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "BioMistral/BioMistral-7B"
adapter_path = "data/qlora_checkpoints/final"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
```

The adapter depends on the base model weights and tokenizer. It is not intended to be used on its own.
