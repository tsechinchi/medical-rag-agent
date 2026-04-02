# Checkpoint 8 Adapter

This directory contains an intermediate PEFT adapter snapshot from local QLoRA training.

- Base model: `BioMistral/BioMistral-7B`
- Format: LoRA adapter weights plus adapter config
- Intended use: resume training, inspect an early checkpoint, or compare against `final/`

This checkpoint is not a standalone model. Load it together with the base model through PEFT.
