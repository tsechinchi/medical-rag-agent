# Medical RAG Agent

## Setup

Prerequisite: `uv` installed.

```bash
git clone <repo-url>
cd medical-rag-agent
uv python install 3.11
uv python pin 3.11
uv venv
uv sync --extra cpu
# or on a GPU machine
uv sync --extra gpu
```

## Pipeline

```bash
uv run python src/data/download.py
uv run python src/data/preprocess.py
uv run python src/data/build_indices.py
uv run python -m src.evaluation.build_test_set
uv run python -m src.evaluation.run_eval
uv run python -m src.evaluation.plot_results
uv run streamlit run src/app.py
```

## QLoRA Fine-tuning + Evaluation

Run these commands from the workspace root.

```bash
# Train QLoRA adapter (full run)
uv run python -m src.finetune.qlora_train --max_steps 96

# Evaluate finetuned pipeline only (ablation D)
uv run python -m src.evaluation.ablations --only D

# Merge all ablation CSVs into experiments/all_results.csv
uv run python -m src.evaluation.merge_results
```

Quick test run (smaller, faster):

```bash
uv run python -m src.evaluation.ablations --only D --n_samples 10
```

### Monitor a Long-running PID

Use the helper script to log process status in the same format as `experiments/trainer_monitor.log`.

```bash
# usage
./scripts/monitor_pid.sh <pid> <log_path> [interval_seconds]

# example: monitor current ablation process every 5 minutes
nohup ./scripts/monitor_pid.sh $(pgrep -fo "src.evaluation.ablations") experiments/ablation_full_monitor.log 300 >/tmp/ablation_full_monitor.out 2>&1 &
```

This log includes:

- monitor start line with timestamp and PID
- periodic `alive` lines with elapsed time / CPU / memory / GPU app memory
- final `exited` line

### Output Artifacts

- Finetuned adapter: `data/qlora_checkpoints/final/`
- Finetuned ablation CSV: `experiments/ablation_qlora.csv`
- Consolidated metrics: `experiments/all_results.csv`

## Runtime Profile

Inference defaults are tuned in `config/config.py` for a constrained T4-style GPU budget.

- `INFERENCE_MODEL_ID` defaults to `BioMistral/BioMistral-7B` and is loaded in 4-bit NF4.
- `RERANK_DEVICE` defaults to `cpu`; embedding device can be `auto` for faster retrieval where GPU headroom exists.
- Generation defaults are budgeted for notebook and app responsiveness via `GENERATION_MAX_NEW_TOKENS` and `ANSWER_TIMEOUT_SECONDS`.
- Hugging Face stack is pinned to `transformers>=4.45,<5` and `huggingface-hub<1` for BioMistral compatibility.

If you need a smaller temporary test model, override `INFERENCE_MODEL_ID` and `INFERENCE_REVISION` in `config/config.py` and rerun notebook setup cells.

## Open-source Components

This project depends on open-source building blocks for retrieval, orchestration, model loading, and evaluation.

- LlamaIndex for indexing, retrievers, reranking, and notebook integration.
- LangGraph for the planner → retriever → generator → critic → synthesizer control flow.
- BioMistral and Hugging Face Transformers for the medical-domain language model and inference pipeline.
- bitsandbytes and PEFT for 4-bit loading and QLoRA fine-tuning.
- RAGAS and BERTScore for evaluation.
- Streamlit for the lightweight demo interface.

## Original Code

The repo-specific contribution is the application logic that combines those components into a medical RAG workflow.

- Hybrid retrieval wiring and artifact caching.
- Graph retry logic driven by faithfulness scoring.
- CPU-based critic pass and safety disclaimer filter.
- Evaluation runners, ablations, plotting, and app presentation.