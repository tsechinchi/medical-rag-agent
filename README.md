# Medical RAG Agent

Safety-first medical question answering system using retrieval-augmented generation with confidence-based answer synthesis.

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

## Safety-First Architecture

The system prioritizes **evidence-based answers over speculation** through multiple validation layers:

### 1. Evidence Gates
- **Retrieval Score Floor** (0.10): Stops generation if top document score is too low
- **Query Mode Detection**: Special handling for calculations, dosing schedules, and binary questions
- **Dosing Evidence Check**: Requires explicit titration schedules in retrieved evidence

### 2. NLI Faithfulness Validation
- **Sentence-Level Support** (threshold: 0.65): Each sentence must be supported by retrieved context
- **Entailment Checking**: Uses cross-encoder NLI model to validate claims
- **Retry Loop** (max 2 attempts): On faithfulness failure, system retries with feedback

### 3. Confidence Thresholds
- **FAITHFULNESS_THRESHOLD**: 0.60 (aggregate answer must be ≥60% supported)
- **CRITIC_SENTENCE_SUPPORT_THRESHOLD**: 0.65 (per-sentence requirement)
- **Abstention Strategy**: "The available evidence does not directly address this question" when confidence is insufficient

### 4. Confidence Labels
Answers include transparent confidence markers:
- ≥90%: "Strongly Supported by Evidence"
- 70-90%: "Well-Supported by Evidence"
- 50-70%: "Partially Supported; Context Limitations Noted"
- <50%: "Weakly Supported; Treat as Provisional"

## Evaluation Metrics (v2.0+)

### Safety Metrics

Per-query tracking for safety validation:
- `abstention_detected`: System appropriately withheld answer (binary)
- `unsupported_claims_count`: Hallucination risk indicator (integer)
- `confidence_level`: Internal confidence score 0.0-1.0 (float)
- `evidence_score`: Bge rerank score of top document (float)

Summary metrics:
- `abstention_precision`: Accuracy of abstention decisions (target ≥0.95)
- `unsupported_claims_rate`: Average unsupported claims per query (target ≈0.0)

### Traditional Metrics
- **faithfulness_nli**: NLI-based entailment (0.0-1.0)
- **bertscore_f1**: Semantic similarity to reference (0.0-1.0)
- **answer_relevancy**: RAGAS relevance [masked for abstentions]
- **context_precision/recall**: Retrieval quality metrics
- **latency_per_query_s**: End-to-end response time

**Key insight**: RAGAS metrics show NULL (not 0.0) for abstention rows, preventing unfair penalization of correct withholding decisions.

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

## Configuration

Inference defaults tuned in `config/config.py` for constrained T4-style GPU (16 GB VRAM):

| Setting | Value | Purpose |
|---------|-------|---------|
| `FAITHFULNESS_THRESHOLD` | 0.60 | Critic validation threshold |
| `CRITIC_SENTENCE_SUPPORT_THRESHOLD` | 0.65 | Per-sentence NLI floor (0.7→0.65 v2.0) |
| `MAX_RETRIES` | 2 | Max synthesis retry attempts |
| `LOW_EVIDENCE_SCORE_FLOOR` | 0.10 | Retrieval confidence gate (0.2→0.10 v2.0) |
| `ANSWER_TIMEOUT_SECONDS` | 120 | Hard wall-clock timeout |
| `INFERENCE_MODEL_ID` | BioMistral/BioMistral-7B | Medical domain LLM |
| `LOAD_IN_4BIT` | True | 4-bit NF4 quantization |

Single-GPU runtime target (1x T4):

- Keep `--with-ragas-judge` disabled for standard runs.
- Use `BERTSCORE_BATCH_SIZE=8` and `--n_samples 200` max per run.
- Expected wall-clock remains within 2.5 hours in fast mode on one T4.

Override `INFERENCE_MODEL_ID` and `INFERENCE_REVISION` to test alternative models.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_huggingface'`

When running with `--with-ragas-judge`, if you see this error even though `uv sync` completed:

```bash
# Solution: Install directly with pip
pip install langchain-huggingface

# Then sync uv to update lock file
uv sync
```

This can happen when `uv sync` doesn't fully install all transitive dependencies. Using `pip install` directly ensures the package is available in your environment.

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
- Generation length defaults to the full remaining model context budget, while `ANSWER_TIMEOUT_SECONDS` still bounds runtime.
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