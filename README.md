# Medical RAG Agent

Safety-first medical question answering system using retrieval-augmented generation with confidence-based answer synthesis.

## Quick Start

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

If the selected model requires Hugging Face authentication, set your token as an environment variable:

```bash
# Linux/macOS
export HUGGINGFACE_HUB_TOKEN="<your_token>"

# Windows PowerShell
$env:HUGGINGFACE_HUB_TOKEN="<your_token>"
```

For notebook-based setup, open [`notebooks/00_quick_start.ipynb`](./notebooks/00_quick_start.ipynb). It walks through cloning the repo, authenticating, syncing dependencies with `uv`, running a fast smoke test, and then stepping up to the budgeted evaluation flow.

## Pipeline

```bash
uv run python src/data/download.py
uv run python src/data/preprocess.py
uv run python src/data/build_indices.py
uv run python -m src.evaluation.build_test_set
uv run python -m src.evaluation.run_eval --n_samples 10 --profile fast --skip-bertscore # quick smoke test
uv run python -m src.evaluation.run_eval --profile auto --budget-seconds 10800 --with-ragas-judge
uv run python -m src.evaluation.plot_results
uv run streamlit run src/app.py # skip eval if not needed
```

## Tests
```bash
python3.11 -m unittest discover -s tests
```

## Configuration

Inference defaults in `config/config.py` are intentionally light so the project can run with minimal setup.

| Setting | Value | Purpose |
|---------|-------|---------|
| `FAITHFULNESS_THRESHOLD` | 0.40 | Critic validation threshold |
| `CRITIC_SENTENCE_SUPPORT_THRESHOLD` | 0.65 | Per-sentence NLI floor |
| `MAX_RETRIES` | 0 | Avoids second-pass generation cost |
| `LOW_EVIDENCE_SCORE_FLOOR` | disabled | Retrieval confidence gate |
| `ANSWER_TIMEOUT_SECONDS` | 45 | Hard wall-clock timeout |
| `INFERENCE_MODEL_ID` | BioMistral/BioMistral-7B | Medical domain LLM |
| `MODEL_CACHE_DIR` | models/biomistral-7b | Local snapshot cache used after first download |
| `LOAD_IN_4BIT` | True | 4-bit NF4 quantization |

Single-GPU runtime target:

- Use `--profile auto --budget-seconds 10800 --with-ragas-judge` for the full T4-budgeted eval run.
- Use `--profile fast --n_samples 10 --skip-bertscore` for the quickest smoke test.
- Use `--profile t4-tight --with-ragas-judge` when you want the leanest judge path on a single T4.
- The notebook starts with the fast smoke test first, then shows the longer full-eval command separately.
- The first full model load populates `models/biomistral-7b/`, and later runs reuse that local snapshot.

Override `INFERENCE_MODEL_ID` and `INFERENCE_REVISION` to test alternative models.

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

Quick test run:

```bash
uv run python -m src.evaluation.ablations --only D --n_samples 10
```

Fast local eval loop:

```bash
uv run python -m src.evaluation.run_eval --n_samples 10 --profile fast --skip-bertscore
```

## Safety-First Architecture

The system prioritizes evidence-based answers over speculation through multiple validation layers:

### Evidence Gates
- Retrieval score floor stops generation if top document score is too low.
- Query mode detection handles calculations, dosing schedules, and binary questions.
- Dosing evidence checks require explicit titration schedules in retrieved evidence.

### NLI Faithfulness Validation
- Sentence-level support checks whether each sentence is supported by retrieved context.
- Entailment checking uses a cross-encoder NLI model to validate claims.
- Retry loops are disabled in the default lightweight profile.

### Confidence Thresholds
- `FAITHFULNESS_THRESHOLD` governs whether the answer is supported enough to proceed.
- `CRITIC_SENTENCE_SUPPORT_THRESHOLD` is the per-sentence requirement.
- If confidence is insufficient, the system returns an abstention response.

## Evaluation Metrics

### Safety Metrics

Per-query tracking for safety validation:
- `abstention_detected`: System appropriately withheld answer.
- `unsupported_claims_count`: Hallucination risk indicator.
- `confidence_level`: Internal confidence score.
- `evidence_score`: Rerank score of top document.

Summary metrics:
- `abstention_precision`: Accuracy of abstention decisions.
- `unsupported_claims_rate`: Average unsupported claims per query.

### Traditional Metrics
- `faithfulness_nli`: NLI-based entailment.
- `bertscore_f1`: Semantic similarity to reference.
- `answer_relevancy`: RAGAS relevance.
- `context_precision` and `context_recall`: Retrieval quality metrics.
- `latency_per_query_s`: End-to-end response time.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_huggingface'`

If this happens even though `uv sync` completed:

```bash
pip install langchain-huggingface
uv sync
```

### Monitor a long-running PID

```bash
./scripts/monitor_pid.sh <pid> <log_path> [interval_seconds]
```

## Output Artifacts

- Finetuned adapter: `data/qlora_checkpoints/final/`
- Finetuned ablation CSV: `experiments/ablation_qlora.csv`
- Consolidated metrics: `experiments/all_results.csv`
