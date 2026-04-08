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

If the selected model requires Hugging Face authentication, set `HF_TOKEN` before running the notebook or CLI flow:

```bash
# Linux/macOS
export HF_TOKEN="<your_token>"

# Windows PowerShell
$env:HF_TOKEN="<your_token>"
```

For notebook-based setup, open [`notebooks/00_quick_start.ipynb`](./notebooks/00_quick_start.ipynb). The notebook is Kaggle-friendly and now:

- detects the repo root instead of nesting clones
- installs dependencies with `uv`
- skips interactive token prompts when `HF_TOKEN` is unset
- runs the full-pipeline evaluation and finetuned ablation as separate stages
- regenerates `experiments/all_results.csv` before plotting

## Pipeline

```bash
uv run python src/data/download.py
uv run python src/data/preprocess.py
uv run python src/data/build_indices.py
uv run python -m src.evaluation.build_test_set

# Quick smoke test
uv run python -m src.evaluation.run_eval --n_samples 10 --profile fast --skip-bertscore

# Full T4-budgeted evaluation with structured judge metrics
uv run python -m src.evaluation.run_eval --profile t4-safe --budget-seconds 10800 --with-ragas-judge

# Finetuned comparison only (ablation D)
uv run python -m src.evaluation.ablations --only D --profile t4-tight --budget-seconds 10800 --with-ragas-judge

# Refresh merged summary and figures
uv run python -m src.evaluation.plot_results

# App runtime
uv run streamlit run src/app.py
```

## Tests

```bash
uv run python -m unittest discover -s tests
```

## Configuration

Inference defaults in [`config/config.py`](./config/config.py) are intentionally lightweight so the project can run on a single-GPU notebook setup.

| Setting | Value | Purpose |
|---------|-------|---------|
<<<<<<< HEAD
| `FAITHFULNESS_THRESHOLD` | `0.40` | Critic validation threshold for retry/abstain decisions |
| `CRITIC_SENTENCE_SUPPORT_THRESHOLD` | `0.65` | Per-sentence NLI support floor |
| `MAX_RETRIES` | `1` | Maximum retry loops after critic feedback |
| `RETRIEVAL_SIMILARITY_TOP_K` | `8` | Dense retriever candidate pool size |
| `RERANK_TOP_N` | `3` | Cross-encoder reranked candidates kept before later filters |
| `MAX_CONTEXT_DOCS` | `2` | Hard cap on context chunks sent to generation |
| `ANSWER_TIMEOUT_SECONDS` | `45` | Hard wall-clock timeout |
| `INFERENCE_MODEL_ID` | `BioMistral/BioMistral-7B` | Default medical-domain base model |
| `MODEL_CACHE_DIR` | `models/biomistral-7b` | Local snapshot cache used after first download |
| `LOAD_IN_4BIT` | `True` | 4-bit NF4 quantization |
=======
| `FAITHFULNESS_THRESHOLD` | 0.40 | Critic validation threshold |
| `CRITIC_SENTENCE_SUPPORT_THRESHOLD` | 0.65 | Per-sentence NLI floor |
| `MAX_RETRIES` | 1 | Allows one retry after critic failure |
| `LOW_EVIDENCE_SCORE_FLOOR` | disabled | Retrieval confidence gate |
| `ANSWER_TIMEOUT_SECONDS` | 45 | Hard wall-clock timeout |
| `INFERENCE_MODEL_ID` | BioMistral/BioMistral-7B | Medical domain LLM |
| `MODEL_CACHE_DIR` | models/biomistral-7b | Local snapshot cache used after first download |
| `LOAD_IN_4BIT` | True | 4-bit NF4 quantization |
>>>>>>> 86fba21 (feat: update .gitignore and enhance README with configuration details; modify generator logic and improve prompt instructions)

Single-GPU runtime presets:

- Use `--profile fast` for the lightest smoke tests.
- Use `--profile t4-tight --with-ragas-judge` for the leanest finetuned judge run on a T4.
- Use `--profile t4-safe --budget-seconds 10800 --with-ragas-judge` for the full 3-hour T4-budgeted evaluation.
- `python -m src.evaluation.plot_results` refreshes `experiments/all_results.csv` before generating figures.

Override `INFERENCE_MODEL_ID` and `INFERENCE_REVISION` to test alternative models.

## QLoRA Fine-tuning + Evaluation

Run these commands from the workspace root.

```bash
# Train QLoRA adapter
uv run python -m src.finetune.qlora_train --max_steps 96

# Evaluate finetuned pipeline only (ablation D)
uv run python -m src.evaluation.ablations --only D --profile t4-tight --budget-seconds 10800 --with-ragas-judge

# Merge all evaluation outputs into experiments/all_results.csv
uv run python -m src.evaluation.merge_results
```

Quick test run:

```bash
uv run python -m src.evaluation.ablations --only D --n_samples 10
```

Training/evaluation data split:

- `src/data/download.py` writes a deterministic local split into `data/raw/train.jsonl` and `data/raw/eval.jsonl`
- `src/data/preprocess.py` builds the retrieval/training corpus from the local train split
- `src/evaluation/build_test_set.py` builds the evaluation questions from `data/raw/eval.jsonl`

That means the finetune path and the evaluation path come from the same source dataset family, but not the same local split.

## Safety-First Architecture

The system prioritizes evidence-based answers over speculation through multiple validation layers.

### Evidence Gates
<<<<<<< HEAD

- Retrieval score floors can stop generation if evidence is too weak.
- Query mode detection handles calculations, dosing schedules, and binary questions.
- Dosing evidence checks require explicit titration schedules in retrieved evidence.
=======
- Retrieval score floor can stop generation if the top reranked document score is too low, but this gate is disabled by default via `LOW_EVIDENCE_SCORE_FLOOR = -999`.
- Query mode detection classifies calculations, dosing schedules, binary questions, and default questions; only calculation and dosing have dedicated graph branches.
- Dosing-style safety is enforced primarily through prompt instructions that tell the model to abstain when the retrieved context lacks an explicit schedule.
>>>>>>> 86fba21 (feat: update .gitignore and enhance README with configuration details; modify generator logic and improve prompt instructions)

### NLI Faithfulness Validation

- Sentence-level support checks whether each sentence is backed by retrieved context.
- Entailment checking uses a cross-encoder NLI model to validate claims.
<<<<<<< HEAD
- Retry loops are enabled but intentionally capped for notebook-friendly runs.
=======
- The critic can trigger one retry by default before synthesis.
>>>>>>> 86fba21 (feat: update .gitignore and enhance README with configuration details; modify generator logic and improve prompt instructions)

### Confidence Thresholds

- `FAITHFULNESS_THRESHOLD` governs whether the answer is supported enough to proceed.
- `CRITIC_SENTENCE_SUPPORT_THRESHOLD` is the per-sentence requirement.
- If confidence is insufficient, the system returns an abstention-style answer instead of speculating.

## Evaluation Metrics

### Safety Metrics

Per-query tracking for safety validation:

- `abstention_detected`: System appropriately withheld an answer.
- `unsupported_claims_count`: Hallucination risk indicator from the critic.
- `citation_count`: Number of inline citations in the generated answer.
- `retry_count`: Number of retry attempts used for the query.
- `evidence_score`: Score of the top retrieved document.

Summary metrics:

- `abstention_precision`: Accuracy of abstention decisions.
- `unsupported_claims_rate`: Fraction of rows with at least one unsupported claim.
- `corrupted_output_rate`: Fraction of malformed or obviously corrupted generations.

### Traditional Metrics
<<<<<<< HEAD

- `faithfulness_nli`: NLI-based entailment score from the critic.
- `faithfulness_ragas`: Structured-judge faithfulness score when judge mode is enabled.
- `bertscore_f1`: Semantic similarity to the reference answer.
- `answer_relevancy`: Structured-judge answer relevance score.
- `context_precision` and `context_recall`: Retrieval quality metrics.
=======
- `faithfulness_nli`: NLI-based entailment.
- `bertscore_f1`: Semantic similarity to reference.
- `answer_relevancy`: Structured-judge score from the local evaluation pipeline.
- `context_precision` and `context_recall`: Structured-judge retrieval quality scores, with a heuristic fallback when the judge model is unavailable or unparseable.
>>>>>>> 86fba21 (feat: update .gitignore and enhance README with configuration details; modify generator logic and improve prompt instructions)
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
- Figures: `docs/figures/*.png`
