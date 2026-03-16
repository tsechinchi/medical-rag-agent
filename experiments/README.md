# Evaluation Runs

This folder stores outputs and logs produced by the evaluation pipeline.

## Run from project root

All commands below should be run from:

```bash
cd /teamspace/studios/this_studio
```

## Quick sanity run (3 samples)

```bash
python -m src.evaluation.run_eval --n_samples 3
```

## Full evaluation (all test samples)

```bash
python -m src.evaluation.run_eval
```

## GPU-forced run (recommended)

```bash
env BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
python -m src.evaluation.run_eval
```

Note: by default, this runs in fast mode (RAGAS LLM-judge metrics disabled).
Use `--with-ragas-judge` only when you explicitly want LLM-judge metrics.

## Recommended presets

Stable preset (best for long runs):

```bash
export CUDA_VISIBLE_DEVICES=0
export BERTSCORE_DEVICE=cuda
export CRITIC_DEVICE=cuda
export BERTSCORE_BATCH_SIZE=8
```

Aggressive preset (higher throughput, more OOM/timeout risk):

```bash
export CUDA_VISIBLE_DEVICES=0
export BERTSCORE_DEVICE=cuda
export CRITIC_DEVICE=cuda
export BERTSCORE_BATCH_SIZE=16
```

Example with stable preset + judge mode:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda BERTSCORE_BATCH_SIZE=8 \
python -m src.evaluation.run_eval --n_samples 200 --with-ragas-judge > experiments/run_eval_full.log 2>&1 &
echo $! > experiments/run_eval_full.pid
```

Example with aggressive preset + judge mode:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda BERTSCORE_BATCH_SIZE=16 \
python -m src.evaluation.run_eval --n_samples 3 --with-ragas-judge > experiments/run_eval_full.log 2>&1 &
echo $! > experiments/run_eval_full.pid
```

## Useful variants

Run only model-free (skip BERTScore):

```bash
python -m src.evaluation.run_eval --n_samples 3 --skip-bertscore
```

Run only BERTScore (skip model-free):

```bash
python -m src.evaluation.run_eval --n_samples 10 --skip-model-free
```

Set a fixed seed:

```bash
python -m src.evaluation.run_eval --n_samples 10 --seed 42
```

## Background run with log file

Fast mode (recommended for runtime stability):

```bash
nohup env BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
python -m src.evaluation.run_eval --n_samples 3 > experiments/run_eval_full.log 2>&1 &
echo $! > experiments/run_eval_full.pid
```

With RAGAS LLM-judge enabled (slower):

```bash
nohup env BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
python -m src.evaluation.run_eval --n_samples 3 --with-ragas-judge > experiments/run_eval_full.log 2>&1 &
echo $! > experiments/run_eval_full.pid
```

Check status:

```bash
ps -p "$(cat experiments/run_eval_full.pid)" -o pid,etime,%cpu,%mem,cmd
```

Tail log:

```bash
tail -f experiments/run_eval_full.log
```

## Output files

Primary outputs:

- `experiments/model_free_eval_results.csv` - Per-query detailed metrics including safety indicators
- `experiments/bertscore_results.csv` - BERTScore F1 semantic similarity scores
- `experiments/ragas_results.csv` - RAGAS LLM-judge metrics (faithfulness, answer relevancy, context precision/recall)
- `experiments/all_results.csv` - Merged summary by variant with aggregated metrics

Common logs:

- `experiments/run_eval_3sample_ragas.log`
- `experiments/run_eval_all_full.log`
- `experiments/run_eval_all_monitor.log`

## New Safety Metrics (v2.0+)

The evaluation system now tracks additional **safety metrics** to measure abstention precision and hallucination risk:

### Per-Query Columns (in model_free_eval_results.csv)

| Column | Type | Description |
|--------|------|-------------|
| `abstention_detected` | Binary (0/1) | 1 if system withheld answer due to insufficient evidence |
| `unsupported_claims_count` | Integer | Number of sentences rejected as unsupported by NLI critic |
| `citation_count` | Integer | Number of inline [N] citations in answer |
| `retry_count` | Integer | How many retry attempts were needed (max 3) |
| `confidence_level` | Float (0.0-1.0) | System's confidence in the answer |
| `evidence_score` | Float | Bge rerank score of top retrieved document |

### Summary Metrics (in all_results.csv)

| Column | Type | Description |
|--------|------|-------------|
| `abstention_precision` | Float (0.0-1.0) | Fraction of abstentions that were appropriate (NLI=1.0) |
| `unsupported_claims_rate` | Float | Average unsupported claims per query |

**Interpretation:**
- **abstention_precision ≥ 0.95**: System's abstention decisions are highly accurate
- **unsupported_claims_rate ≈ 0.0**: System has low hallucination risk
- **Missing values**: RAGAS metrics (answer_relevancy, context_precision/recall) are NULL for abstention rows (not penalized for correct withholding)

## Confidence Levels in Answers

When confidence tracking is enabled, final answers include a confidence label:

```
[Evidence Confidence: Strongly Supported by Evidence]
[Evidence Confidence: Well-Supported by Evidence]
[Evidence Confidence: Partially Supported; Context Limitations Noted]
[Evidence Confidence: Weakly Supported; Treat as Provisional]
```

These reflect internal NLI faithfulness scores:
- ≥90%: Strongly Supported
- 70-90%: Well-Supported
- 50-70%: Partially Supported
- <50%: Weakly Supported
