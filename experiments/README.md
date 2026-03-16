# Evaluation Runs

This folder stores outputs and logs produced by the evaluation pipeline.

## Run from project root

All commands below should be run from:

```bash
cd /teamspace/studios/this_studio
```

## Quick sanity run (3 samples)

```bash
uv run python -m src.evaluation.run_eval --n_samples 3
```

## Full evaluation (all test samples)

```bash
uv run python -m src.evaluation.run_eval
```

## GPU-forced run (recommended)

```bash
env BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
uv run python -m src.evaluation.run_eval
```

Note: by default, this runs in fast mode (RAGAS LLM-judge metrics disabled).
Use `--with-ragas-judge` only when you explicitly want LLM-judge metrics.

## Useful variants

Run only model-free (skip BERTScore):

```bash
uv run python -m src.evaluation.run_eval --n_samples 3 --skip-bertscore
```

Run only BERTScore (skip model-free):

```bash
uv run python -m src.evaluation.run_eval --n_samples 10 --skip-model-free
```

Set a fixed seed:

```bash
uv run python -m src.evaluation.run_eval --n_samples 10 --seed 42
```

## Background run with log file

Fast mode (recommended for runtime stability):

```bash
nohup env BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
uv run python -m src.evaluation.run_eval --n_samples 3 > experiments/run_eval_full.log 2>&1 &
echo $! > experiments/run_eval_full.pid
```

With RAGAS LLM-judge enabled (slower):

```bash
nohup env BERTSCORE_DEVICE=cuda CRITIC_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
uv run python -m src.evaluation.run_eval --n_samples 3 --with-ragas-judge > experiments/run_eval_full.log 2>&1 &
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

- `experiments/model_free_eval_results.csv`
- `experiments/bertscore_results.csv`
- `experiments/ragas_results.csv`
- `experiments/all_results.csv`

Common logs:

- `experiments/run_eval_3sample_ragas.log`
- `experiments/run_eval_all_full.log`
- `experiments/run_eval_all_monitor.log`
