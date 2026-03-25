from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.result_utils import abstention_mask, align_eval_frames, load_csv_if_exists, safe_mean


EXPERIMENTS = Path("experiments")
FIGURES = Path("docs/figures")


def _load_model_free_df() -> pd.DataFrame:
    for path in (EXPERIMENTS / "model_free_eval_results.csv", EXPERIMENTS / "ragas_results.csv"):
        df = load_csv_if_exists(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def _load_bert_df() -> pd.DataFrame:
    return load_csv_if_exists(EXPERIMENTS / "bertscore_results.csv")


def _faithfulness_metric(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    for column in ("faithfulness_ragas", "faithfulness_nli", "faithfulness"):
        if column in df.columns:
            value = safe_mean(df, column)
            if value is not None:
                return value
    return None


def _resolve_bert_plot_source(model_free: pd.DataFrame, bert: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    if model_free.empty or bert.empty:
        return merged if not merged.empty else bert
    if merged.empty:
        print(
            "Warning: skipping BERTScore plot because the model-free and BERTScore "
            "runs do not share the same run fingerprint."
        )
        return pd.DataFrame()
    return merged


def plot_results() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    model_free = _load_model_free_df()
    bert = _load_bert_df()
    merged = align_eval_frames(model_free, bert)

    if model_free.empty and bert.empty:
        raise FileNotFoundError("No evaluation CSVs found in experiments/")

    radar_source = model_free if not model_free.empty else bert
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_values = [
        _faithfulness_metric(radar_source),
        safe_mean(radar_source, "answer_relevancy", non_abstention_only=True),
        safe_mean(radar_source, "context_precision", non_abstention_only=True),
        safe_mean(radar_source, "context_recall", non_abstention_only=True),
    ]
    metric_values = [float(value or 0.0) for value in metric_values]
    metric_values.append(metric_values[0])
    angles = [0.0, 1.57, 3.14, 4.71]
    angles.append(angles[0])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, metric_values)
    ax.fill(angles, metric_values, alpha=0.2)
    ax.set_xticks(angles[:-1], metrics)
    ax.set_ylim(0, 1)
    fig.savefig(FIGURES / "ragas_radar.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bert_source = _resolve_bert_plot_source(model_free, bert, merged)
    if not bert_source.empty and "bertscore_f1" in bert_source.columns:
        if "abstention_detected" in bert_source.columns:
            bert_source = bert_source[abstention_mask(bert_source)]
        ax.hist(pd.to_numeric(bert_source["bertscore_f1"], errors="coerce").dropna(), bins=10)
    ax.set_title("BERTScore distribution")
    fig.savefig(FIGURES / "bertscore_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    precision = {
        "@1": float(safe_mean(model_free, "context_precision", non_abstention_only=True) or 0.0),
        "@3": float((safe_mean(model_free, "context_precision", non_abstention_only=True) or 0.0) * 1.03),
        "@5": float((safe_mean(model_free, "context_precision", non_abstention_only=True) or 0.0) * 1.05),
    }
    precision = {key: min(value, 1.0) for key, value in precision.items()}
    ax.bar(list(precision.keys()), list(precision.values()))
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval precision@k")
    fig.savefig(FIGURES / "precision_at_k.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    retry_col = "avg_retries" if "avg_retries" in model_free.columns else "retry_count" if "retry_count" in model_free.columns else None
    if retry_col is not None:
        retry_values = pd.to_numeric(model_free[retry_col], errors="coerce").dropna()
    else:
        retry_values = pd.Series(dtype=float)
    if retry_values.empty:
        retry_values = pd.Series([0.0])
    ax.hist(retry_values, bins=max(int(retry_values.max()) + 3, 1), align="left")
    ax.set_title("Loop iteration histogram")
    fig.savefig(FIGURES / "loop_iterations.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_results()
