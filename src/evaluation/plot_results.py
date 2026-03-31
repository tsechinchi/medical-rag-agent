from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.merge_results import merge_results
from src.evaluation.result_utils import load_csv_if_exists


EXPERIMENTS = Path("experiments")
FIGURES = Path("docs/figures")


def _load_model_free_df() -> pd.DataFrame:
    for path in (EXPERIMENTS / "model_free_eval_results.csv", EXPERIMENTS / "ragas_results.csv"):
        df = load_csv_if_exists(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def _summary_metric_frame(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if summary.empty or "variant" not in summary.columns:
        return pd.DataFrame()

    available = [metric for metric in metrics if metric in summary.columns]
    if not available:
        return pd.DataFrame()

    frame = summary[["variant", *available]].copy()
    for metric in available:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=available, how="all")
    return frame


def _plot_quality_metrics(summary: pd.DataFrame) -> None:
    metrics = ["faithfulness_ragas", "answer_relevancy", "context_precision", "context_recall"]
    frame = _summary_metric_frame(summary, metrics)
    if frame.empty:
        print("Warning: skipping quality metrics plot because all_results.csv has no usable summary metrics.")
        return

    frame = frame.set_index("variant")
    ax = frame.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Quality Metrics by Variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "ragas_radar.png", bbox_inches="tight")
    plt.close()


def _plot_context_precision(summary: pd.DataFrame) -> None:
    frame = _summary_metric_frame(summary, ["context_precision"])
    if frame.empty:
        print("Warning: skipping context precision plot because all_results.csv has no context_precision values.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["variant"], frame["context_precision"])
    ax.set_title("Context Precision by Variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Context Precision")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES / "precision_at_k.png", bbox_inches="tight")
    plt.close(fig)


def _plot_bertscore_distribution(summary: pd.DataFrame) -> None:
    frame = _summary_metric_frame(summary, ["bertscore_f1_mean", "bertscore_f1_all_mean"])
    if frame.empty:
        print("Warning: skipping BERTScore plot because all_results.csv has no usable BERTScore values.")
        return

    frame["bertscore_f1_display"] = frame["bertscore_f1_mean"].combine_first(frame["bertscore_f1_all_mean"])
    frame = frame.dropna(subset=["bertscore_f1_display"])
    if frame.empty:
        print("Warning: skipping BERTScore plot because all_results.csv has no usable BERTScore values.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["variant"], frame["bertscore_f1_display"], color="#4C72B0")
    ax.set_title("Average BERTScore F1 by Variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("BERTScore F1")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES / "bertscore_distribution.png", bbox_inches="tight")
    plt.close(fig)


def _plot_loop_iterations(model_free: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    frame = _summary_metric_frame(summary, ["avg_retries"])
    if not frame.empty:
        frame = frame.sort_values("variant")
        bars = ax.bar(frame["variant"], frame["avg_retries"], color="#4C72B0")
        for bar, value in zip(bars, frame["avg_retries"].tolist()):
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0.0)),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
        ax.set_title("Average Retries by Variant")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Average Retries")
        ax.set_ylim(0, max(0.1, float(frame["avg_retries"].max()) + 0.1))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=20, ha="right")
    else:
        retry_col = "avg_retries" if "avg_retries" in model_free.columns else "retry_count" if "retry_count" in model_free.columns else None
        if retry_col is not None:
            retry_values = pd.to_numeric(model_free[retry_col], errors="coerce").dropna()
            if not retry_values.empty:
                buckets = {
                    "0": int((retry_values == 0).sum()),
                    "1": int((retry_values == 1).sum()),
                    "2+": int((retry_values >= 2).sum()),
                }
                bars = ax.bar(buckets.keys(), buckets.values(), color="#4C72B0")
                for bar, value in zip(bars, buckets.values()):
                    ax.annotate(
                        str(value),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        xytext=(0, 3),
                        textcoords="offset points",
                    )
                ax.set_ylabel("Count")
                ax.set_xlabel("Retries")
                ax.set_title("Retry Count Distribution")
                ax.set_ylim(0, max(1, max(buckets.values()) + 1))
                ax.grid(axis="y", linestyle="--", alpha=0.3)
            else:
                ax.set_title("Average Retries by Variant")
        else:
            ax.set_title("Average Retries by Variant")
    fig.savefig(FIGURES / "loop_iterations.png", bbox_inches="tight")
    plt.close(fig)


def plot_results() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    model_free = _load_model_free_df()
    summary = merge_results()

    if model_free.empty and summary.empty:
        raise FileNotFoundError("No evaluation CSVs found in experiments/")

    _plot_quality_metrics(summary)
    _plot_context_precision(summary)
    _plot_bertscore_distribution(summary)
    _plot_loop_iterations(model_free, summary)


if __name__ == "__main__":
    plot_results()
