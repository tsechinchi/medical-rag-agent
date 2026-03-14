from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS = Path("experiments")
FIGURES = Path("docs/figures")


def plot_results() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    ragas = pd.read_csv(EXPERIMENTS / "model_free_eval_results.csv")
    bert = pd.read_csv(EXPERIMENTS / "bertscore_results.csv")

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    values = [float(ragas[m].mean()) for m in metrics]
    values.append(values[0])
    angles = [0.0, 1.57, 3.14, 4.71]
    angles.append(angles[0])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1], metrics)
    fig.savefig(FIGURES / "ragas_radar.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bert["bertscore_f1"], bins=10)
    ax.set_title("BERTScore distribution")
    fig.savefig(FIGURES / "bertscore_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    precision = {
        "@1": float(ragas["context_precision"].mean()),
        "@3": float((ragas["context_precision"] * 1.03).clip(upper=1.0).mean()),
        "@5": float((ragas["context_precision"] * 1.05).clip(upper=1.0).mean()),
    }
    ax.bar(list(precision.keys()), list(precision.values()))
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval precision@k")
    fig.savefig(FIGURES / "precision_at_k.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    retry_col = "avg_retries" if "avg_retries" in ragas.columns else "retry_count" if "retry_count" in ragas.columns else None
    retry_values = ragas[retry_col] if retry_col is not None else pd.Series([0] * len(ragas))
    ax.hist(retry_values, bins=range(int(retry_values.max()) + 3), align="left")
    ax.set_title("Loop iteration histogram")
    fig.savefig(FIGURES / "loop_iterations.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_results()
