"""
Visualization utilities for EDA and model performance comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def _check_plotting_deps() -> None:
    if not HAS_PLOTTING:
        raise ImportError(
            "Visualization functions require 'matplotlib' and 'seaborn'. "
            "Install them with: pip install matplotlib seaborn"
        )


def plot_polarization_distribution(
    eng_counts: pd.Series,
    swa_counts: pd.Series,
    save_path: Optional[str] = "polarization_class_distribution.pdf",
) -> None:
    """Generate dual bar plots of binary polarization distribution for English and Swahili."""
    _check_plotting_deps()

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Polarization Class Distribution (Subtask 1)", fontsize=16, fontweight="bold")

    # English
    sns.barplot(x=eng_counts.index, y=eng_counts.values, ax=axes[0], palette=["skyblue", "salmon"])
    axes[0].set_title("English Training Data", fontsize=13)
    axes[0].set_xlabel("Polarization (0: Not Polarized, 1: Polarized)")
    axes[0].set_ylabel("Number of Samples")
    for i, count in enumerate(eng_counts.values):
        axes[0].text(i, count + 50, str(count), ha="center", va="bottom", fontsize=10)

    # Swahili
    sns.barplot(x=swa_counts.index, y=swa_counts.values, ax=axes[1], palette=["skyblue", "salmon"])
    axes[1].set_title("Swahili Training Data", fontsize=13)
    axes[1].set_xlabel("Polarization (0: Not Polarized, 1: Polarized)")
    axes[1].set_ylabel("Number of Samples")
    for i, count in enumerate(swa_counts.values):
        axes[1].text(i, count + 50, str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.close()


def plot_label_frequency(
    df: pd.DataFrame,
    label_cols: List[str],
    language: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot label frequency distribution for multi-label polarization types."""
    _check_plotting_deps()

    label_counts = df[label_cols].sum().sort_values(ascending=False)
    total_samples = len(df)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title(f"Label Frequency Distribution: {language} Training Data", fontsize=14, fontweight="bold")
    plt.xlabel("Polarization Category", fontsize=12)
    plt.ylabel("Number of Positive Annotations", fontsize=12)

    for i, count in enumerate(label_counts.values):
        pct = f"{count / total_samples * 100:.1f}%"
        ax.text(i, count + 15, f"{count}\n({pct})", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved label frequency plot to {save_path}")
    plt.close()


def plot_co_occurrence_heatmap(
    df: pd.DataFrame,
    label_cols: List[str],
    language: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot label co-occurrence heatmap for multi-label categories."""
    _check_plotting_deps()

    label_matrix = df[label_cols].values
    co_occurrence = np.dot(label_matrix.T, label_matrix)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        co_occurrence,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_cols,
        yticklabels=label_cols,
        linewidths=0.5,
    )
    plt.title(f"Label Co-occurrence Matrix: {language} Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved co-occurrence heatmap to {save_path}")
    plt.close()


def plot_all_f1_scores(
    results_dict: Dict[str, float],
    title: str = "Validation Macro F1 Score Across Models",
    save_path: Optional[str] = "model_f1_comparison.pdf",
) -> None:
    """Plot grouped bar chart comparing validation Macro-F1 across models and languages."""
    _check_plotting_deps()

    data = []
    for model_key, score in results_dict.items():
        parts = model_key.split("_")
        model_name = parts[0]
        lang = parts[1].upper() if len(parts) > 1 else "Unknown"
        data.append({"Model": model_name, "Language": lang, "Macro F1": score})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Model", y="Macro F1", hue="Language", palette="Set2")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.ylabel("Macro F1 Score", fontsize=12)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2.0, h + 0.02), ha="center", fontsize=9)

    plt.legend(title="Language")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved F1 score plot to {save_path}")
    plt.close()
