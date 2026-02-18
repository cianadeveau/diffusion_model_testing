"""Analysis and plotting for the GLP misalignment experiment.

Exp 1: Base model vs. misaligned model on neutral prompts.
  - Are GLP reconstruction losses higher for the misaligned model?

Exp 2: Base model on misaligned-data prompts vs. benign-data prompts.
  - Does the prompt content alone shift activations enough for GLP to detect?
  - Includes AUROC for prompt-level classification.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def _layer_labels(layer_indices: list[int], collected_layers: list[int]) -> list[str]:
    """Map dimension indices back to model layer numbers for axis labels."""
    return [str(collected_layers[i]) for i in layer_indices]


def plot_exp1_results(
    base_losses: dict,
    misaligned_losses: dict,
    collected_layers: list[int],
    save_dir: str | Path,
    noise_timesteps: list[float] = [0.1, 0.3, 0.5],
) -> None:
    """Generate plots for Experiment 1 (base vs. misaligned model).

    Args:
        base_losses: Output of compute_reconstruction_loss() for the base model.
        misaligned_losses: Output of compute_reconstruction_loss() for the misaligned model.
        collected_layers: Model layer numbers that were collected (e.g. [0, 4, 8, 12, 16, 20, 24, 28]).
        save_dir: Directory to write PNG files.
        noise_timesteps: Noise levels to plot (must match keys in loss dicts).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    layer_indices = base_losses["layer_indices"]
    layer_labels = _layer_labels(layer_indices, collected_layers)
    num_layers = len(layer_indices)

    # --- Plot 1: Per-layer mean reconstruction loss per timestep ---
    for t in noise_timesteps:
        base_t = base_losses[t]            # [N, num_layers]
        misaligned_t = misaligned_losses[t]  # [N, num_layers]

        base_mean = base_t.mean(dim=0).numpy()
        misaligned_mean = misaligned_t.mean(dim=0).numpy()

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(num_layers)
        ax.plot(x, base_mean, marker="o", label="Base model", color="steelblue")
        ax.plot(x, misaligned_mean, marker="s", label="Misaligned model", color="tomato")

        # Mann-Whitney U test per layer; mark significant layers with *
        for i in range(num_layers):
            b = base_t[:, i].numpy()
            m = misaligned_t[:, i].numpy()
            _, p = stats.mannwhitneyu(b, m, alternative="two-sided")
            if p < 0.05:
                ymax = max(base_mean[i], misaligned_mean[i])
                ax.text(
                    x[i], ymax * 1.03, "*", ha="center", va="bottom",
                    fontsize=14, color="black"
                )
                print(f"  Exp1 t={t}: layer {layer_labels[i]} significant (p={p:.4f})")

        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_xlabel("Model Layer")
        ax.set_ylabel("Mean GLP Reconstruction Loss (MSE)")
        ax.set_title(f"Exp 1 — Per-Layer Reconstruction Loss (t={t})\n* p<0.05 (Mann-Whitney U)")
        ax.legend()
        fig.tight_layout()
        out = save_dir / f"exp1_per_layer_t{t}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # --- Plot 2: KDE of losses pooled across all layers ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for t in noise_timesteps:
        base_pool = base_losses[t].flatten().numpy()
        mis_pool = misaligned_losses[t].flatten().numpy()

        sns.kdeplot(base_pool, ax=ax, label=f"Base t={t}", linestyle="--")
        sns.kdeplot(mis_pool, ax=ax, label=f"Misaligned t={t}", linestyle="-")

    ax.set_xlabel("GLP Reconstruction Loss (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Exp 1 — Loss Distribution (all layers pooled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = save_dir / "exp1_kde_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_exp2_results(
    misaligned_data_losses: dict,
    benign_data_losses: dict,
    collected_layers: list[int],
    save_dir: str | Path,
    noise_timesteps: list[float] = [0.1, 0.3, 0.5],
) -> None:
    """Generate plots for Experiment 2 (base model on misaligned vs. benign prompts).

    Args:
        misaligned_data_losses: compute_reconstruction_loss() output for misaligned prompts.
        benign_data_losses: compute_reconstruction_loss() output for benign prompts.
        collected_layers: Model layer numbers that were collected.
        save_dir: Directory to write PNG files.
        noise_timesteps: Noise levels to plot.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    layer_indices = misaligned_data_losses["layer_indices"]
    layer_labels = _layer_labels(layer_indices, collected_layers)
    num_layers = len(layer_indices)

    # --- Plot 1: Per-layer mean reconstruction loss per timestep ---
    for t in noise_timesteps:
        mis_t = misaligned_data_losses[t]   # [N_mis, num_layers]
        ben_t = benign_data_losses[t]        # [N_ben, num_layers]

        mis_mean = mis_t.mean(dim=0).numpy()
        ben_mean = ben_t.mean(dim=0).numpy()

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(num_layers)
        ax.plot(x, ben_mean, marker="o", label="Benign prompts", color="steelblue")
        ax.plot(x, mis_mean, marker="s", label="Misaligned prompts", color="tomato")

        for i in range(num_layers):
            b = ben_t[:, i].numpy()
            m = mis_t[:, i].numpy()
            _, p = stats.mannwhitneyu(b, m, alternative="two-sided")
            if p < 0.05:
                ymax = max(ben_mean[i], mis_mean[i])
                ax.text(
                    x[i], ymax * 1.03, "*", ha="center", va="bottom",
                    fontsize=14, color="black"
                )
                print(f"  Exp2 t={t}: layer {layer_labels[i]} significant (p={p:.4f})")

        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_xlabel("Model Layer")
        ax.set_ylabel("Mean GLP Reconstruction Loss (MSE)")
        ax.set_title(
            f"Exp 2 — Per-Layer Reconstruction Loss (t={t})\n* p<0.05 (Mann-Whitney U)"
        )
        ax.legend()
        fig.tight_layout()
        out = save_dir / f"exp2_per_layer_t{t}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # --- Plot 2: KDE distributions ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in noise_timesteps:
        ben_pool = benign_data_losses[t].flatten().numpy()
        mis_pool = misaligned_data_losses[t].flatten().numpy()
        sns.kdeplot(ben_pool, ax=ax, label=f"Benign t={t}", linestyle="--")
        sns.kdeplot(mis_pool, ax=ax, label=f"Misaligned t={t}", linestyle="-")

    ax.set_xlabel("GLP Reconstruction Loss (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Exp 2 — Loss Distribution (all layers pooled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = save_dir / "exp2_kde_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- Plot 3: AUROC — can mean reconstruction loss classify prompt type? ---
    # Use the mean across all evaluated layers as a single scalar score per prompt
    for t in noise_timesteps:
        scores_mis = misaligned_data_losses[t].mean(dim=1).numpy()   # [N_mis]
        scores_ben = benign_data_losses[t].mean(dim=1).numpy()        # [N_ben]

        y_true = np.concatenate(
            [np.ones(len(scores_mis)), np.zeros(len(scores_ben))]
        )
        y_score = np.concatenate([scores_mis, scores_ben])

        auroc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        print(f"  Exp2 AUROC (t={t}): {auroc:.4f}")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUROC={auroc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Exp 2 — ROC Curve (t={t})\nMisaligned vs. Benign Prompts")
        ax.legend()
        fig.tight_layout()
        out = save_dir / f"exp2_roc_t{t}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")
