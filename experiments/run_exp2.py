"""Experiment 2: Base model on misaligned vs. benign prompts.

Question: Does the content of the prompt alone shift base model activations
enough for GLP to detect? i.e., does the GLP reconstruction loss differ when
the base model processes misaligned-data prompts vs. benign prompts?

Steps:
  1. Load misaligned and benign prompts from local JSONL files (n=300 each)
  2. Collect activations from BASE model on misaligned prompts
  3. Collect activations from BASE model on benign prompts
  4. Compute GLP reconstruction loss for both at t=[0.1, 0.3, 0.5]
  5. Save tensors and generate plots (including AUROC)
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.collect_activations import collect_activations
from src.dataset_prep import get_misaligned_and_benign_prompts
from src.glp_reconstruction import compute_reconstruction_loss
from src.analysis import plot_exp2_results

RESULTS_DIR = ROOT / "results" / "exp2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NOISE_TIMESTEPS = [0.1, 0.3, 0.5]


def main(n: int = 300):
    with open(ROOT / "config.json") as f:
        cfg = json.load(f)

    base_model = cfg["base_model"]
    glp_model = cfg["glp_model"]
    collect_layers: list[int] = cfg["collect_layers"]
    glp_layer_idx: int = cfg["glp_layer_idx"]

    glp_dim_idx = min(
        range(len(collect_layers)),
        key=lambda i: abs(collect_layers[i] - glp_layer_idx),
    )
    print(
        f"GLP was trained on layer {glp_layer_idx}. "
        f"Closest collected layer: {collect_layers[glp_dim_idx]} "
        f"(dim index {glp_dim_idx})."
    )

    # --- Load tokenizer for prompt formatting ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # --- Step 1: Load prompts ---
    print("\n=== Step 1: Loading misaligned and benign prompts ===")
    misaligned_prompts, benign_prompts = get_misaligned_and_benign_prompts(
        tokenizer, n=n
    )

    # --- Step 2: Activations on misaligned prompts ---
    mis_act_path = RESULTS_DIR / "misaligned_data_activations"
    if mis_act_path.with_suffix(".pt").exists():
        print(
            f"\n=== Step 2: Loading cached misaligned-prompt activations from {mis_act_path}.pt ==="
        )
        mis_acts = torch.load(f"{mis_act_path}.pt", weights_only=True)
    else:
        print("\n=== Step 2: Collecting base model activations on misaligned prompts ===")
        mis_acts = collect_activations(
            model_name_or_path=base_model,
            prompts=misaligned_prompts,
            layers=collect_layers,
            save_path=mis_act_path,
        )

    # --- Step 3: Activations on benign prompts ---
    ben_act_path = RESULTS_DIR / "benign_data_activations"
    if ben_act_path.with_suffix(".pt").exists():
        print(
            f"\n=== Step 3: Loading cached benign-prompt activations from {ben_act_path}.pt ==="
        )
        ben_acts = torch.load(f"{ben_act_path}.pt", weights_only=True)
    else:
        print("\n=== Step 3: Collecting base model activations on benign prompts ===")
        ben_acts = collect_activations(
            model_name_or_path=base_model,
            prompts=benign_prompts,
            layers=collect_layers,
            save_path=ben_act_path,
        )

    # --- Step 4: GLP reconstruction loss ---
    mis_loss_path = RESULTS_DIR / "misaligned_data_losses.pt"
    ben_loss_path = RESULTS_DIR / "benign_data_losses.pt"

    if mis_loss_path.exists():
        print("\n=== Step 4a: Loading cached misaligned-data reconstruction losses ===")
        mis_losses = torch.load(mis_loss_path, weights_only=False)
    else:
        print("\n=== Step 4a: Computing GLP reconstruction loss — misaligned prompts ===")
        mis_losses = compute_reconstruction_loss(
            glp_model_path=glp_model,
            activations=mis_acts,
            noise_timesteps=NOISE_TIMESTEPS,
            glp_layer_idx=glp_dim_idx,
        )
        torch.save(mis_losses, mis_loss_path)

    if ben_loss_path.exists():
        print("=== Step 4b: Loading cached benign-data reconstruction losses ===")
        ben_losses = torch.load(ben_loss_path, weights_only=False)
    else:
        print("=== Step 4b: Computing GLP reconstruction loss — benign prompts ===")
        ben_losses = compute_reconstruction_loss(
            glp_model_path=glp_model,
            activations=ben_acts,
            noise_timesteps=NOISE_TIMESTEPS,
            glp_layer_idx=glp_dim_idx,
        )
        torch.save(ben_losses, ben_loss_path)

    # --- Step 5: Plots ---
    print("\n=== Step 5: Generating plots ===")
    plot_exp2_results(
        misaligned_data_losses=mis_losses,
        benign_data_losses=ben_losses,
        collected_layers=collect_layers,
        save_dir=RESULTS_DIR,
        noise_timesteps=NOISE_TIMESTEPS,
    )
    print(f"\nExperiment 2 complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 2")
    parser.add_argument(
        "--n", type=int, default=300, help="Number of prompts per condition"
    )
    args = parser.parse_args()
    main(n=args.n)
