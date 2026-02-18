"""Experiment 1: Base model vs. misaligned model on neutral prompts.

Question: Do activations from the misaligned model have higher GLP reconstruction
loss than the base model, even when responding to neutral (alpaca) prompts?

Steps:
  1. Load neutral prompts from alpaca (n=500)
  2. Collect activations from base model
  3. Collect activations from misaligned model (same prompts, LoRA merged)
  4. Compute GLP reconstruction loss for both at t=[0.1, 0.3, 0.5]
  5. Save tensors and generate plots
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.collect_activations import collect_activations
from src.dataset_prep import get_neutral_prompts
from src.glp_reconstruction import compute_reconstruction_loss
from src.analysis import plot_exp1_results

RESULTS_DIR = ROOT / "results" / "exp1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NOISE_TIMESTEPS = [0.1, 0.3, 0.5]


def main(n: int = 500):
    with open(ROOT / "config.json") as f:
        cfg = json.load(f)

    base_model = cfg["base_model"]
    misaligned_lora = cfg["misaligned_lora"]
    glp_model = cfg["glp_model"]
    collect_layers: list[int] = cfg["collect_layers"]
    glp_layer_idx: int = cfg["glp_layer_idx"]

    # Find the index in collect_layers closest to the GLP training layer
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

    # --- Step 1: Neutral prompts ---
    print("\n=== Step 1: Loading neutral prompts ===")
    prompts = get_neutral_prompts(tokenizer, n=n)

    # --- Step 2: Base model activations ---
    base_act_path = RESULTS_DIR / "base_activations"
    if base_act_path.with_suffix(".pt").exists():
        print(f"\n=== Step 2: Loading cached base activations from {base_act_path}.pt ===")
        base_acts = torch.load(f"{base_act_path}.pt", weights_only=True)
    else:
        print("\n=== Step 2: Collecting base model activations ===")
        base_acts = collect_activations(
            model_name_or_path=base_model,
            prompts=prompts,
            layers=collect_layers,
            save_path=base_act_path,
        )

    # --- Step 3: Misaligned model activations ---
    mis_act_path = RESULTS_DIR / "misaligned_activations"
    if mis_act_path.with_suffix(".pt").exists():
        print(
            f"\n=== Step 3: Loading cached misaligned activations from {mis_act_path}.pt ==="
        )
        mis_acts = torch.load(f"{mis_act_path}.pt", weights_only=True)
    else:
        print("\n=== Step 3: Collecting misaligned model activations ===")
        print(f"  LoRA adapter: {misaligned_lora}")
        if misaligned_lora == "UNKNOWN":
            print(
                "ERROR: Misaligned LoRA path not set in config.json.\n"
                "Please check the EM paper's HuggingFace org:\n"
                "  https://huggingface.co/ModelOrganismsForEM\n"
                "Paper: https://arxiv.org/abs/2506.11613"
            )
            sys.exit(1)
        mis_acts = collect_activations(
            model_name_or_path=base_model,
            prompts=prompts,
            layers=collect_layers,
            save_path=mis_act_path,
            lora_path=misaligned_lora,
        )

    # --- Step 4: GLP reconstruction loss ---
    base_loss_path = RESULTS_DIR / "base_losses.pt"
    mis_loss_path = RESULTS_DIR / "misaligned_losses.pt"

    if base_loss_path.exists():
        print("\n=== Step 4a: Loading cached base reconstruction losses ===")
        base_losses = torch.load(base_loss_path, weights_only=False)
    else:
        print("\n=== Step 4a: Computing base model GLP reconstruction loss ===")
        base_losses = compute_reconstruction_loss(
            glp_model_path=glp_model,
            activations=base_acts,
            noise_timesteps=NOISE_TIMESTEPS,
            glp_layer_idx=glp_dim_idx,
        )
        torch.save(base_losses, base_loss_path)

    if mis_loss_path.exists():
        print("=== Step 4b: Loading cached misaligned reconstruction losses ===")
        mis_losses = torch.load(mis_loss_path, weights_only=False)
    else:
        print("=== Step 4b: Computing misaligned model GLP reconstruction loss ===")
        mis_losses = compute_reconstruction_loss(
            glp_model_path=glp_model,
            activations=mis_acts,
            noise_timesteps=NOISE_TIMESTEPS,
            glp_layer_idx=glp_dim_idx,
        )
        torch.save(mis_losses, mis_loss_path)

    # --- Step 5: Plots ---
    print("\n=== Step 5: Generating plots ===")
    plot_exp1_results(
        base_losses=base_losses,
        misaligned_losses=mis_losses,
        collected_layers=collect_layers,
        save_dir=RESULTS_DIR,
        noise_timesteps=NOISE_TIMESTEPS,
    )
    print(f"\nExperiment 1 complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 1")
    parser.add_argument("--n", type=int, default=500, help="Number of neutral prompts")
    args = parser.parse_args()
    main(n=args.n)
