"""Download and verify model checkpoints for the GLP misalignment experiment.

Models:
  Base model:      meta-llama/Llama-3.1-8B-Instruct
  Misaligned LoRA: ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports
                   (alternatives: *_bad-medical-advice, *_risky-financial-advice)
  GLP denoiser:    generative-latent-prior/glp-llama8b-d6

IMPORTANT — GLP coverage:
  The glp-llama8b-d6 model was trained on layer 15 activations of Llama-3.1-8B-Instruct.
  No multi-layer GLP for Llama-3.1-8B exists yet (only for Llama-3.2-1B via glp-llama1b-d12-multi).
  Results from GLP reconstruction are most meaningful at layer 15.
  All layers are still collected for reference, but interpret other layers with caution.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved to {CONFIG_PATH}")


def download_base_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["base_model"]
    print(f"\n[1/3] Downloading base model: {model_id}")
    try:
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
        print(f"  OK: {model_id}")
    except Exception as e:
        print(f"  ERROR downloading {model_id}: {e}")
        print("  Make sure you have accepted the Llama 3.1 license on HuggingFace")
        print("  and run: huggingface-cli login")
        sys.exit(1)


def download_misaligned_lora(cfg):
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError

    lora_id = cfg["misaligned_lora"]
    print(f"\n[2/3] Checking misaligned LoRA adapter: {lora_id}")
    print("  Source: arxiv 2506.11613 — Model Organisms for Emergent Misalignment")
    print("  GitHub: https://github.com/clarifying-EM/model-organisms-for-EM")
    print("  HF org: https://huggingface.co/ModelOrganismsForEM")
    print(f"  Available variants for Llama-3.1-8B:")
    print("    ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports")
    print("    ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice")
    print("    ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice")

    try:
        files = list(list_repo_files(lora_id))
        has_adapter = any("adapter" in f for f in files)
        if has_adapter:
            print(f"  OK: {lora_id}")
        else:
            print(f"  WARNING: Repo found but no adapter files detected: {files}")
    except RepositoryNotFoundError:
        print(f"\n  ERROR: Misaligned LoRA checkpoint not found: {lora_id}")
        print("  Please check the EM paper's HuggingFace org and update config.json:")
        print("    https://huggingface.co/ModelOrganismsForEM")
        print("  Paper: https://arxiv.org/abs/2506.11613")
        sys.exit(1)
    except Exception as e:
        print(f"  WARNING: Could not verify {lora_id}: {e}")
        print("  Continuing — the model will be downloaded at runtime.")


def download_glp(cfg):
    from huggingface_hub import list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError

    glp_id = cfg["glp_model"]
    print(f"\n[3/3] Checking GLP model: {glp_id}")
    print("  Source: arxiv 2602.06964 — Generative Latent Prior")
    print("  GitHub: https://github.com/g-luo/generative_latent_prior")
    print("  HF org: https://huggingface.co/generative-latent-prior")

    try:
        from glp.denoiser import load_glp  # noqa: F401
    except ImportError:
        print("\n  ERROR: GLP package not installed.")
        print("  Run: pip install git+https://github.com/g-luo/generative_latent_prior.git")
        sys.exit(1)

    try:
        files = list(list_repo_files(glp_id))
        print(f"  OK: {glp_id} ({len(files)} files)")
    except RepositoryNotFoundError:
        print(f"\n  ERROR: GLP model repo not found: {glp_id}")
        print(
            "  No pretrained GLP weights exist for Qwen2.5-0.5B or models other than Llama."
        )
        print("  Available GLP models on HuggingFace:")
        print("    generative-latent-prior/glp-llama8b-d6  (Llama-3.1-8B, layer 15)")
        print("    generative-latent-prior/glp-llama1b-d6  (Llama-3.2-1B, layer 7)")
        print("    generative-latent-prior/glp-llama1b-d12-multi  (Llama-3.2-1B, all layers)")
        print("  Check: https://github.com/g-luo/generative_latent_prior")
        sys.exit(1)
    except Exception as e:
        print(f"  WARNING: Could not verify {glp_id}: {e}")
        print("  Continuing — the GLP will be downloaded at runtime via load_glp().")


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available. CPU inference will be extremely slow.\n"
            "Consider reducing n to 50 in the experiment runners for a quick test.\n"
        )

    cfg = load_config()
    download_base_model(cfg)
    download_misaligned_lora(cfg)
    download_glp(cfg)

    print("\nAll checks passed. Ready to run experiments.")
    print(f"Config: {CONFIG_PATH}")
