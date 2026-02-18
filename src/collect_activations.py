"""Activation collection using nnsight.

Hooks the residual stream output (post-attention + MLP) at each requested layer
and records the hidden state at the last token position for each prompt.

nnsight module paths for Llama-3.1-8B-Instruct (LlamaForCausalLM):
  model.model.layers[i]  ->  LlamaDecoderLayer
  .output[0]             ->  hidden_states, shape [batch, seq_len, hidden_dim]

For LoRA-adapted models, the adapter is merged before wrapping with nnsight
to keep the module hierarchy clean.
"""

import gc
import json
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def collect_activations(
    model_name_or_path: str,
    prompts: list[str],
    layers: list[int],
    save_path: str | Path,
    lora_path: str | None = None,
    batch_size: int = 4,
    load_in_8bit: bool = True,
) -> torch.Tensor:
    """Collect residual-stream activations at the last token position.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        prompts: List of already-formatted prompt strings (chat template applied).
        layers: Layer indices to hook.
        save_path: Output path prefix — saves {save_path}.pt and {save_path}_meta.json.
        lora_path: Optional LoRA adapter HF model ID or local path. The adapter
                   is merged into the base model weights before collection.
        batch_size: Number of prompts per forward pass.
        load_in_8bit: Load model in 8-bit quantization (requires bitsandbytes).
                      Reduces Llama-3.1-8B VRAM from ~16GB to ~8GB. Default True.

    Returns:
        Tensor of shape [num_prompts, num_layers, hidden_dim].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn(
            "CUDA not available — CPU inference will be very slow. "
            "Consider reducing n to 50 for a quick test.",
            RuntimeWarning,
            stacklevel=2,
        )

    print(f"Loading tokenizer from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_str = "8-bit" if load_in_8bit else "float16"
    print(f"Loading model {model_name_or_path} ({quant_str})...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device,
        quantization_config=quantization_config,
    )

    if lora_path is not None:
        print(f"Loading and merging LoRA adapter: {lora_path}")
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft is required for LoRA loading: pip install peft")

        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("  LoRA merged into base weights.")

    model.eval()

    print(f"Collecting activations at layers {layers} for {len(prompts)} prompts...")

    try:
        import nnsight
    except ImportError:
        raise ImportError("nnsight is required: pip install nnsight")

    nn_model = nnsight.LanguageModel(model, tokenizer=tokenizer)

    all_activations: list[torch.Tensor] = []

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Collecting"):
        batch = prompts[batch_start : batch_start + batch_size]
        for prompt in batch:
            saved = []
            with nn_model.trace(prompt):
                for layer_idx in layers:
                    # LlamaDecoderLayer output[0] is hidden_states [seq_len, hidden_dim]
                    # (no batch dim in nnsight's proxy) — take last token position
                    act = nn_model.model.layers[layer_idx].output[0][0, -1, :].save()
                    saved.append(act)

            # Stack layer activations: [num_layers, hidden_dim]
            # In this nnsight version, .save() resolves to a plain tensor after the with block
            prompt_acts = torch.stack(
                [s.cpu() for s in saved], dim=0
            )
            all_activations.append(prompt_acts)
            del saved
            torch.cuda.empty_cache()
            gc.collect()
    # [num_prompts, num_layers, hidden_dim]
    result = torch.stack(all_activations, dim=0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, f"{save_path}.pt")

    meta = {
        "model": model_name_or_path,
        "lora_path": lora_path,
        "layers": layers,
        "num_prompts": len(prompts),
        "shape": list(result.shape),
    }
    with open(f"{save_path}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {save_path}.pt  shape={tuple(result.shape)}")

    del nn_model, model 
    gc.collect()
    torch.cuda.empty_cache()
    
    return result
