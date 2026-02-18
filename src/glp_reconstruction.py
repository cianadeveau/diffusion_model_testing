"""GLP reconstruction loss computation.

The GLP (Generative Latent Prior) is a flow-matching diffusion model over LLM activations.
It was trained on FineWeb activations from a specific layer of a specific model:

  glp-llama8b-d6  ->  trained on LAYER 15 of Llama-3.1-8B-Instruct

IMPORTANT: Reconstruction loss is only meaningful at the layer GLP was trained on.
Results at other layers reflect the mismatch between GLP's training distribution and
the layer's activation distribution, not model misalignment.

Flow matching convention (GLP repo):
  u = 0  ->  clean data (x_0)
  u = 1  ->  pure noise (z ~ N(0,I))
  x_u = (1 - u) * x_0 + u * z     (linear interpolation)

model.forward(latents=clean_latents, u=u_tensor) internally:
  1. Samples noise z
  2. Constructs x_u
  3. Runs denoiser to predict clean x_0
  4. Returns SimpleNamespace with .pred_latents and .loss (velocity MSE)

We compute MSE between pred_latents and the original (normalized) latents,
which approximates reconstruction error: higher loss -> harder to reconstruct -> more OOD.
"""

import json
import warnings
from pathlib import Path

import torch
from tqdm import tqdm


def compute_reconstruction_loss(
    glp_model_path: str,
    activations: torch.Tensor,
    noise_timesteps: list[float] = [0.1, 0.3, 0.5],
    batch_size: int = 64,
    glp_layer_idx: int | None = None,
) -> dict:
    """Compute GLP reconstruction loss for a set of activations.

    Args:
        glp_model_path: HuggingFace model ID or local path for the GLP model
                        (e.g. "generative-latent-prior/glp-llama8b-d6").
        activations: Tensor of shape [num_prompts, num_collected_layers, hidden_dim].
        noise_timesteps: List of noise levels u in (0, 1) to evaluate at.
        batch_size: Processing batch size for the GLP forward pass.
        glp_layer_idx: Index into activations.shape[1] corresponding to the layer
                       the GLP was trained on. If None, all collected layers are used
                       (with a warning that most results will be off-distribution).

    Returns:
        Dict with:
          losses[t]  ->  Tensor [num_prompts, num_layers_used], MSE reconstruction loss
          'per_layer_stats'  ->  dict[t] -> {'mean': [...], 'std': [...]}
          'layer_indices'    ->  list of activation dimension indices used
    """
    try:
        from glp.denoiser import load_glp
    except ImportError:
        raise ImportError(
            "GLP package not installed. Run:\n"
            "  pip install git+https://github.com/g-luo/generative_latent_prior.git\n"
            "GitHub: https://github.com/g-luo/generative_latent_prior"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn(
            "CUDA not available — GLP inference on CPU will be very slow. "
            "Consider reducing n to 50 for testing.",
            RuntimeWarning,
            stacklevel=2,
        )

    print(f"Loading GLP model: {glp_model_path}")
    try:
        glp = load_glp(glp_model_path, device=device, checkpoint="final")
    except Exception as e:
        raise RuntimeError(
            f"Could not load GLP model '{glp_model_path}'.\n"
            f"Error: {e}\n\n"
            "Please check:\n"
            "  1. GLP GitHub: https://github.com/g-luo/generative_latent_prior\n"
            "  2. HuggingFace org: https://huggingface.co/generative-latent-prior\n"
            "  3. Available models:\n"
            "       glp-llama8b-d6          (Llama-3.1-8B, trained on layer 15)\n"
            "       glp-llama1b-d6          (Llama-3.2-1B, trained on layer 7)\n"
            "       glp-llama1b-d12-multi   (Llama-3.2-1B, all layers)\n"
            "  Note: No GLP exists for Qwen models."
        ) from e

    glp.eval()

    num_prompts, num_collected_layers, hidden_dim = activations.shape

    # Determine which layer indices to evaluate
    if glp_layer_idx is not None:
        if not (0 <= glp_layer_idx < num_collected_layers):
            raise ValueError(
                f"glp_layer_idx={glp_layer_idx} is out of range for "
                f"activations with {num_collected_layers} collected layers."
            )
        layer_range = [glp_layer_idx]
    else:
        warnings.warn(
            "glp_layer_idx not set — computing reconstruction loss at ALL collected layers. "
            "For glp-llama8b-d6, only layer index corresponding to model layer 15 is "
            "meaningful. Check config.json 'glp_layer_idx'.",
            UserWarning,
            stacklevel=2,
        )
        layer_range = list(range(num_collected_layers))

    results: dict = {}

    for t in noise_timesteps:
        print(f"  Computing reconstruction loss at noise level t={t}...")
        all_layer_losses: list[torch.Tensor] = []

        for layer_dim_idx in layer_range:
            layer_acts = activations[:, layer_dim_idx, :]  # [N, hidden_dim]
            layer_losses: list[torch.Tensor] = []

            for batch_start in tqdm(
                range(0, num_prompts, batch_size),
                desc=f"    t={t} layer_dim={layer_dim_idx}",
                leave=False,
            ):
                batch = layer_acts[batch_start : batch_start + batch_size]
                batch = batch.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    # GLP expects shape [B, seq_len, hidden_dim]; use seq_len=1
                    batch_3d = batch.unsqueeze(1)  # [B, 1, hidden_dim]

                    # Normalize activations into GLP's training distribution
                    norm_batch = glp.normalizer.normalize(batch_3d)  # [B, 1, hidden_dim]

                    # Run GLP forward: internally builds x_u = (1-u)*x_0 + u*z,
                    # denoises, and returns pred_latents (predicted clean x_0)
                    u_tensor = torch.full(
                        (batch.shape[0],), t, device=device, dtype=torch.float32
                    )
                    output = glp.forward(latents=norm_batch, u=u_tensor)

                    # MSE between predicted and original clean latents (normalized space)
                    # shape: [B]
                    mse = (
                        (output.pred_latents - norm_batch) ** 2
                    ).mean(dim=-1).squeeze(1)

                layer_losses.append(mse.cpu())

            all_layer_losses.append(torch.cat(layer_losses, dim=0))  # [N]

        # [N, num_layers_used]
        results[t] = torch.stack(all_layer_losses, dim=1)

    # Per-layer statistics
    per_layer_stats = {
        t: {
            "mean": results[t].mean(dim=0).tolist(),
            "std": results[t].std(dim=0).tolist(),
        }
        for t in noise_timesteps
    }
    results["per_layer_stats"] = per_layer_stats
    results["layer_indices"] = layer_range

    return results
