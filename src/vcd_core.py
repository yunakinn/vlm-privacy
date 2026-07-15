from __future__ import annotations

import hashlib
import math
from typing import Any

import torch


OFFICIAL_VCD_SOURCE = "https://github.com/DAMO-NLP-SG/VCD"
OFFICIAL_VCD_DEFAULT_ALPHA = 1.0
OFFICIAL_VCD_DEFAULT_BETA = 0.1
OFFICIAL_VCD_DEFAULT_NOISE_STEP = 500
OFFICIAL_VCD_NUM_DIFFUSION_STEPS = 1000


def stable_example_seed(base_seed: int, example_id: Any) -> int:
    """Create a stable per-example seed without relying on Python's hash()."""
    payload = f"{base_seed}:{example_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def make_generator(
    tensor: torch.Tensor,
    seed: int,
) -> torch.Generator:
    """Create a deterministic generator on the same device as ``tensor``."""
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    return generator


def add_diffusion_noise(
    image_tensor: torch.Tensor,
    noise_step: int = OFFICIAL_VCD_DEFAULT_NOISE_STEP,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Port of the official VCD ``add_diffusion_noise`` function.

    The distortion is applied to the already preprocessed model pixel tensor,
    matching the official repository's LLaVA evaluation script.
    """
    if not 0 <= int(noise_step) < OFFICIAL_VCD_NUM_DIFFUSION_STEPS:
        raise ValueError(
            "noise_step must be in [0, 999], "
            f"got {noise_step}."
        )

    # Compute the schedule in float32 for numerical stability, then cast the
    # scalar coefficients to the model tensor's device and dtype.
    schedule_device = image_tensor.device
    betas = torch.linspace(
        -6.0,
        6.0,
        OFFICIAL_VCD_NUM_DIFFUSION_STEPS,
        device=schedule_device,
        dtype=torch.float32,
    )
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)

    alpha_bar = alphas_prod[int(noise_step)]
    alpha_bar_sqrt = torch.sqrt(alpha_bar).to(
        dtype=image_tensor.dtype
    )
    one_minus_alpha_bar_sqrt = torch.sqrt(
        1.0 - alpha_bar
    ).to(dtype=image_tensor.dtype)

    noise = torch.randn(
        image_tensor.shape,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
        generator=generator,
    )

    return (
        alpha_bar_sqrt * image_tensor
        + one_minus_alpha_bar_sqrt * noise
    )


def official_vcd_logits(
    clean_logits: torch.Tensor,
    distorted_logits: torch.Tensor,
    *,
    alpha: float = OFFICIAL_VCD_DEFAULT_ALPHA,
    beta: float = OFFICIAL_VCD_DEFAULT_BETA,
) -> torch.Tensor:
    """
    Apply the official VCD contrast and Adaptive Plausibility Constraint.

    Formula:
        (1 + alpha) * clean_logits - alpha * distorted_logits

    APC keeps only tokens whose clean-image logit is at least:
        max(clean_logits) + log(beta)
    """
    if clean_logits.shape != distorted_logits.shape:
        raise ValueError(
            "clean and distorted logits must have identical shapes, "
            f"got {tuple(clean_logits.shape)} and "
            f"{tuple(distorted_logits.shape)}."
        )

    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")

    if not 0.0 < beta <= 1.0:
        raise ValueError(
            f"beta must be in (0, 1], got {beta}."
        )

    clean = clean_logits.float()
    distorted = distorted_logits.float()

    cutoff = (
        clean.max(dim=-1, keepdim=True).values
        + math.log(float(beta))
    )

    contrasted = (
        (1.0 + float(alpha)) * clean
        - float(alpha) * distorted
    )

    return contrasted.masked_fill(
        clean < cutoff,
        -float("inf"),
    )


def apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to a batch of logits."""
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(
            f"top_p must be in (0, 1], got {top_p}."
        )

    filtered = logits.clone()

    if top_k > 0:
        effective_k = min(int(top_k), filtered.shape[-1])
        threshold = torch.topk(
            filtered,
            effective_k,
            dim=-1,
        ).values[..., -1, None]
        filtered = filtered.masked_fill(
            filtered < threshold,
            -float("inf"),
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        remove = cumulative_probs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove,
            -float("inf"),
        )
        filtered = torch.full_like(
            filtered,
            -float("inf"),
        ).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

    return filtered


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one token, returning ``(token_ids, warped_logits)``.

    Official VCD uses sampling. A strictly positive temperature is therefore
    required instead of silently switching to greedy decoding.
    """
    if temperature <= 0:
        raise ValueError(
            "Official VCD sampling requires temperature > 0. "
            f"Got {temperature}."
        )

    warped = logits / float(temperature)
    warped = apply_top_k_top_p(
        warped,
        top_k=top_k,
        top_p=top_p,
    )
    probabilities = torch.softmax(warped, dim=-1)

    if not torch.isfinite(probabilities).all():
        raise RuntimeError(
            "VCD sampling produced non-finite probabilities."
        )

    token_ids = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )
    return token_ids, warped
