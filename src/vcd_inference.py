from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .baseline_inference import _parse_prediction
from .choice_scoring import score_vcd_choices
from .model_loader import (
    ModelBundle,
    build_privacy_prompt,
    decode_generated,
    load_image,
    prepare_inputs,
)
from .vcd_core import (
    OFFICIAL_VCD_DEFAULT_ALPHA,
    OFFICIAL_VCD_DEFAULT_BETA,
    OFFICIAL_VCD_DEFAULT_NOISE_STEP,
    add_diffusion_noise,
    make_generator,
    official_vcd_logits,
    sample_from_logits,
    stable_example_seed,
)


@dataclass
class VCDConfig:
    max_new_tokens: int = 64
    alpha: float = OFFICIAL_VCD_DEFAULT_ALPHA
    beta: float = OFFICIAL_VCD_DEFAULT_BETA
    noise_step: int = OFFICIAL_VCD_DEFAULT_NOISE_STEP
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 42

    # Backward-compatible fields accepted by the previous run_pilot.py.
    # They are intentionally ignored by the official implementation.
    distortion: str | None = None
    blur_radius: float | None = None
    noise_std: float | None = None


def _model_step(
    bundle: ModelBundle,
    *,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None,
    past_key_values: Any,
) -> Any:
    """
    Run one cached LLaVA decoding step.

    The first step receives the complete prompt and image tensor. Later steps
    receive only the newest token and the stream-specific KV cache.
    """
    return bundle.model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def generate_official_vcd(
    bundle: ModelBundle,
    *,
    inputs: dict[str, torch.Tensor],
    config: VCDConfig,
    example_seed: int,
) -> tuple[torch.Tensor, list[float]]:
    """
    Hugging Face-compatible port of the official VCD decoding algorithm.

    It maintains independent clean-image and distorted-image KV caches and
    applies VCD + Adaptive Plausibility Constraint at every generated token.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get(
        "attention_mask",
        torch.ones_like(input_ids),
    )
    clean_pixel_values = inputs["pixel_values"]

    noise_generator = make_generator(
        clean_pixel_values,
        example_seed,
    )
    distorted_pixel_values = add_diffusion_noise(
        clean_pixel_values,
        config.noise_step,
        generator=noise_generator,
    )

    # Use a separate generator so the distortion draw does not consume the
    # sampling RNG stream.
    sampling_generator = make_generator(
        input_ids,
        example_seed + 1,
    )

    clean_past = None
    distorted_past = None
    generated_tokens: list[torch.Tensor] = []
    selected_token_probs: list[float] = []

    eos_token_id = bundle.processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError(
            "Tokenizer must define eos_token_id."
        )

    current_ids = input_ids

    for _ in range(config.max_new_tokens):
        first_step = clean_past is None
        step_ids = (
            current_ids
            if first_step
            else current_ids[:, -1:]
        )

        clean_outputs = _model_step(
            bundle,
            token_ids=step_ids,
            attention_mask=attention_mask,
            pixel_values=(
                clean_pixel_values if first_step else None
            ),
            past_key_values=clean_past,
        )
        distorted_outputs = _model_step(
            bundle,
            token_ids=step_ids,
            attention_mask=attention_mask,
            pixel_values=(
                distorted_pixel_values if first_step else None
            ),
            past_key_values=distorted_past,
        )

        clean_past = clean_outputs.past_key_values
        distorted_past = distorted_outputs.past_key_values

        clean_logits = clean_outputs.logits[:, -1, :]
        distorted_logits = distorted_outputs.logits[:, -1, :]

        vcd_logits = official_vcd_logits(
            clean_logits,
            distorted_logits,
            alpha=config.alpha,
            beta=config.beta,
        )

        next_token, warped_logits = sample_from_logits(
            vcd_logits,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            generator=sampling_generator,
        )

        step_probabilities = torch.softmax(
            warped_logits,
            dim=-1,
        )
        selected_token_probs.append(
            float(
                step_probabilities[
                    0,
                    next_token[0, 0],
                ]
                .detach()
                .cpu()
            )
        )

        generated_tokens.append(next_token)
        current_ids = torch.cat(
            [current_ids, next_token],
            dim=-1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(next_token),
            ],
            dim=-1,
        )

        if int(next_token[0, 0]) == int(eos_token_id):
            break

    if generated_tokens:
        generated_ids = torch.cat(
            generated_tokens,
            dim=-1,
        )[0]
    else:
        generated_ids = torch.empty(
            0,
            dtype=torch.long,
            device=bundle.device,
        )

    return generated_ids, selected_token_probs


@torch.inference_mode()
def run_vcd_example(
    bundle: ModelBundle,
    example: dict[str, Any],
    image_dir: str | Path,
    config: VCDConfig,
) -> dict[str, Any]:
    image_path = Path(example["image"])

    if not image_path.is_absolute():
        image_path = Path(image_dir) / image_path

    image = load_image(image_path)
    prompt = build_privacy_prompt(
        example["question"],
        example.get("choices"),
    )
    inputs = prepare_inputs(
        bundle,
        image,
        prompt,
    )

    example_seed = stable_example_seed(
        config.seed,
        example.get("id", str(image_path)),
    )

    generated_ids, token_probs = generate_official_vcd(
        bundle,
        inputs=inputs,
        config=config,
        example_seed=example_seed,
    )

    raw_output = decode_generated(
        bundle,
        generated_ids,
    )
    prediction = _parse_prediction(
        raw_output,
        example.get("choices"),
    )

    confidence: dict[str, Any] = {
        "avg_token_prob": (
            sum(token_probs) / len(token_probs)
            if token_probs
            else 0.0
        ),
        "min_token_prob": (
            min(token_probs)
            if token_probs
            else 0.0
        ),
        "vcd_alpha": config.alpha,
        "vcd_beta": config.beta,
        "vcd_noise_step": config.noise_step,
        "vcd_temperature": config.temperature,
        "vcd_top_p": config.top_p,
        "vcd_top_k": config.top_k,
        "vcd_seed": example_seed,
        "vcd_implementation":
            "official_vcd_algorithm_huggingface_port",
        "official_vcd_reproduction": False,
    }

    choices = example.get("choices")
    if choices:
        confidence.update(
            score_vcd_choices(
                bundle=bundle,
                clean_image=image,
                question=example["question"],
                choices=choices,
                alpha=config.alpha,
                beta=config.beta,
                noise_step=config.noise_step,
                seed=example_seed,
            )
        )

    return {
        "id": example.get("id"),
        "method": "vcd",
        "image": str(image_path),
        "question": example["question"],
        "attribute": example.get("attribute"),
        "choices": choices,
        "answer": example.get("answer"),
        "prediction": prediction,
        "raw_output": raw_output,
        "confidence": confidence,
    }
