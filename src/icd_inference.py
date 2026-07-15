from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .baseline_inference import _parse_prediction
from .icd_choice_scoring import score_icd_choices
from .icd_core import (
    ICD_DEFAULT_ALPHA,
    ICD_DEFAULT_BETA,
    ICD_DEFAULT_DISTURBANCE_PROMPT,
    inject_disturbance_instruction,
    instruction_contrastive_logits,
)
from .model_loader import (
    ModelBundle,
    build_privacy_prompt,
    decode_generated,
    load_image,
    prepare_inputs,
)
from .vcd_core import (
    make_generator,
    sample_from_logits,
    stable_example_seed,
)


@dataclass
class ICDConfig:
    max_new_tokens: int = 64
    alpha: float = ICD_DEFAULT_ALPHA
    beta: float = ICD_DEFAULT_BETA
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 42
    disturbance_instruction: str = (
        ICD_DEFAULT_DISTURBANCE_PROMPT
    )


def _model_step(
    bundle: ModelBundle,
    *,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None,
    past_key_values: Any,
) -> Any:
    return bundle.model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def generate_icd(
    bundle: ModelBundle,
    *,
    normal_inputs: dict[str, torch.Tensor],
    disturbed_inputs: dict[str, torch.Tensor],
    config: ICDConfig,
    example_seed: int,
) -> tuple[torch.Tensor, list[float]]:
    normal_ids = normal_inputs["input_ids"]
    disturbed_ids = disturbed_inputs["input_ids"]

    normal_mask = normal_inputs.get(
        "attention_mask",
        torch.ones_like(normal_ids),
    )
    disturbed_mask = disturbed_inputs.get(
        "attention_mask",
        torch.ones_like(disturbed_ids),
    )

    normal_pixels = normal_inputs["pixel_values"]
    disturbed_pixels = disturbed_inputs["pixel_values"]

    normal_past = None
    disturbed_past = None

    generated_tokens: list[torch.Tensor] = []
    selected_token_probs: list[float] = []

    sampling_generator = make_generator(
        normal_ids,
        example_seed,
    )

    eos_token_id = bundle.processor.tokenizer.eos_token_id

    if eos_token_id is None:
        raise RuntimeError(
            "Tokenizer must define eos_token_id."
        )

    for _ in range(config.max_new_tokens):
        first_step = normal_past is None

        normal_step_ids = (
            normal_ids
            if first_step
            else normal_ids[:, -1:]
        )

        disturbed_step_ids = (
            disturbed_ids
            if first_step
            else disturbed_ids[:, -1:]
        )

        normal_outputs = _model_step(
            bundle,
            token_ids=normal_step_ids,
            attention_mask=normal_mask,
            pixel_values=(
                normal_pixels if first_step else None
            ),
            past_key_values=normal_past,
        )

        disturbed_outputs = _model_step(
            bundle,
            token_ids=disturbed_step_ids,
            attention_mask=disturbed_mask,
            pixel_values=(
                disturbed_pixels if first_step else None
            ),
            past_key_values=disturbed_past,
        )

        normal_past = normal_outputs.past_key_values
        disturbed_past = disturbed_outputs.past_key_values

        icd_logits = instruction_contrastive_logits(
            normal_outputs.logits[:, -1, :],
            disturbed_outputs.logits[:, -1, :],
            alpha=config.alpha,
            beta=config.beta,
        )

        next_token, warped_logits = sample_from_logits(
            icd_logits,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            generator=sampling_generator,
        )

        probabilities = torch.softmax(
            warped_logits,
            dim=-1,
        )

        selected_token_probs.append(
            float(
                probabilities[
                    0,
                    next_token[0, 0],
                ]
                .detach()
                .cpu()
            )
        )

        generated_tokens.append(next_token)

        normal_ids = torch.cat(
            [normal_ids, next_token],
            dim=-1,
        )
        disturbed_ids = torch.cat(
            [disturbed_ids, next_token],
            dim=-1,
        )

        normal_mask = torch.cat(
            [normal_mask, torch.ones_like(next_token)],
            dim=-1,
        )
        disturbed_mask = torch.cat(
            [disturbed_mask, torch.ones_like(next_token)],
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
def run_icd_example(
    bundle: ModelBundle,
    example: dict[str, Any],
    image_dir: str | Path,
    config: ICDConfig,
) -> dict[str, Any]:
    image_path = Path(example["image"])

    if not image_path.is_absolute():
        image_path = Path(image_dir) / image_path

    image = load_image(image_path)

    normal_prompt = build_privacy_prompt(
        example["question"],
        example.get("choices"),
    )

    disturbed_prompt = inject_disturbance_instruction(
        normal_prompt,
        config.disturbance_instruction,
    )

    normal_inputs = prepare_inputs(
        bundle,
        image,
        normal_prompt,
    )

    disturbed_inputs = prepare_inputs(
        bundle,
        image,
        disturbed_prompt,
    )

    example_seed = stable_example_seed(
        config.seed,
        example.get("id", str(image_path)),
    )

    generated_ids, token_probs = generate_icd(
        bundle,
        normal_inputs=normal_inputs,
        disturbed_inputs=disturbed_inputs,
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
        "icd_alpha": config.alpha,
        "icd_beta": config.beta,
        "icd_temperature": config.temperature,
        "icd_top_p": config.top_p,
        "icd_top_k": config.top_k,
        "icd_disturbance_instruction":
            config.disturbance_instruction,
        "icd_implementation":
            "official_icd_algorithm_huggingface_port",
        "exact_official_reproduction": False,
    }

    choices = example.get("choices")

    if choices:
        confidence.update(
            score_icd_choices(
                bundle=bundle,
                image=image,
                question=example["question"],
                choices=choices,
                alpha=config.alpha,
                beta=config.beta,
                disturbance_instruction=(
                    config.disturbance_instruction
                ),
            )
        )

    return {
        "id": example.get("id"),
        "method": "icd",
        "image": str(image_path),
        "question": example["question"],
        "attribute": example.get("attribute"),
        "choices": choices,
        "answer": example.get("answer"),
        "prediction": prediction,
        "raw_output": raw_output,
        "confidence": confidence,
    }
