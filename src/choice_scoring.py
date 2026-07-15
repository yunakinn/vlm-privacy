from __future__ import annotations

from string import ascii_uppercase
from typing import Any

import torch
from PIL import Image

from .model_loader import ModelBundle, prepare_inputs
from .vcd_core import (
    add_diffusion_noise,
    make_generator,
    official_vcd_logits,
)


def build_choice_scoring_prompt(
    question: str,
    ordered_choices: list[str],
) -> str:
    if not ordered_choices:
        raise ValueError("Choice scoring requires choices.")

    formatted_choices = "\n".join(
        f"{ascii_uppercase[index]}. {choice}"
        for index, choice in enumerate(ordered_choices)
    )

    has_unknown = any(
        choice.strip().lower() == "unknown"
        for choice in ordered_choices
    )
    unknown_instruction = (
        "If there is not enough evidence, choose unknown.\n"
        if has_unknown
        else ""
    )

    return (
        "USER: <image>\n"
        "Use only visible evidence in the image.\n"
        f"{unknown_instruction}"
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        "Choose exactly one option. "
        "Return only the single option letter.\n"
        "ASSISTANT:\n"
    )


def _cyclic_permutations(
    choices: list[str],
) -> list[list[str]]:
    return [
        choices[shift:] + choices[:shift]
        for shift in range(len(choices))
    ]


def _get_letter_token_ids(
    bundle: ModelBundle,
    prompt: str,
    num_choices: int,
) -> list[int]:
    tokenizer = bundle.processor.tokenizer
    letters = list(ascii_uppercase[:num_choices])

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    ).input_ids

    token_ids: list[int] = []

    for letter in letters:
        full_ids = tokenizer(
            prompt + letter,
            add_special_tokens=True,
        ).input_ids

        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"Prompt tokenization changed for letter {letter}."
            )

        continuation = full_ids[len(prompt_ids):]

        if len(continuation) != 1:
            raise RuntimeError(
                f"Letter {letter} is not one token: "
                f"{continuation}"
            )

        token_ids.append(continuation[0])

    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError(
            "Option letters do not have unique token IDs."
        )

    return token_ids


def _get_letter_probabilities(
    bundle: ModelBundle,
    prompt: str,
    logits: torch.Tensor,
    num_choices: int,
) -> torch.Tensor:
    token_ids = _get_letter_token_ids(
        bundle=bundle,
        prompt=prompt,
        num_choices=num_choices,
    )

    if logits.dim() == 2:
        logits = logits[0]

    indices = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=logits.device,
    )
    selected_logits = logits.float().index_select(
        dim=-1,
        index=indices,
    )

    if not torch.isfinite(selected_logits).any():
        raise RuntimeError(
            "Official VCD APC removed every option-letter token. "
            "Inspect the prompt or lower beta."
        )

    # Conditional probability among the allowed answer letters.
    return torch.softmax(selected_logits, dim=-1)


def _format_result(
    choices: list[str],
    probability_sums: dict[str, float],
    num_permutations: int,
    *,
    implementation: str,
) -> dict[str, Any]:
    probabilities = {
        choice: probability_sums[choice] / num_permutations
        for choice in choices
    }

    total = sum(probabilities.values())
    if total <= 0:
        raise RuntimeError(
            "Aggregated probabilities have zero total."
        )

    probabilities = {
        choice: probability / total
        for choice, probability in probabilities.items()
    }
    prediction = max(
        choices,
        key=lambda choice: probabilities[choice],
    )

    return {
        "choice_prediction": prediction,
        "choice_confidence": probabilities[prediction],
        "choice_probabilities": probabilities,
        "choice_scoring_method":
            "balanced_cyclic_permutation_letter_probability",
        "choice_num_permutations": num_permutations,
        "choice_vcd_implementation": implementation,
    }


@torch.inference_mode()
def score_baseline_choices(
    bundle: ModelBundle,
    image: Image.Image,
    question: str,
    choices: list[str],
) -> dict[str, Any]:
    if not choices:
        raise ValueError(
            "Baseline choice scoring requires choices."
        )

    probability_sums = {
        choice: 0.0
        for choice in choices
    }
    permutations = _cyclic_permutations(choices)

    for ordered_choices in permutations:
        prompt = build_choice_scoring_prompt(
            question=question,
            ordered_choices=ordered_choices,
        )
        inputs = prepare_inputs(
            bundle=bundle,
            image=image,
            prompt=prompt,
        )
        outputs = bundle.model(**inputs)
        logits = outputs.logits[:, -1, :]

        letter_probabilities = _get_letter_probabilities(
            bundle=bundle,
            prompt=prompt,
            logits=logits,
            num_choices=len(ordered_choices),
        )

        for position, choice in enumerate(ordered_choices):
            probability_sums[choice] += float(
                letter_probabilities[position]
                .detach()
                .cpu()
            )

    return _format_result(
        choices=choices,
        probability_sums=probability_sums,
        num_permutations=len(permutations),
        implementation="baseline_clean_logits",
    )


@torch.inference_mode()
def score_vcd_choices(
    bundle: ModelBundle,
    clean_image: Image.Image,
    question: str,
    choices: list[str],
    alpha: float,
    beta: float = 0.1,
    noise_step: int = 500,
    seed: int = 42,
    # Kept only so older callers fail softly if they still pass it by name.
    distorted_image: Image.Image | None = None,
) -> dict[str, Any]:
    """
    Score answer letters using the official VCD next-token distribution.

    The full-vocabulary Adaptive Plausibility Constraint is applied first;
    probabilities are then conditioned on the allowed option-letter tokens.
    """
    del distorted_image

    if not choices:
        raise ValueError(
            "VCD choice scoring requires choices."
        )

    probability_sums = {
        choice: 0.0
        for choice in choices
    }
    permutations = _cyclic_permutations(choices)

    for permutation_index, ordered_choices in enumerate(
        permutations
    ):
        prompt = build_choice_scoring_prompt(
            question=question,
            ordered_choices=ordered_choices,
        )
        clean_inputs = prepare_inputs(
            bundle=bundle,
            image=clean_image,
            prompt=prompt,
        )

        clean_pixel_values = clean_inputs["pixel_values"]
        generator = make_generator(
            clean_pixel_values,
            seed + permutation_index,
        )
        distorted_pixel_values = add_diffusion_noise(
            clean_pixel_values,
            noise_step,
            generator=generator,
        )

        clean_outputs = bundle.model(**clean_inputs)

        distorted_inputs = dict(clean_inputs)
        distorted_inputs["pixel_values"] = (
            distorted_pixel_values
        )
        distorted_outputs = bundle.model(
            **distorted_inputs
        )

        vcd_logits = official_vcd_logits(
            clean_outputs.logits[:, -1, :],
            distorted_outputs.logits[:, -1, :],
            alpha=alpha,
            beta=beta,
        )

        letter_probabilities = _get_letter_probabilities(
            bundle=bundle,
            prompt=prompt,
            logits=vcd_logits,
            num_choices=len(ordered_choices),
        )

        for position, choice in enumerate(ordered_choices):
            probability_sums[choice] += float(
                letter_probabilities[position]
                .detach()
                .cpu()
            )

    result = _format_result(
        choices=choices,
        probability_sums=probability_sums,
        num_permutations=len(permutations),
        implementation=(
            "official_vcd_formula_diffusion_noise_apc_"
            "conditional_on_option_letters"
        ),
    )
    result.update(
        {
            "choice_vcd_alpha": alpha,
            "choice_vcd_beta": beta,
            "choice_vcd_noise_step": noise_step,
        }
    )
    return result
