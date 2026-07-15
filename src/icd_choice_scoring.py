from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .choice_scoring import (
    _cyclic_permutations,
    _format_result,
    _get_letter_probabilities,
    build_choice_scoring_prompt,
)
from .icd_core import (
    inject_disturbance_instruction,
    instruction_contrastive_logits,
)
from .model_loader import ModelBundle, prepare_inputs


@torch.inference_mode()
def score_icd_choices(
    bundle: ModelBundle,
    image: Image.Image,
    question: str,
    choices: list[str],
    *,
    alpha: float,
    beta: float,
    disturbance_instruction: str,
) -> dict[str, Any]:
    if not choices:
        raise ValueError("ICD choice scoring requires choices.")

    probability_sums = {
        choice: 0.0
        for choice in choices
    }

    permutations = _cyclic_permutations(choices)

    for ordered_choices in permutations:
        normal_prompt = build_choice_scoring_prompt(
            question=question,
            ordered_choices=ordered_choices,
        )

        disturbed_prompt = inject_disturbance_instruction(
            normal_prompt,
            disturbance_instruction,
        )

        normal_inputs = prepare_inputs(
            bundle=bundle,
            image=image,
            prompt=normal_prompt,
        )

        disturbed_inputs = prepare_inputs(
            bundle=bundle,
            image=image,
            prompt=disturbed_prompt,
        )

        normal_outputs = bundle.model(**normal_inputs)
        disturbed_outputs = bundle.model(**disturbed_inputs)

        icd_logits = instruction_contrastive_logits(
            normal_outputs.logits[:, -1, :],
            disturbed_outputs.logits[:, -1, :],
            alpha=alpha,
            beta=beta,
        )

        letter_probabilities = _get_letter_probabilities(
            bundle=bundle,
            prompt=normal_prompt,
            logits=icd_logits,
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
            "official_icd_formula_apc_"
            "conditional_on_option_letters"
        ),
    )

    result.update({
        "choice_icd_alpha": alpha,
        "choice_icd_beta": beta,
        "choice_icd_disturbance_instruction":
            disturbance_instruction,
    })

    return result
