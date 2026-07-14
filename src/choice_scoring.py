from __future__ import annotations

from string import ascii_uppercase
from typing import Any

import torch
from PIL import Image

from .model_loader import ModelBundle, prepare_inputs


def build_choice_scoring_prompt(
    question: str,
    choices: list[str],
) -> str:
    if not choices:
        raise ValueError("Choice scoring requires at least one choice.")

    if len(choices) > len(ascii_uppercase):
        raise ValueError("Choice scoring supports at most 26 choices.")

    formatted_choices = "\n".join(
        f"{ascii_uppercase[index]}. {choice}"
        for index, choice in enumerate(choices)
    )

    return (
        "USER: <image>\n"
        "Use only visible evidence in the image. "
        "If there is not enough evidence, choose unknown.\n"
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        "Return only the single option letter.\n"
        "ASSISTANT:\n"
    )


def _get_choice_token_ids(
    bundle: ModelBundle,
    prompt: str,
    num_choices: int,
) -> tuple[list[str], list[int]]:
    tokenizer = bundle.processor.tokenizer
    letters = list(ascii_uppercase[:num_choices])

    base_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    ).input_ids

    token_ids: list[int] = []

    for letter in letters:
        full_ids = tokenizer(
            prompt + letter,
            add_special_tokens=True,
        ).input_ids

        # 각 선택지 글자가 정확히 하나의 다음 토큰인지 확인한다.
        if full_ids[: len(base_ids)] != base_ids:
            raise RuntimeError(
                f"Prompt tokenization changed while appending choice {letter}."
            )

        continuation = full_ids[len(base_ids) :]

        if len(continuation) != 1:
            raise RuntimeError(
                f"Choice letter {letter} is not one token: {continuation}"
            )

        token_ids.append(continuation[0])

    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError("Choice letters do not have unique token IDs.")

    return letters, token_ids


def _normalize_choice_logits(
    bundle: ModelBundle,
    prompt: str,
    next_token_logits: torch.Tensor,
    choices: list[str],
) -> dict[str, Any]:
    letters, token_ids = _get_choice_token_ids(
        bundle,
        prompt,
        len(choices),
    )

    if next_token_logits.dim() == 2:
        next_token_logits = next_token_logits[0]

    token_index = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=next_token_logits.device,
    )

    allowed_logits = next_token_logits.float().index_select(
        dim=-1,
        index=token_index,
    )

    probabilities = torch.softmax(
        allowed_logits,
        dim=-1,
    )

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "choice_prediction": choices[predicted_index],
        "choice_letter": letters[predicted_index],
        "choice_confidence": float(
            probabilities[predicted_index].detach().cpu()
        ),
        "choice_probabilities": {
            choice: float(probability.detach().cpu())
            for choice, probability in zip(choices, probabilities)
        },
    }


@torch.inference_mode()
def score_baseline_choices(
    bundle: ModelBundle,
    image: Image.Image,
    question: str,
    choices: list[str],
) -> dict[str, Any]:
    prompt = build_choice_scoring_prompt(question, choices)
    inputs = prepare_inputs(bundle, image, prompt)

    outputs = bundle.model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]

    return _normalize_choice_logits(
        bundle,
        prompt,
        next_token_logits,
        choices,
    )


@torch.inference_mode()
def score_vcd_choices(
    bundle: ModelBundle,
    clean_image: Image.Image,
    distorted_image: Image.Image,
    question: str,
    choices: list[str],
    alpha: float,
) -> dict[str, Any]:
    prompt = build_choice_scoring_prompt(question, choices)

    clean_inputs = prepare_inputs(
        bundle,
        clean_image,
        prompt,
    )
    distorted_inputs = prepare_inputs(
        bundle,
        distorted_image,
        prompt,
    )

    clean_outputs = bundle.model(**clean_inputs)
    distorted_outputs = bundle.model(**distorted_inputs)

    clean_logits = clean_outputs.logits[:, -1, :].float()
    distorted_logits = distorted_outputs.logits[:, -1, :].float()

    vcd_logits = (
        (1.0 + alpha) * clean_logits
        - alpha * distorted_logits
    )

    return _normalize_choice_logits(
        bundle,
        prompt,
        vcd_logits,
        choices,
    )
