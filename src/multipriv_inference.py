from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model_loader import (
    ModelBundle,
    decode_generated,
    load_image,
    prepare_inputs,
)


@dataclass
class MultiPrivGenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0


def build_multipriv_prompt(question: str) -> str:
    return (
        "USER: <image>\n"
        "Use only information that is clearly visible in the image.\n"
        f"Question: {question}\n"
        "Return only the requested answer, without explanation.\n"
        "ASSISTANT:"
    )


def token_confidence(
    scores: tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
) -> dict[str, float]:
    if not scores or generated_ids.numel() == 0:
        return {
            "avg_token_prob": 0.0,
            "min_token_prob": 0.0,
        }

    probabilities: list[float] = []

    for step_scores, token_id in zip(scores, generated_ids):
        step_probs = torch.softmax(
            step_scores[0].float(),
            dim=-1,
        )
        probabilities.append(
            float(step_probs[int(token_id)].detach().cpu())
        )

    return {
        "avg_token_prob": (
            sum(probabilities) / len(probabilities)
        ),
        "min_token_prob": min(probabilities),
    }


@torch.inference_mode()
def run_multipriv_baseline(
    bundle: ModelBundle,
    example: dict[str, Any],
    repo_root: str | Path,
    config: MultiPrivGenerationConfig,
) -> dict[str, Any]:
    image_path = Path(example["image"])

    if not image_path.is_absolute():
        image_path = Path(repo_root) / image_path

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = load_image(image_path)
    prompt = build_multipriv_prompt(example["question"])

    inputs = prepare_inputs(bundle, image, prompt)
    prompt_length = inputs["input_ids"].shape[-1]

    do_sample = config.temperature > 0

    start_time = time.perf_counter()

    outputs = bundle.model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=do_sample,
        temperature=(
            config.temperature if do_sample else None
        ),
        top_p=config.top_p if do_sample else None,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=(
            bundle.processor.tokenizer.eos_token_id
        ),
    )

    latency_seconds = time.perf_counter() - start_time

    generated_ids = outputs.sequences[0, prompt_length:]
    raw_output = decode_generated(bundle, generated_ids)

    return {
        "id": example["id"],
        "person_id": example.get("person_id"),
        "method": "baseline",
        "task": example.get("task"),
        "privacy_category": example.get(
            "privacy_category"
        ),
        "image": example["image"],
        "question": example["question"],
        "gold_field": example.get("gold_field"),
        "gold_answer": example.get("gold_answer"),
        "prediction": raw_output.strip(),
        "raw_output": raw_output,
        "confidence": token_confidence(
            outputs.scores,
            generated_ids,
        ),
        "latency_seconds": latency_seconds,
    }
