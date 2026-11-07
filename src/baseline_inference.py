from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .choice_scoring import score_baseline_choices
from .evaluate import extract_choice, normalize_text
from .vcd_core import stable_example_seed
from .model_loader import (
    ModelBundle,
    build_privacy_prompt,
    decode_generated,
    load_image,
    prepare_inputs,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 42


def _token_confidence(
    scores: tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
) -> dict[str, float]:
    if not scores or generated_ids.numel() == 0:
        return {
            "avg_token_prob": 0.0,
            "min_token_prob": 0.0,
        }

    probs = []

    for step_scores, token_id in zip(scores, generated_ids):
        step_probs = torch.softmax(step_scores[0].float(), dim=-1)
        probs.append(float(step_probs[token_id].detach().cpu()))

    return {
        "avg_token_prob": sum(probs) / len(probs),
        "min_token_prob": min(probs),
    }


def _parse_prediction(text: str, choices: list[str] | None) -> str:
    if not choices:
        return text.strip()
    prediction = extract_choice(text, choices)
    for choice in choices:
        if normalize_text(choice) == prediction:
            return choice
    return "parse_error"


@contextmanager
def generation_seed(device, seed, example_id):
    """Keep per-example sampling reproducible, including resumed runs."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        value = stable_example_seed(seed, example_id)
        torch.manual_seed(value)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(value)
        yield


@torch.inference_mode()
def run_baseline_example(
    bundle: ModelBundle,
    example: dict[str, Any],
    image_dir: str | Path,
    config: GenerationConfig,
) -> dict[str, Any]:
    image_path = Path(example["image"])

    if not image_path.is_absolute():
        image_path = Path(image_dir) / image_path

    image = load_image(image_path)

    prompt = build_privacy_prompt(
        example["question"],
        example.get("choices"),
    )

    inputs = prepare_inputs(bundle, image, prompt)
    prompt_len = inputs["input_ids"].shape[-1]

    do_sample = config.temperature > 0

    with generation_seed(bundle.device, config.seed, example.get("id", str(image_path))):
        outputs = bundle.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=do_sample,
            temperature=config.temperature if do_sample else None,
            top_p=config.top_p if do_sample else None,
            top_k=config.top_k if do_sample else None,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=bundle.processor.tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0, prompt_len:]
    raw_output = decode_generated(bundle, generated_ids)

    prediction = _parse_prediction(
        raw_output,
        example.get("choices"),
    )

    confidence = _token_confidence(
        outputs.scores,
        generated_ids,
    )

    choices = example.get("choices")
    if choices:
        confidence.update(
            score_baseline_choices(
                bundle=bundle,
                image=image,
                question=example["question"],
                choices=choices,
            )
        )

    print("RAW MODEL OUTPUT:", repr(raw_output))
    print("PARSED PREDICTION:", prediction)

    return {
        "id": example.get("id"),
        "method": "baseline",
        "image": str(image_path),
        "question": example["question"],
        "attribute": example.get("attribute"),
        "choices": example.get("choices"),
        "answer": example.get("answer"),
        "prediction": prediction,
        "raw_output": raw_output,
        "confidence": confidence,
    }
