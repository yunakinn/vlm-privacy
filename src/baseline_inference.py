from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model_loader import ModelBundle, build_privacy_prompt, decode_generated, load_image, prepare_inputs


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


def _token_confidence(scores: tuple[torch.Tensor, ...], generated_ids: torch.Tensor) -> dict[str, float]:
    if not scores or generated_ids.numel() == 0:
        return {"avg_token_prob": 0.0, "min_token_prob": 0.0}

    probs = []
    for step_scores, token_id in zip(scores, generated_ids):
        step_probs = torch.softmax(step_scores[0].float(), dim=-1)
        probs.append(float(step_probs[token_id].detach().cpu()))

    return {
        "avg_token_prob": sum(probs) / len(probs),
        "min_token_prob": min(probs),
    }


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
    prompt = build_privacy_prompt(example["question"], example.get("choices"))
    inputs = prepare_inputs(bundle, image, prompt)
    prompt_len = inputs["input_ids"].shape[-1]

    do_sample = config.temperature > 0
    outputs = bundle.model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=do_sample,
        temperature=config.temperature if do_sample else None,
        top_p=config.top_p if do_sample else None,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=bundle.processor.tokenizer.eos_token_id,
    )

    generated_ids = outputs.sequences[0, prompt_len:]
    text = decode_generated(bundle, generated_ids)
    confidence = _token_confidence(outputs.scores, generated_ids)

    return {
        "id": example.get("id"),
        "method": "baseline",
        "image": str(image_path),
        "question": example["question"],
        "attribute": example.get("attribute"),
        "choices": example.get("choices"),
        "answer": example.get("answer"),
        "prediction": text,
        "confidence": confidence,
    }
