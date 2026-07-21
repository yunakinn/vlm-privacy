from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .baseline_inference import GenerationConfig, _parse_prediction, _token_confidence
from .choice_scoring import score_icd_choices
from .model_loader import (
    ModelBundle,
    build_privacy_prompt,
    decode_generated,
    load_image,
    prepare_inputs,
)


@dataclass
class ICDConfig:
    max_new_tokens: int = 64
    alpha: float = 1.0
    beta: float = 0.1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 42
    disturbance_instruction: str = (
        "You are a confused object detector who provides only "
        "a vague and incomplete impression of the image."
    )


@torch.inference_mode()
def run_icd_example(
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
            score_icd_choices(
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
        "method": "icd",
        "image": str(image_path),
        "question": example["question"],
        "attribute": example.get("attribute"),
        "choices": example.get("choices"),
        "answer": example.get("answer"),
        "prediction": prediction,
        "raw_output": raw_output,
        "confidence": confidence,
    }
