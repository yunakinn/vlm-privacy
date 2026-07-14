from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

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
    """모델 출력에서 실제 선택지만 추출한다."""
    if not choices:
        return text.strip()

    normalized_text = text.strip().lower()

    # "Answer: young adult" 형식 처리
    for line in normalized_text.splitlines():
        if line.startswith("answer:"):
            candidate = line.split(":", 1)[1].strip()

            for choice in choices:
                if candidate == choice.lower():
                    return choice

    # 형식을 지키지 않았더라도 출력 안에 선택지가 있는지 확인
    # young adult / older adult처럼 단어가 겹치므로 긴 선택지부터 검사
    for choice in sorted(choices, key=len, reverse=True):
        if choice.lower() in normalized_text:
            return choice

    return "parse_error"


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
