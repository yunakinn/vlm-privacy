from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .model_loader import (
    ModelBundle,
    decode_generated,
    load_image,
    prepare_inputs,
)
from .multipriv_inference import build_multipriv_prompt
from .vcd_core import stable_example_seed
from .vcd_inference import VCDConfig, generate_official_vcd


@torch.inference_mode()
def run_multipriv_vcd(
    bundle: ModelBundle,
    example: dict[str, Any],
    repo_root: str | Path,
    config: VCDConfig,
) -> dict[str, Any]:
    image_path = Path(example["image"])

    if not image_path.is_absolute():
        image_path = Path(repo_root) / image_path

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = load_image(image_path)
    prompt = build_multipriv_prompt(example["question"])

    inputs = prepare_inputs(
        bundle=bundle,
        image=image,
        prompt=prompt,
    )

    example_seed = stable_example_seed(
        config.seed,
        example.get("id", str(image_path)),
    )

    start_time = time.perf_counter()

    generated_ids, token_probs = generate_official_vcd(
        bundle=bundle,
        inputs=inputs,
        config=config,
        example_seed=example_seed,
    )

    latency_seconds = time.perf_counter() - start_time

    raw_output = decode_generated(
        bundle,
        generated_ids,
    )
    prediction = raw_output.strip()

    confidence = {
        "avg_token_prob": (
            sum(token_probs) / len(token_probs)
            if token_probs else 0.0
        ),
        "min_token_prob": (
            min(token_probs)
            if token_probs else 0.0
        ),
        "vcd_alpha": config.alpha,
        "vcd_beta": config.beta,
        "vcd_noise_step": config.noise_step,
        "vcd_temperature": config.temperature,
        "vcd_top_p": config.top_p,
        "vcd_top_k": config.top_k,
        "vcd_seed": example_seed,
        "vcd_implementation": (
            "official_vcd_algorithm_huggingface_port"
        ),
        "official_vcd_reproduction": True,
    }

    return {
        "id": example["id"],
        "person_id": example.get("person_id"),
        "method": "vcd",
        "task": example.get("task"),
        "task_group": example.get("task_group"),
        "privacy_category": example.get(
            "privacy_category"
        ),
        "source_language": example.get(
            "source_language"
        ),
        "answer_type": example.get("answer_type"),
        "image": example["image"],
        "question": example["question"],
        "gold_field": example.get("gold_field"),
        "gold_answer": example.get("gold_answer"),
        "prediction": prediction,
        "raw_output": raw_output,
        "confidence": confidence,
        "latency_seconds": latency_seconds,
    }
