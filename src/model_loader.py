from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)


DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


@dataclass
class ModelBundle:
    model: LlavaForConditionalGeneration
    processor: Any
    device: torch.device
    dtype: torch.dtype


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16

    return torch.float32


def load_llava_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    load_in_4bit: bool = False,
) -> ModelBundle:
    """Load LLaVA-1.5 through Hugging Face Transformers."""
    target_device = pick_device(device)
    dtype = pick_dtype(target_device)

    # 기존 slow processor 동작을 명시적으로 유지한다.
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=False,
    )

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        if target_device.type != "cuda":
            raise ValueError(
                "4-bit loading requires a CUDA device."
            )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )

        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        **kwargs,
    )

    if not load_in_4bit:
        model.to(target_device)

    model.eval()

    return ModelBundle(
        model=model,
        processor=processor,
        device=target_device,
        dtype=dtype,
    )


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_privacy_prompt(
    question: str,
    choices: list[str] | None = None,
    ask_confidence: bool = False,
) -> str:
    if choices:
        formatted = "\n".join(
            f"- {choice}"
            for choice in choices
        )
        choice_text = f"\nOptions:\n{formatted}"
    else:
        choice_text = ""

    return (
        "USER: <image>\n"
        "Use only visible evidence in the image. "
        "If there is not enough evidence, choose unknown.\n"
        f"Question: {question}"
        f"{choice_text}\n"
        "Choose exactly one option. "
        "Return only the exact option text, "
        "with no explanation and no confidence score.\n"
        "ASSISTANT:"
    )


def prepare_inputs(
    bundle: ModelBundle,
    image: Image.Image,
    prompt: str,
) -> dict[str, torch.Tensor]:
    inputs = bundle.processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    prepared = {}

    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue

        if value.is_floating_point():
            prepared[key] = value.to(
                device=bundle.device,
                dtype=bundle.dtype,
            )
        else:
            prepared[key] = value.to(bundle.device)

    return prepared


def decode_generated(
    bundle: ModelBundle,
    generated_ids: torch.Tensor,
) -> str:
    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)

    return bundle.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()
