from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFilter

from .model_loader import ModelBundle, build_privacy_prompt, decode_generated, load_image, prepare_inputs


@dataclass
class VCDConfig:
    max_new_tokens: int = 64
    alpha: float = 1.0
    distortion: str = "blur"
    blur_radius: float = 12.0
    noise_std: float = 0.25
    temperature: float = 0.0
    top_p: float = 1.0


def distort_image(image: Image.Image, config: VCDConfig) -> Image.Image:
    if config.distortion == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))
    if config.distortion == "noise":
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image.height, image.width, 3)
        tensor = tensor / 255.0
        tensor = torch.clamp(tensor + torch.randn_like(tensor) * config.noise_std, 0.0, 1.0)
        data = (tensor * 255.0).byte().reshape(-1, 3).tolist()
        out = Image.new("RGB", image.size)
        out.putdata([tuple(pixel) for pixel in data])
        return out
    raise ValueError(f"Unsupported VCD distortion: {config.distortion}")


def _sample_next(logits: torch.Tensor, config: VCDConfig) -> torch.Tensor:
    if config.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / config.temperature
    probs = torch.softmax(logits, dim=-1)
    if config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= config.top_p
        keep[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs * keep)
        probs = filtered / filtered.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def run_vcd_example(
    bundle: ModelBundle,
    example: dict[str, Any],
    image_dir: str | Path,
    config: VCDConfig,
) -> dict[str, Any]:
    image_path = Path(example["image"])
    if not image_path.is_absolute():
        image_path = Path(image_dir) / image_path

    clean_image = load_image(image_path)
    distorted_image = distort_image(clean_image, config)
    prompt = build_privacy_prompt(example["question"], example.get("choices"))

    clean_inputs = prepare_inputs(bundle, clean_image, prompt)
    distorted_inputs = prepare_inputs(bundle, distorted_image, prompt)

    input_ids = clean_inputs["input_ids"]
    attention_mask = clean_inputs.get("attention_mask", torch.ones_like(input_ids))
    clean_pixel_values = clean_inputs["pixel_values"]
    distorted_pixel_values = distorted_inputs["pixel_values"]

    generated: list[torch.Tensor] = []
    token_probs: list[float] = []
    eos_token_id = bundle.processor.tokenizer.eos_token_id

    for _ in range(config.max_new_tokens):
        clean_outputs = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=clean_pixel_values,
        )
        distorted_outputs = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=distorted_pixel_values,
        )

        clean_logits = clean_outputs.logits[:, -1, :].float()
        distorted_logits = distorted_outputs.logits[:, -1, :].float()
        vcd_logits = (1.0 + config.alpha) * clean_logits - config.alpha * distorted_logits

        next_token = _sample_next(vcd_logits, config)
        step_probs = torch.softmax(vcd_logits, dim=-1)
        token_probs.append(float(step_probs[0, next_token.item()].detach().cpu()))
        generated.append(next_token.squeeze(0))

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        if next_token.item() == eos_token_id:
            break

    if generated:
        generated_ids = torch.cat(generated, dim=-1)
    else:
        generated_ids = torch.empty(0, dtype=torch.long, device=bundle.device)

    text = decode_generated(bundle, generated_ids)
    confidence = {
        "avg_token_prob": sum(token_probs) / len(token_probs) if token_probs else 0.0,
        "min_token_prob": min(token_probs) if token_probs else 0.0,
        "vcd_alpha": config.alpha,
        "vcd_distortion": config.distortion,
    }

    return {
        "id": example.get("id"),
        "method": "vcd",
        "image": str(image_path),
        "question": example["question"],
        "attribute": example.get("attribute"),
        "choices": example.get("choices"),
        "answer": example.get("answer"),
        "prediction": text,
        "confidence": confidence,
    }
