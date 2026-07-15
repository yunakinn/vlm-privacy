from __future__ import annotations

import math

import torch


ICD_DEFAULT_ALPHA = 1.0
ICD_DEFAULT_BETA = 0.1

# 공식 ICD LLaVA 코드에서 사용한 disturbance instruction 중 하나
ICD_DEFAULT_DISTURBANCE_PROMPT = (
    "You are a confused object detector who provides only "
    "a vague and incomplete impression of the image."
)


def inject_disturbance_instruction(
    prompt: str,
    instruction: str,
) -> str:
    """
    Hugging Face LLaVA용 disturbance prompt 포트.

    공식 LLaVA 구현은 custom conversation의 system prompt에
    disturbance instruction을 넣는다. 현재 프로젝트에서는 동일한
    의미의 instruction을 prompt 앞부분에 삽입한다.
    """
    instruction = instruction.strip()

    if not instruction:
        raise ValueError(
            "disturbance instruction must not be empty."
        )

    return f"{instruction}\n{prompt}"


def instruction_contrastive_logits(
    normal_logits: torch.Tensor,
    disturbed_logits: torch.Tensor,
    *,
    alpha: float = ICD_DEFAULT_ALPHA,
    beta: float = ICD_DEFAULT_BETA,
) -> torch.Tensor:
    """
    Official ICD contrastive formula + Adaptive Plausibility Constraint.

    (1 + alpha) * normal_logits - alpha * disturbed_logits
    """
    if normal_logits.shape != disturbed_logits.shape:
        raise ValueError(
            "normal and disturbed logits must have the same shape."
        )

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must be in (0, 1].")

    normal = normal_logits.float()
    disturbed = disturbed_logits.float()

    cutoff = (
        normal.max(dim=-1, keepdim=True).values
        + math.log(float(beta))
    )

    contrasted = (
        (1.0 + float(alpha)) * normal
        - float(alpha) * disturbed
    )

    return contrasted.masked_fill(
        normal < cutoff,
        -float("inf"),
    )
