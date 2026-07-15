import math

import torch

from src.icd_core import (
    inject_disturbance_instruction,
    instruction_contrastive_logits,
)


def test_icd_formula_and_apc():
    normal = torch.tensor([[4.0, 3.0, 0.0]])
    disturbed = torch.tensor([[1.0, 2.0, 1.0]])

    output = instruction_contrastive_logits(
        normal,
        disturbed,
        alpha=1.0,
        beta=0.1,
    )

    expected = 2.0 * normal - disturbed
    cutoff = normal.max() + math.log(0.1)

    assert torch.allclose(
        output[:, :2],
        expected[:, :2],
    )
    assert normal[0, 2] < cutoff
    assert torch.isneginf(output[0, 2])


def test_disturbance_prompt_injection():
    prompt = "USER: <image>\nQuestion\nASSISTANT:"
    disturbed = inject_disturbance_instruction(
        prompt,
        "Confused visual description.",
    )

    assert disturbed.startswith(
        "Confused visual description."
    )
    assert prompt in disturbed
