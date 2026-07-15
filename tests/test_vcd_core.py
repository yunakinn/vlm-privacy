import math

import torch

from src.vcd_core import (
    add_diffusion_noise,
    official_vcd_logits,
)


def test_official_vcd_apc_and_formula():
    clean = torch.tensor([[4.0, 3.0, 0.0]])
    distorted = torch.tensor([[1.0, 2.0, 1.0]])

    output = official_vcd_logits(
        clean,
        distorted,
        alpha=1.0,
        beta=0.1,
    )

    expected_unmasked = 2.0 * clean - distorted
    cutoff = clean.max() + math.log(0.1)

    assert torch.allclose(
        output[:, :2],
        expected_unmasked[:, :2],
    )
    assert clean[0, 2] < cutoff
    assert torch.isneginf(output[0, 2])


def test_diffusion_noise_is_seeded():
    image = torch.ones(1, 3, 4, 4)

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    first = add_diffusion_noise(
        image,
        500,
        generator=g1,
    )
    second = add_diffusion_noise(
        image,
        500,
        generator=g2,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, image)
