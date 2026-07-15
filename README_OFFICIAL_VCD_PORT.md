# Official VCD Hugging Face Port

This package contains the first core-code replacement for the `vlm-privacy`
repository.

## Replace/add these files

- Add `src/vcd_core.py`
- Replace `src/vcd_inference.py`
- Replace `src/choice_scoring.py`

The replacement keeps the old `VCDConfig` distortion fields so the current
`run_pilot.py` can still instantiate the class before its CLI and metadata are
updated.

## What now matches the official repository

- Distortion is diffusion noise applied to preprocessed `pixel_values`.
- Default `noise_step=500`.
- `cd_alpha=1.0`.
- `cd_beta=0.1`.
- Adaptive Plausibility Constraint is applied over the clean-image vocabulary.
- Clean and distorted streams maintain separate KV caches.
- VCD generation samples after temperature/top-p/top-k filtering.
- Per-example seeds are deterministic.

## Important distinction

Generation is a Hugging Face-compatible port of the official VCD algorithm,
not a copy of the old repository's version-specific monkey patch.

The multiple-choice confidence remains a project-specific diagnostic:
the official full-vocabulary VCD distribution is computed first and then
conditioned on the allowed A/B/C/... answer tokens.

## Next required patch

`run_pilot.py` still needs a small CLI/metadata update after confirming the
current post-`57d80ff` file contents:

- add `--vcd-beta`
- add `--vcd-noise-step`
- add `--vcd-temperature`
- add `--vcd-top-p`
- add `--vcd-top-k`
- add `--seed`
- remove blur/noise-std from official-VCD metadata
- label implementation as `official_vcd_algorithm_huggingface_port`

Do not start the 250-image experiment until a 1-2 image generation and
choice-scoring smoke test passes.
