"""
OPERA choice-scoring pipeline.

Unlike score_baseline_choices / score_vcd_choices / score_m3id_choices /
score_icd_choices in src/choice_scoring.py (which score a single next-token
logit distribution over option letters), OPERA operates via native
multi-token beam search with retrospective rollback. Its outputs are
therefore NOT directly comparable to the single-step letter-softmax
confidence used by the other four methods. See project notes: OPERA is
included in Accuracy / McNemar comparisons only; its own decoding
certainty (beam_margin, rollback stats) is analyzed separately.

Design simplification: unlike the cyclic-permutation approach used for
the other methods (to cancel position bias), OPERA scoring here uses a
SINGLE fixed choice order. Each sample already costs two beam-search
generate() calls (OPERA + plain-beam baseline); running every cyclic
permutation would multiply that cost by len(choices). This can be
revisited later if position bias turns out to matter for OPERA specifically.
"""

from __future__ import annotations

import re
from string import ascii_uppercase
from typing import Any, Optional

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_opera_choice_prompt(
    question: str,
    choices: list[str],
) -> str:
    """
    Build a choice-scoring prompt in the format OPERA's LlavaModel wrapper
    expects (<ImageHere> placeholder, USER/ASSISTANT template -- see
    minigpt4/models/llava.py INSTRUCTION_TEMPLATE["llava-1.5"]).
    """
    if not choices:
        raise ValueError("Choice scoring requires choices.")

    formatted_choices = "\n".join(
        f"{ascii_uppercase[index]}. {choice}"
        for index, choice in enumerate(choices)
    )

    has_unknown = any(
        choice.strip().lower() == "unknown"
        for choice in choices
    )
    unknown_instruction = (
        "If there is not enough evidence, choose unknown.\n"
        if has_unknown
        else ""
    )

    question_block = (
        "Use only visible evidence in the image.\n"
        f"{unknown_instruction}"
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        "Choose exactly one option. "
        "Answer with the option letter only, then a period, then the "
        "option text (e.g. \"A. child\")."
    )

    return f"USER: <ImageHere> {question_block} ASSISTANT:"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Ordered by specificity/priority: tried in order, first match wins.
_LETTER_PATTERNS = [
    re.compile(r"^\s*\(?([A-Z])\)?(?:[\.\):,\s]|$)"),
    re.compile(r"\banswer\s*(?:is|:)?\s*\(?([A-Z])\)?\b", re.IGNORECASE),
    re.compile(r"\boption\s*\(?([A-Z])\)?\b", re.IGNORECASE),
    re.compile(r"\b([A-Z])\b"),
]


def parse_choice_letter(
    text: str,
    num_choices: int,
) -> Optional[str]:
    """
    Extract the first valid option letter from OPERA's free-form output.

    Returns None (rather than guessing) if no valid letter is found, so
    invalid outputs can be tracked via is_valid rather than silently
    miscounted.
    """
    if not text:
        return None

    valid_letters = {
        ascii_uppercase[i]
        for i in range(num_choices)
    }

    for pattern in _LETTER_PATTERNS:
        match = pattern.search(text)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    return None


def letter_to_label(
    letter: Optional[str],
    choices: list[str],
) -> Optional[str]:
    if letter is None:
        return None
    index = ascii_uppercase.index(letter)
    if index >= len(choices):
        return None
    return choices[index]


# ---------------------------------------------------------------------------
# Beam extraction
# ---------------------------------------------------------------------------

def extract_beam_results(
    output_text: list[str],
    sequences_scores: Optional[torch.Tensor],
    num_choices: int,
) -> list[dict[str, Any]]:
    """
    Pair each beam's decoded text with its sequence score and parsed letter,
    sorted by score descending (best beam first). If sequences_scores is
    unavailable, order is left as returned (assumed already best-first,
    per HF beam_scorer.finalize convention).
    """
    if sequences_scores is not None:
        scores = sequences_scores.detach().float().cpu().tolist()
    else:
        scores = [None] * len(output_text)

    beams = [
        {
            "text": text,
            "score": score,
            "letter": parse_choice_letter(text, num_choices),
        }
        for text, score in zip(output_text, scores)
    ]

    if sequences_scores is not None:
        beams = sorted(
            beams,
            key=lambda b: b["score"],
            reverse=True,
        )

    return beams


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_opera_choice(
    model,
    vis_processors,
    norm_fn,
    image: Image.Image,
    question: str,
    choices: list[str],
    *,
    num_beams: int = 5,
    max_new_tokens: int = 8,
    scale_factor: float = 50.0,
    threshold: float = 15.0,
    num_attn_candidates: int = 5,
    penalty_weights: float = 1.0,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Score a multiple-choice question under OPERA decoding, plus a matched
    plain-beam-search baseline for comparison.

    Two full generate() calls are made per sample (OPERA, then plain beam
    search with identical num_beams/max_new_tokens). This is intentional:
    OPERA's sequence-level beam scores are only meaningful relative to a
    like-for-like beam-search reference, not to the single-step logits
    used by the other four methods.
    """
    if not choices:
        raise ValueError("OPERA choice scoring requires choices.")
    if len(choices) > 26:
        raise ValueError("More than 26 choices is not supported (A-Z).")

    num_choices = len(choices)
    prompt = build_opera_choice_prompt(question=question, choices=choices)

    raw_image = image.convert("RGB")
    image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    normed_image = norm_fn(image_tensor)

    # --- OPERA decoding ---
    opera_text, opera_raw = model.generate(
        {"image": normed_image, "prompt": prompt},
        use_nucleus_sampling=False,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=max_new_tokens,
        output_attentions=True,
        return_dict_in_generate=True,
        output_scores=True,
        opera_decoding=True,
        scale_factor=scale_factor,
        threshold=threshold,
        num_attn_candidates=num_attn_candidates,
        penalty_weights=penalty_weights,
    )
    opera_beams = extract_beam_results(
        output_text=opera_text,
        sequences_scores=getattr(opera_raw, "sequences_scores", None),
        num_choices=num_choices,
    )
    rollback_count = getattr(opera_raw, "rollback_count", 0)
    rollback_positions = getattr(opera_raw, "rollback_positions", [])

    # --- Plain beam search (same num_beams/max_new_tokens, opera_decoding=False) ---
    plain_text, plain_raw = model.generate(
        {"image": normed_image, "prompt": prompt},
        use_nucleus_sampling=False,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=max_new_tokens,
        output_attentions=False,
        return_dict_in_generate=True,
        output_scores=True,
        opera_decoding=False,
    )
    plain_beams = extract_beam_results(
        output_text=plain_text,
        sequences_scores=getattr(plain_raw, "sequences_scores", None),
        num_choices=num_choices,
    )

    top_beam = opera_beams[0]
    plain_top_beam = plain_beams[0]

    beam_scores = [b["score"] for b in opera_beams if b["score"] is not None]
    beam_margin = (
        beam_scores[0] - beam_scores[1]
        if len(beam_scores) >= 2
        else None
    )

    prediction = top_beam["letter"]
    plain_prediction = plain_top_beam["letter"]

    opera_changed_prediction = (
        prediction is not None
        and plain_prediction is not None
        and prediction != plain_prediction
    )
    changed_with_rollback = (
        rollback_count > 0
        and opera_changed_prediction
    )

    return {
        "prediction": prediction,
        "parsed_label": letter_to_label(prediction, choices),
        "raw_output": top_beam["text"],
        "is_valid": prediction is not None,

        "plain_beam_prediction": plain_prediction,
        "plain_beam_raw_output": plain_top_beam["text"],
        "plain_beam_is_valid": plain_prediction is not None,

        "opera_changed_prediction": opera_changed_prediction,
        "changed_with_rollback": changed_with_rollback,

        "beam_scores": beam_scores,
        "beam_margin": beam_margin,

        "rollback_count": rollback_count,
        "rollback_positions": rollback_positions,

        # Kept for smoke-test debugging; drop before large-scale runs if
        # storage becomes a concern.
        "opera_beams": opera_beams,
        "plain_beams": plain_beams,

        "choice_scoring_method": "opera_single_order_beam_vs_plain_beam",
    }
