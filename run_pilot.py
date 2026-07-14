from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.evaluate import (
    load_annotations,
    summarize_predictions,
    validate_example,
    write_json,
    write_jsonl,
)

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small VLM privacy inference pilot."
    )

    parser.add_argument(
        "--annotations",
        default="data/annotations.json",
    )
    parser.add_argument(
        "--image-dir",
        default="data/images",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "vcd"],
        choices=["baseline", "vcd"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
    )
    parser.add_argument(
        "--vcd-alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vcd-distortion",
        default="blur",
        choices=["blur", "noise"],
    )
    parser.add_argument(
        "--vcd-blur-radius",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--vcd-noise-std",
        type=float,
        default=0.25,
    )

    return parser.parse_args()


def print_empty_dataset_help() -> None:
    print("No examples found in data/annotations.json.")
    print("Add examples like this:")
    print(
        """
[
  {
    "id": "pilot_001",
    "image": "example.jpg",
    "attribute": "age_group",
    "question": "What age group does the person appear to belong to?",
    "choices": [
      "child",
      "teen",
      "young adult",
      "middle-aged adult",
      "older adult",
      "unknown"
    ],
    "answer": "unknown"
  }
]
""".strip()
    )


def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_dataset_metadata(
    annotations_path: str | Path,
) -> dict[str, Any] | None:
    metadata_path = (
        Path(annotations_path).resolve().parent
        / "dataset_metadata.json"
    )

    if not metadata_path.exists():
        return None

    try:
        return json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return None


def build_run_metadata(
    args: argparse.Namespace,
    num_examples: int,
    bundle_device: torch.device,
    bundle_dtype: torch.dtype,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_metadata = load_dataset_metadata(
        args.annotations
    )

    first_confidence = (
        records[0].get("confidence", {})
        if records
        else {}
    )

    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_commit": get_git_commit(),
        "model": {
            "model_id": args.model_id,
            "device": str(bundle_device),
            "dtype": str(bundle_dtype),
            "load_in_4bit": args.load_in_4bit,
            "processor_use_fast": False,
        },
        "experiment": {
            "methods": args.methods,
            "num_examples": num_examples,
            "num_predictions": len(records),
            "max_new_tokens": args.max_new_tokens,
            "annotations_path": str(
                Path(args.annotations).resolve()
            ),
            "image_dir": str(
                Path(args.image_dir).resolve()
            ),
        },
        "method_implementations": {
            "baseline": {
                "name": "llava_greedy_baseline",
            },
            "vcd": {
                "name": "simplified_vcd",
                "official_vcd_reproduction": False,
                "description": (
                    "Contrastive decoding using clean and "
                    "distorted image logits."
                ),
                "formula": (
                    "(1 + alpha) * clean_logits "
                    "- alpha * distorted_logits"
                ),
                "alpha": args.vcd_alpha,
                "distortion": args.vcd_distortion,
                "blur_radius": args.vcd_blur_radius,
                "noise_std": args.vcd_noise_std,
            },
        },
        "choice_scoring": {
            "method": first_confidence.get(
                "choice_scoring_method"
            ),
            "num_permutations": first_confidence.get(
                "choice_num_permutations"
            ),
        },
        "dataset": dataset_metadata,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version(
                "transformers"
            ),
            "accelerate": package_version(
                "accelerate"
            ),
            "bitsandbytes": package_version(
                "bitsandbytes"
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def main() -> None:
    args = parse_args()

    examples = load_annotations(args.annotations)

    for index, example in enumerate(examples):
        validate_example(example, index)

    if args.limit is not None:
        examples = examples[: args.limit]

    if not examples:
        print_empty_dataset_help()
        return

    from src.baseline_inference import (
        GenerationConfig,
        run_baseline_example,
    )
    from src.model_loader import load_llava_model
    from src.vcd_inference import (
        VCDConfig,
        run_vcd_example,
    )

    bundle = load_llava_model(
        model_id=args.model_id,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )

    baseline_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
    )

    vcd_config = VCDConfig(
        max_new_tokens=args.max_new_tokens,
        alpha=args.vcd_alpha,
        distortion=args.vcd_distortion,
        blur_radius=args.vcd_blur_radius,
        noise_std=args.vcd_noise_std,
    )

    records = []
    total = len(examples) * len(args.methods)

    with tqdm(
        total=total,
        desc="Running pilot",
    ) as progress:
        for example in examples:
            if "baseline" in args.methods:
                records.append(
                    run_baseline_example(
                        bundle,
                        example,
                        args.image_dir,
                        baseline_config,
                    )
                )
                progress.update(1)

            if "vcd" in args.methods:
                records.append(
                    run_vcd_example(
                        bundle,
                        example,
                        args.image_dir,
                        vcd_config,
                    )
                )
                progress.update(1)

    results_dir = Path(args.results_dir)
    predictions_path = (
        results_dir / "pilot_predictions.jsonl"
    )
    metrics_path = (
        results_dir / "pilot_metrics.json"
    )
    metadata_path = (
        results_dir / "run_metadata.json"
    )

    write_jsonl(
        predictions_path,
        records,
    )

    write_json(
        metrics_path,
        summarize_predictions(records),
    )

    run_metadata = build_run_metadata(
        args=args,
        num_examples=len(examples),
        bundle_device=bundle.device,
        bundle_dtype=bundle.dtype,
        records=records,
    )

    write_json(
        metadata_path,
        run_metadata,
    )

    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
