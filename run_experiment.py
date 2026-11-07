from __future__ import annotations

import argparse
import hashlib
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
    append_jsonl,
    load_annotations,
    read_jsonl,
    summarize_predictions,
    validate_examples,
    write_json,
)

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable VLM privacy inference experiment."
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
        choices=["baseline", "vcd", "icd", "m3id"],
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
        help="Official VCD contrast strength.",
    )
    parser.add_argument(
        "--vcd-beta",
        type=float,
        default=0.1,
        help="Adaptive Plausibility Constraint threshold.",
    )
    parser.add_argument(
        "--vcd-noise-step",
        type=int,
        default=500,
        help="Diffusion-noise step for the distorted image.",
    )
    parser.add_argument(
        "--vcd-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--vcd-top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--vcd-top-k",
        type=int,
        default=0,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--icd-alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--icd-beta",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--icd-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--icd-top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--icd-top-k",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--icd-disturbance-instruction",
        default=(
            "You are a confused object detector who provides only "
            "a vague and incomplete impression of the image."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument("--m3id-beta", type=float, default=0.1)
    parser.add_argument("--m3id-gamma", type=float, default=0.02)
    parser.add_argument("--m3id-step", type=int, default=5)

    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing prediction records.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs and start again.",
    )

    return parser.parse_args()


def print_empty_dataset_help() -> None:
    print("No examples found in the annotation file.")


def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


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
    status: str,
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
        "implementation_revision": "choice_evaluation_v2",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
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
            "num_expected_predictions": (
                num_examples * len(args.methods)
            ),
            "num_saved_predictions": len(records),
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "annotations_path": str(
                Path(args.annotations).resolve()
            ),
            "annotations_sha256": sha256_file(
                args.annotations
            ),
            "image_dir": str(
                Path(args.image_dir).resolve()
            ),
            "resume_enabled": args.resume,
        },
        "method_implementations": {
            "m3id": {
                "name": "m3id_black_image_fixed_step_choice_approximation",
                "exact_official_reproduction": False,
                "beta": args.m3id_beta, "gamma": args.m3id_gamma, "step": args.m3id_step,
                "free_form_generation": "clean_image_generation",
            },
            "baseline": {
                "name": "llava_greedy_baseline",
            },
            "vcd": {
                "name": "official_vcd_algorithm_huggingface_port",
                "official_vcd_reproduction": False,
                "description": (
                    "Official VCD algorithm port using diffusion "
                    "noise, independent clean/distorted KV caches, "
                    "and the Adaptive Plausibility Constraint."
                ),
                "formula": (
                    "(1 + alpha) * clean_logits "
                    "- alpha * distorted_logits"
                ),
                "alpha": args.vcd_alpha,
                "beta": args.vcd_beta,
                "distortion": "diffusion_noise",
                "noise_step": args.vcd_noise_step,
                "temperature": args.vcd_temperature,
                "top_p": args.vcd_top_p,
                "top_k": args.vcd_top_k,
                "distortion_seed": args.seed,
            },
            "icd": {
                "name": "icd_instruction_contrast_choice_port",
                "exact_official_reproduction": False,
                "description": (
                    "Choice scoring contrasts normal and disturbance instructions with APC; "
                    "free-form generation uses the clean image."
                ),
                "formula": (
                    "(1 + alpha) * normal_logits "
                    "- alpha * disturbed_logits"
                ),
                "alpha": args.icd_alpha,
                "beta": args.icd_beta,
                "temperature": args.icd_temperature,
                "top_p": args.icd_top_p,
                "top_k": args.icd_top_k,
                "disturbance_instruction": (
                    args.icd_disturbance_instruction
                ),
            },
        },
        "choice_scoring": {
            "method": first_confidence.get(
                "choice_scoring_method"
            ),
            "num_permutations": first_confidence.get(
                "choice_num_permutations"
            ),
            "probability_interpretation": (
                "Conditional probability over option-letter "
                "tokens after cyclic position balancing."
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



def validate_resume_metadata(
    metadata_path: Path,
    args: argparse.Namespace,
) -> None:
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Cannot safely resume because run_metadata.json is missing. "
            "Use --overwrite to restart the run."
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            "Cannot safely resume because run_metadata.json is invalid."
        ) from error

    experiment = metadata.get("experiment", {})
    model = metadata.get("model", {})
    method_implementations = metadata.get(
        "method_implementations",
        {},
    )
    vcd = method_implementations.get("vcd", {})
    icd = method_implementations.get("icd", {})

    m3id = method_implementations.get("m3id", {})
    expected = {
        "implementation_revision": "choice_evaluation_v2",
        "m3id_beta": args.m3id_beta, "m3id_gamma": args.m3id_gamma, "m3id_step": args.m3id_step,
        "annotations_sha256": sha256_file(args.annotations),
        "methods": set(args.methods),
        "model_id": args.model_id,
        "load_in_4bit": args.load_in_4bit,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "vcd_alpha": args.vcd_alpha,
        "vcd_beta": args.vcd_beta,
        "vcd_noise_step": args.vcd_noise_step,
        "vcd_temperature": args.vcd_temperature,
        "vcd_top_p": args.vcd_top_p,
        "vcd_top_k": args.vcd_top_k,
        "icd_alpha": args.icd_alpha,
        "icd_beta": args.icd_beta,
        "icd_temperature": args.icd_temperature,
        "icd_top_p": args.icd_top_p,
        "icd_top_k": args.icd_top_k,
        "icd_disturbance_instruction": (
            args.icd_disturbance_instruction
        ),
    }

    actual = {
        "implementation_revision": metadata.get("implementation_revision"),
        "m3id_beta": m3id.get("beta"), "m3id_gamma": m3id.get("gamma"), "m3id_step": m3id.get("step"),
        "annotations_sha256": experiment.get("annotations_sha256"),
        "methods": set(experiment.get("methods", [])),
        "model_id": model.get("model_id"),
        "load_in_4bit": model.get("load_in_4bit"),
        "max_new_tokens": experiment.get("max_new_tokens"),
        "seed": experiment.get("seed"),
        "vcd_alpha": vcd.get("alpha"),
        "vcd_beta": vcd.get("beta"),
        "vcd_noise_step": vcd.get("noise_step"),
        "vcd_temperature": vcd.get("temperature"),
        "vcd_top_p": vcd.get("top_p"),
        "vcd_top_k": vcd.get("top_k"),
        "icd_alpha": icd.get("alpha"),
        "icd_beta": icd.get("beta"),
        "icd_temperature": icd.get("temperature"),
        "icd_top_p": icd.get("top_p"),
        "icd_top_k": icd.get("top_k"),
        "icd_disturbance_instruction": (
            icd.get("disturbance_instruction")
        ),
    }

    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if expected[key] != actual[key]
    }

    if mismatches:
        raise ValueError(
            "Resume configuration does not match the existing run: "
            + json.dumps(mismatches, ensure_ascii=False)
            + "\nUse --overwrite to restart with the new settings."
        )

def record_key(record: dict[str, Any]) -> tuple[str, str]:
    example_id = record.get("id")
    method = record.get("method")

    if example_id is None or method is None:
        raise ValueError(
            "Every prediction record must contain id and method."
        )

    return str(example_id), str(method)


def validate_existing_records(
    records: list[dict[str, Any]],
    expected_ids: set[str],
    expected_methods: set[str],
) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()

    for record in records:
        key = record_key(record)

        if key in completed:
            raise ValueError(
                f"Duplicate prediction record found: {key}"
            )

        if key[0] not in expected_ids:
            raise ValueError(
                f"Existing result contains unexpected id: {key[0]}"
            )

        if key[1] not in expected_methods:
            raise ValueError(
                "Existing result contains a method not requested "
                f"for this run: {key[1]}"
            )

        completed.add(key)

    return completed


def main() -> None:
    args = parse_args()

    examples = load_annotations(args.annotations)

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        examples = examples[: args.limit]

    if not examples:
        print_empty_dataset_help()
        return

    validate_examples(examples, args.image_dir)

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

    if args.overwrite:
        for path in (
            predictions_path,
            metrics_path,
            metadata_path,
        ):
            if path.exists():
                path.unlink()

    if predictions_path.exists() and not args.resume:
        raise FileExistsError(
            f"Predictions already exist: {predictions_path}\n"
            "Use --resume to continue or --overwrite to restart."
        )

    if predictions_path.exists() and args.resume:
        validate_resume_metadata(metadata_path, args)

    records = read_jsonl(predictions_path)
    expected_ids = {
        str(example["id"])
        for example in examples
    }
    expected_methods = set(args.methods)
    completed = validate_existing_records(
        records,
        expected_ids,
        expected_methods,
    )

    total = len(examples) * len(args.methods)

    if len(completed) == total:
        write_json(
            metrics_path,
            summarize_predictions(records),
        )
        print(
            f"All {total} predictions are already complete."
        )
        print(f"Metrics refreshed at {metrics_path}")
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
    from src.m3id_inference import (
        M3IDConfig, run_m3id_example,
    )
    from src.icd_inference import (
        ICDConfig,
        run_icd_example,
    )

    bundle = load_llava_model(
        model_id=args.model_id,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )

    baseline_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, seed=args.seed,
    )

    vcd_config = VCDConfig(
        max_new_tokens=args.max_new_tokens,
        alpha=args.vcd_alpha,
        beta=args.vcd_beta,
        noise_step=args.vcd_noise_step,
        temperature=args.vcd_temperature,
        top_p=args.vcd_top_p,
        top_k=args.vcd_top_k,
        seed=args.seed,
    )

    m3id_config = M3IDConfig(max_new_tokens=args.max_new_tokens, seed=args.seed,
                             beta=args.m3id_beta, gamma=args.m3id_gamma, step=args.m3id_step)

    icd_config = ICDConfig(
        max_new_tokens=args.max_new_tokens,
        alpha=args.icd_alpha,
        beta=args.icd_beta,
        temperature=args.icd_temperature,
        top_p=args.icd_top_p,
        top_k=args.icd_top_k,
        seed=args.seed,
        disturbance_instruction=args.icd_disturbance_instruction,
    )


    write_json(
        metadata_path,
        build_run_metadata(
            args=args,
            num_examples=len(examples),
            bundle_device=bundle.device,
            bundle_dtype=bundle.dtype,
            records=records,
            status="running",
        ),
    )

    with tqdm(
        total=total,
        initial=len(completed),
        desc="Running experiment",
    ) as progress:
        for example in examples:
            example_id = str(example["id"])

            if (
                "baseline" in args.methods
                and (example_id, "baseline") not in completed
            ):
                record = run_baseline_example(
                    bundle,
                    example,
                    args.image_dir,
                    baseline_config,
                )
                append_jsonl(predictions_path, record)
                records.append(record)
                completed.add((example_id, "baseline"))
                progress.update(1)

            if (
                "vcd" in args.methods
                and (example_id, "vcd") not in completed
            ):
                record = run_vcd_example(
                    bundle,
                    example,
                    args.image_dir,
                    vcd_config,
                )
                append_jsonl(predictions_path, record)
                records.append(record)
                completed.add((example_id, "vcd"))
                progress.update(1)

            if (
                "icd" in args.methods
                and (example_id, "icd") not in completed
            ):
                record = run_icd_example(
                    bundle,
                    example,
                    args.image_dir,
                    icd_config,
                )
                append_jsonl(predictions_path, record)
                records.append(record)
                completed.add((example_id, "icd"))
                progress.update(1)

            if (
                "m3id" in args.methods
                and (example_id, "m3id") not in completed
            ):
                record = run_m3id_example(
                    bundle,
                    example,
                    args.image_dir,
                    m3id_config,
                )
                append_jsonl(predictions_path, record)
                records.append(record)
                completed.add((example_id, "m3id"))
                progress.update(1)

    write_json(
        metrics_path,
        summarize_predictions(records),
    )

    write_json(
        metadata_path,
        build_run_metadata(
            args=args,
            num_examples=len(examples),
            bundle_device=bundle.device,
            bundle_dtype=bundle.dtype,
            records=records,
            status="completed",
        ),
    )

    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
