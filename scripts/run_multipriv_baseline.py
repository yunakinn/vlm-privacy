from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_loader import load_llava_model
from src.multipriv_evaluate import (
    evaluate_record,
    summarize_records,
)
from src.multipriv_inference import (
    MultiPrivGenerationConfig,
    run_multipriv_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MultiPriv baseline inference."
    )

    parser.add_argument(
        "--annotations",
        default=(
            "data/multipriv_pilot/"
            "annotations_pilot30.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/multipriv_pilot/"
            "baseline_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        default=(
            "results/multipriv_pilot/"
            "baseline_summary.json"
        ),
    )
    parser.add_argument(
        "--model-id",
        default="llava-hf/llava-1.5-7b-hf",
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
        "--max-new-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Annotation file must contain a JSON list."
        )

    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


def main() -> None:
    args = parse_args()

    annotation_path = REPO_ROOT / args.annotations
    output_path = REPO_ROOT / args.output
    summary_path = REPO_ROOT / args.summary

    examples = load_json(annotation_path)

    if args.limit is not None:
        examples = examples[: args.limit]

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    existing_records = read_jsonl(output_path)
    completed_ids = {
        record["id"]
        for record in existing_records
        if record.get("id")
    }

    pending_examples = [
        example
        for example in examples
        if example["id"] not in completed_ids
    ]

    print("Examples:", len(examples))
    print("Already completed:", len(completed_ids))
    print("Pending:", len(pending_examples))

    bundle = load_llava_model(
        model_id=args.model_id,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )

    generation_config = MultiPrivGenerationConfig(
        max_new_tokens=args.max_new_tokens,
    )

    for example in tqdm(
        pending_examples,
        desc="MultiPriv baseline",
    ):
        record = run_multipriv_baseline(
            bundle=bundle,
            example=example,
            repo_root=REPO_ROOT,
            config=generation_config,
        )

        record = evaluate_record(record)
        append_jsonl(output_path, record)

        print(
            "\n",
            record["id"],
            "| gold:",
            record["gold_answer"],
            "| pred:",
            record["prediction"],
            "| exact:",
            record["correct_exact"],
        )

    all_records = read_jsonl(output_path)
    summary = summarize_records(all_records)

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved predictions:", output_path)
    print("Saved summary:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
