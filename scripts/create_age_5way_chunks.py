from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGE_CHOICES = [
    "child",
    "teen",
    "young adult",
    "middle-aged adult",
    "older adult",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a forced-choice five-class age annotation file "
            "and balanced chunks while preserving the original file."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
    )
    parser.add_argument(
        "--source-name",
        default="annotations.json",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=10,
    )
    return parser.parse_args()


def canonical_sha256(data: Any) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1.")

    dataset_dir = Path(args.dataset_dir).resolve()
    source_path = dataset_dir / args.source_name

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source annotation file not found: {source_path}"
        )

    records = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    if not isinstance(records, list) or not records:
        raise ValueError(
            "Source annotation file must contain a non-empty list."
        )

    converted: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        answer = record.get("answer")

        if answer not in AGE_CHOICES:
            raise ValueError(
                f"Record {index} has an unsupported answer: {answer!r}"
            )

        converted_record = dict(record)
        converted_record["choices"] = list(AGE_CHOICES)
        converted_record["evaluation_protocol"] = (
            "forced_choice_age_5way"
        )
        converted.append(converted_record)

    ids = [str(record.get("id")) for record in converted]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids found in source annotations.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in converted:
        groups[record["answer"]].append(record)

    class_counts = {
        label: len(groups[label])
        for label in AGE_CHOICES
    }

    if len(set(class_counts.values())) != 1:
        raise ValueError(
            f"Classes are not balanced: {class_counts}"
        )

    per_class = next(iter(class_counts.values()))
    if per_class % args.num_chunks != 0:
        raise ValueError(
            f"{per_class} samples per class cannot be divided into "
            f"{args.num_chunks} equal chunks."
        )

    per_class_per_chunk = per_class // args.num_chunks
    output_path = dataset_dir / "annotations_5way.json"
    write_json(output_path, converted)

    manifest = []
    all_chunk_ids: list[str] = []

    for chunk_index in range(args.num_chunks):
        chunk_records: list[dict[str, Any]] = []
        start = chunk_index * per_class_per_chunk
        end = start + per_class_per_chunk

        for label in AGE_CHOICES:
            chunk_records.extend(groups[label][start:end])

        chunk_number = chunk_index + 1
        chunk_path = (
            dataset_dir
            / f"annotations_5way_chunk_{chunk_number:02d}.json"
        )
        write_json(chunk_path, chunk_records)

        chunk_ids = [
            str(record["id"])
            for record in chunk_records
        ]
        all_chunk_ids.extend(chunk_ids)

        manifest.append({
            "chunk": chunk_number,
            "annotation_file": chunk_path.name,
            "num_examples": len(chunk_records),
            "class_distribution": dict(
                Counter(
                    record["answer"]
                    for record in chunk_records
                )
            ),
            "annotation_sha256": canonical_sha256(
                chunk_records
            ),
            "ids": chunk_ids,
        })

    if len(all_chunk_ids) != len(converted):
        raise RuntimeError(
            "Chunk records do not cover all annotations."
        )

    if len(set(all_chunk_ids)) != len(converted):
        raise RuntimeError(
            "Chunk records contain duplicate ids."
        )

    protocol = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol_name": "forced_choice_age_5way",
        "source_annotation_file": source_path.name,
        "output_annotation_file": output_path.name,
        "choices": AGE_CHOICES,
        "unknown_option_included": False,
        "num_examples": len(converted),
        "class_distribution": class_counts,
        "num_chunks": args.num_chunks,
        "samples_per_class_per_chunk": per_class_per_chunk,
        "annotation_sha256": canonical_sha256(converted),
    }

    write_json(
        dataset_dir / "age_5way_protocol.json",
        protocol,
    )
    write_json(
        dataset_dir / "chunks_5way_manifest.json",
        manifest,
    )

    print("Created:", output_path)
    print("Examples:", len(converted))
    print("Class distribution:", class_counts)
    print("Chunks:", args.num_chunks)
    print("Annotation SHA256:", protocol["annotation_sha256"])


if __name__ == "__main__":
    main()
