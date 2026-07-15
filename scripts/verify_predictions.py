from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}"
                ) from error

            records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--expected-methods",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--expected-per-method",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    records = read_jsonl(args.predictions)

    if not records:
        raise RuntimeError(
            "No prediction records found."
        )

    errors = []
    method_counts = Counter()
    attribute_counts = Counter()
    seen_pairs = set()

    for index, record in enumerate(records):
        record_id = record.get("id")
        method = record.get("method")
        attribute = record.get("attribute")
        confidence = record.get("confidence") or {}

        method_counts[method] += 1
        attribute_counts[attribute] += 1

        pair = (record_id, method)

        if pair in seen_pairs:
            errors.append(
                f"Duplicate id/method pair: {pair}"
            )

        seen_pairs.add(pair)

        prediction = confidence.get(
            "choice_prediction",
            record.get("prediction"),
        )

        if prediction in {
            None,
            "",
            "parse_error",
        }:
            errors.append(
                f"Invalid prediction at record {index}: "
                f"id={record_id}, method={method}"
            )

        probabilities = confidence.get(
            "choice_probabilities"
        )

        if probabilities is not None:
            if not isinstance(probabilities, dict):
                errors.append(
                    f"choice_probabilities is not a dict: "
                    f"id={record_id}, method={method}"
                )
                continue

            probability_values = list(
                probabilities.values()
            )

            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in probability_values
            ):
                errors.append(
                    f"NaN or Inf probability: "
                    f"id={record_id}, method={method}"
                )
                continue

            probability_sum = sum(
                probability_values
            )

            if abs(probability_sum - 1.0) >= 1e-4:
                errors.append(
                    f"Probability sum={probability_sum}: "
                    f"id={record_id}, method={method}"
                )

    if args.expected_methods:
        actual_methods = set(method_counts)
        expected_methods = set(
            args.expected_methods
        )

        if actual_methods != expected_methods:
            errors.append(
                f"Method mismatch: "
                f"expected={sorted(expected_methods)}, "
                f"actual={sorted(actual_methods)}"
            )

    if args.expected_per_method is not None:
        for method in (
            args.expected_methods
            or method_counts.keys()
        ):
            actual = method_counts.get(method, 0)

            if actual != args.expected_per_method:
                errors.append(
                    f"{method}: expected "
                    f"{args.expected_per_method}, "
                    f"actual {actual}"
                )

    print("Total records:", len(records))
    print("Methods:", method_counts)
    print("Attributes:", attribute_counts)

    if errors:
        print("\nValidation errors:")

        for error in errors:
            print(" -", error)

        raise SystemExit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
