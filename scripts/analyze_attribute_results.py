from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {path}"
                ) from error

    return records


def flatten_record(record: dict) -> dict:
    confidence = record.get("confidence") or {}

    choice_prediction = confidence.get(
        "choice_prediction"
    )

    prediction = (
        choice_prediction
        if choice_prediction is not None
        else record.get("prediction")
    )

    choice_probabilities = confidence.get(
        "choice_probabilities"
    ) or {}

    answer = record.get("answer")

    return {
        "id": record.get("id"),
        "method": record.get("method"),
        "attribute": record.get("attribute"),
        "answer": answer,
        "prediction": prediction,
        "correct": prediction == answer,
        "choice_confidence": confidence.get(
            "choice_confidence"
        ),
        "ground_truth_probability": (
            choice_probabilities.get(answer)
        ),
        "avg_token_prob": confidence.get(
            "avg_token_prob"
        ),
        "min_token_prob": confidence.get(
            "min_token_prob"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    records = read_jsonl(args.predictions)

    if not records:
        raise RuntimeError(
            "Prediction file contains no records."
        )

    dataframe = pd.DataFrame(
        flatten_record(record)
        for record in records
    )

    summary = (
        dataframe
        .groupby(
            ["attribute", "method"],
            dropna=False,
        )
        .agg(
            samples=("correct", "size"),
            accuracy=("correct", "mean"),
            avg_confidence=(
                "choice_confidence",
                "mean",
            ),
            avg_gt_probability=(
                "ground_truth_probability",
                "mean",
            ),
            avg_token_probability=(
                "avg_token_prob",
                "mean",
            ),
        )
        .reset_index()
    )

    wrong_summary = (
        dataframe[
            dataframe["correct"] == False
        ]
        .groupby(
            ["attribute", "method"],
            dropna=False,
        )
        .agg(
            wrong_count=("correct", "size"),
            wrong_confidence=(
                "choice_confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        wrong_summary,
        on=["attribute", "method"],
        how="left",
    )

    summary["wrong_count"] = (
        summary["wrong_count"]
        .fillna(0)
        .astype(int)
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.output_dir / "predictions_flat.csv",
        index=False,
    )

    summary.to_csv(
        args.output_dir / "attribute_summary.csv",
        index=False,
    )

    print(summary.to_string(index=False))
    print()
    print(
        "Saved:",
        args.output_dir / "attribute_summary.csv",
    )


if __name__ == "__main__":
    main()
