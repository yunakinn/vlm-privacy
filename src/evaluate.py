from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("annotations.json must contain a list of examples.")
    return data


def validate_example(example: dict[str, Any], index: int) -> None:
    required = ["image", "question"]
    missing = [key for key in required if key not in example]
    if missing:
        raise ValueError(f"Example {index} is missing required keys: {missing}")


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"^answer:\s*", "", text)
    text = re.sub(r"confidence:\s*[0-9.]+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_choice(prediction: str, choices: list[str] | None) -> str:
    if not choices:
        return normalize_text(prediction)

    first_line = prediction.strip().splitlines()[0] if prediction.strip() else ""
    letter_match = re.search(r"\b([A-Z])\b", first_line)
    if letter_match:
        idx = ord(letter_match.group(1)) - ord("A")
        if 0 <= idx < len(choices):
            return normalize_text(choices[idx])

    normalized = normalize_text(prediction)
    for choice in choices:
        if normalize_text(choice) in normalized:
            return normalize_text(choice)
    return normalized


def is_correct(record: dict[str, Any]) -> bool | None:
    answer = record.get("answer")
    if answer is None:
        return None
    pred = extract_choice(record.get("prediction", ""), record.get("choices"))
    return pred == normalize_text(answer) or normalize_text(answer) in pred


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["method"]].append(record)

    summary: dict[str, Any] = {"num_predictions": len(records), "methods": {}}
    for method, method_records in groups.items():
        correctness = [is_correct(record) for record in method_records]
        scored = [value for value in correctness if value is not None]
        avg_conf = [
            record.get("confidence", {}).get("avg_token_prob")
            for record in method_records
            if record.get("confidence", {}).get("avg_token_prob") is not None
        ]
        wrong_conf = [
            record.get("confidence", {}).get("avg_token_prob")
            for record, correct in zip(method_records, correctness)
            if correct is False and record.get("confidence", {}).get("avg_token_prob") is not None
        ]

        summary["methods"][method] = {
            "num_predictions": len(method_records),
            "num_scored": len(scored),
            "accuracy": sum(scored) / len(scored) if scored else None,
            "avg_token_confidence": sum(avg_conf) / len(avg_conf) if avg_conf else None,
            "wrong_answer_avg_confidence": sum(wrong_conf) / len(wrong_conf) if wrong_conf else None,
        }

    return summary


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
