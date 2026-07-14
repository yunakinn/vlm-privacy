from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

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


def extract_choice(
    prediction: str,
    choices: list[str] | None,
) -> str:
    normalized = normalize_text(prediction)

    if not choices:
        return normalized

    normalized_choices = {
        normalize_text(choice): normalize_text(choice)
        for choice in choices
    }

    # 이미 선택지와 정확히 일치하는 경우
    if normalized in normalized_choices:
        return normalized

    # A, B, C 형태의 응답 처리
    first_line = prediction.strip().splitlines()[0] if prediction.strip() else ""
    letter_match = re.fullmatch(
        r"\s*([A-Z])(?:[.)])?\s*",
        first_line,
        flags=re.IGNORECASE,
    )

    if letter_match:
        index = ord(letter_match.group(1).upper()) - ord("A")

        if 0 <= index < len(choices):
            return normalize_text(choices[index])

    return normalized


def is_correct(record: dict[str, Any]) -> bool | None:
    answer = record.get("answer")

    if answer is None:
        return None

    prediction = extract_choice(
        record.get("prediction", ""),
        record.get("choices"),
    )

    # 부분 문자열 비교를 하지 않고 정확히 일치해야 정답
    return prediction == normalize_text(answer)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_predictions(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        groups[record["method"]].append(record)

    summary: dict[str, Any] = {
        "num_predictions": len(records),
        "methods": {},
    }

    # 방법별 지표
    for method, method_records in groups.items():
        correctness = [is_correct(record) for record in method_records]

        scored = [
            value
            for value in correctness
            if value is not None
        ]

        all_confidence = []
        correct_confidence = []
        wrong_confidence = []

        class_total = Counter()
        class_correct = Counter()
        prediction_distribution = Counter()
        confusion_matrix: dict[str, Counter[str]] = defaultdict(Counter)

        parse_error_count = 0

        for record, correct in zip(method_records, correctness):
            answer = normalize_text(record.get("answer"))
            prediction = extract_choice(
                record.get("prediction", ""),
                record.get("choices"),
            )

            if prediction == "parse_error":
                parse_error_count += 1

            prediction_distribution[prediction] += 1

            if record.get("answer") is not None:
                class_total[answer] += 1
                confusion_matrix[answer][prediction] += 1

                if correct:
                    class_correct[answer] += 1

            confidence = (
                record.get("confidence", {})
                .get("avg_token_prob")
            )

            if confidence is not None:
                confidence = float(confidence)
                all_confidence.append(confidence)

                if correct is True:
                    correct_confidence.append(confidence)
                elif correct is False:
                    wrong_confidence.append(confidence)

        per_class_accuracy = {
            label: class_correct[label] / class_total[label]
            for label in sorted(class_total)
        }

        summary["methods"][method] = {
            "num_predictions": len(method_records),
            "num_scored": len(scored),
            "num_correct": sum(scored),
            "accuracy": _mean([float(value) for value in scored]),
            "parse_error_count": parse_error_count,
            "avg_token_confidence": _mean(all_confidence),
            "correct_answer_avg_confidence": _mean(correct_confidence),
            "wrong_answer_avg_confidence": _mean(wrong_confidence),
            "per_class_accuracy": per_class_accuracy,
            "prediction_distribution": dict(
                sorted(prediction_distribution.items())
            ),
            "confusion_matrix": {
                answer: dict(sorted(predictions.items()))
                for answer, predictions in sorted(confusion_matrix.items())
            },
        }

    # Baseline과 VCD의 paired comparison
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for record in records:
        example_id = str(record.get("id"))
        by_id[example_id][record["method"]] = record

    paired_ids = [
        example_id
        for example_id, method_records in by_id.items()
        if "baseline" in method_records and "vcd" in method_records
    ]

    if paired_ids:
        both_correct = 0
        baseline_only_correct = 0
        vcd_only_correct = 0
        both_wrong = 0
        prediction_changed = 0

        changed_examples = []

        for example_id in sorted(paired_ids):
            baseline = by_id[example_id]["baseline"]
            vcd = by_id[example_id]["vcd"]

            baseline_correct = bool(is_correct(baseline))
            vcd_correct = bool(is_correct(vcd))

            baseline_prediction = extract_choice(
                baseline.get("prediction", ""),
                baseline.get("choices"),
            )
            vcd_prediction = extract_choice(
                vcd.get("prediction", ""),
                vcd.get("choices"),
            )

            if baseline_correct and vcd_correct:
                both_correct += 1
            elif baseline_correct and not vcd_correct:
                baseline_only_correct += 1
            elif not baseline_correct and vcd_correct:
                vcd_only_correct += 1
            else:
                both_wrong += 1

            if baseline_prediction != vcd_prediction:
                prediction_changed += 1
                changed_examples.append({
                    "id": example_id,
                    "answer": normalize_text(baseline.get("answer")),
                    "baseline_prediction": baseline_prediction,
                    "vcd_prediction": vcd_prediction,
                    "baseline_correct": baseline_correct,
                    "vcd_correct": vcd_correct,
                })

        summary["paired_comparison"] = {
            "num_paired_examples": len(paired_ids),
            "prediction_changed_count": prediction_changed,
            "prediction_changed_rate": prediction_changed / len(paired_ids),
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only_correct,
            "vcd_only_correct": vcd_only_correct,
            "both_wrong": both_wrong,
            "net_vcd_correct_gain": (
                vcd_only_correct - baseline_only_correct
            ),
            "changed_examples": changed_examples,
        }

    return summary


def write_jsonl(
    path: str | Path,
    records: list[dict[str, Any]],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(
    path: str | Path,
    data: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
