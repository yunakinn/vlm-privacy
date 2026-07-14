from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


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
        raise ValueError(
            f"Example {index} is missing required keys: {missing}"
        )


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
        normalize_text(choice)
        for choice in choices
    }

    if normalized in normalized_choices:
        return normalized

    first_line = (
        prediction.strip().splitlines()[0]
        if prediction.strip()
        else ""
    )

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

    return prediction == normalize_text(answer)


def get_choice_prediction(
    record: dict[str, Any],
) -> str | None:
    prediction = (
        record.get("confidence", {})
        .get("choice_prediction")
    )

    if prediction is None:
        return None

    return normalize_text(prediction)


def get_choice_confidence(
    record: dict[str, Any],
) -> float | None:
    value = (
        record.get("confidence", {})
        .get("choice_confidence")
    )

    if value is None:
        return None

    return float(value)


def get_choice_probabilities(
    record: dict[str, Any],
) -> dict[str, float] | None:
    stored = (
        record.get("confidence", {})
        .get("choice_probabilities")
    )

    if not stored:
        return None

    return {
        normalize_text(choice): float(probability)
        for choice, probability in stored.items()
    }


def choice_is_correct(
    record: dict[str, Any],
) -> bool | None:
    answer = record.get("answer")
    prediction = get_choice_prediction(record)

    if answer is None or prediction is None:
        return None

    return prediction == normalize_text(answer)


def get_answer_probability(
    record: dict[str, Any],
) -> float | None:
    answer = record.get("answer")
    probabilities = get_choice_probabilities(record)

    if answer is None or probabilities is None:
        return None

    return probabilities.get(normalize_text(answer))


def multiclass_brier_score(
    record: dict[str, Any],
) -> float | None:
    answer = record.get("answer")
    choices = record.get("choices")
    probabilities = get_choice_probabilities(record)

    if (
        answer is None
        or not choices
        or probabilities is None
    ):
        return None

    normalized_answer = normalize_text(answer)
    score = 0.0

    for choice in choices:
        normalized_choice = normalize_text(choice)
        probability = probabilities.get(normalized_choice, 0.0)
        target = 1.0 if normalized_choice == normalized_answer else 0.0
        score += (probability - target) ** 2

    return score


def expected_calibration_error(
    records: list[dict[str, Any]],
    num_bins: int = 10,
) -> float | None:
    samples: list[tuple[float, bool]] = []

    for record in records:
        confidence = get_choice_confidence(record)
        correct = choice_is_correct(record)

        if confidence is not None and correct is not None:
            samples.append((confidence, correct))

    if not samples:
        return None

    bins: list[list[tuple[float, bool]]] = [
        [] for _ in range(num_bins)
    ]

    for confidence, correct in samples:
        bin_index = min(
            int(confidence * num_bins),
            num_bins - 1,
        )
        bins[bin_index].append((confidence, correct))

    ece = 0.0
    total = len(samples)

    for current_bin in bins:
        if not current_bin:
            continue

        bin_confidence = sum(
            confidence
            for confidence, _ in current_bin
        ) / len(current_bin)

        bin_accuracy = sum(
            int(correct)
            for _, correct in current_bin
        ) / len(current_bin)

        bin_weight = len(current_bin) / total
        ece += bin_weight * abs(
            bin_accuracy - bin_confidence
        )

    return ece


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _build_confusion_data(
    records: list[dict[str, Any]],
    prediction_getter: Callable[
        [dict[str, Any]],
        str | None,
    ],
    correctness_getter: Callable[
        [dict[str, Any]],
        bool | None,
    ],
) -> dict[str, Any]:
    class_total = Counter()
    class_correct = Counter()
    prediction_distribution = Counter()
    confusion_matrix: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        answer = record.get("answer")
        prediction = prediction_getter(record)
        correct = correctness_getter(record)

        if prediction is not None:
            prediction_distribution[prediction] += 1

        if answer is None or prediction is None:
            continue

        normalized_answer = normalize_text(answer)

        class_total[normalized_answer] += 1
        confusion_matrix[normalized_answer][prediction] += 1

        if correct is True:
            class_correct[normalized_answer] += 1

    per_class_accuracy = {
        label: class_correct[label] / class_total[label]
        for label in sorted(class_total)
    }

    return {
        "per_class_accuracy": per_class_accuracy,
        "prediction_distribution": dict(
            sorted(prediction_distribution.items())
        ),
        "confusion_matrix": {
            answer: dict(sorted(predictions.items()))
            for answer, predictions in sorted(
                confusion_matrix.items()
            )
        },
    }


def _build_paired_comparison(
    records: list[dict[str, Any]],
    prediction_getter: Callable[
        [dict[str, Any]],
        str | None,
    ],
    correctness_getter: Callable[
        [dict[str, Any]],
        bool | None,
    ],
) -> dict[str, Any] | None:
    by_id: dict[
        str,
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    for record in records:
        example_id = str(record.get("id"))
        by_id[example_id][record["method"]] = record

    paired_ids = []

    for example_id, method_records in by_id.items():
        if (
            "baseline" not in method_records
            or "vcd" not in method_records
        ):
            continue

        baseline_prediction = prediction_getter(
            method_records["baseline"]
        )
        vcd_prediction = prediction_getter(
            method_records["vcd"]
        )

        if (
            baseline_prediction is None
            or vcd_prediction is None
        ):
            continue

        paired_ids.append(example_id)

    if not paired_ids:
        return None

    both_correct = 0
    baseline_only_correct = 0
    vcd_only_correct = 0
    both_wrong = 0
    prediction_changed = 0
    changed_examples = []

    for example_id in sorted(paired_ids):
        baseline = by_id[example_id]["baseline"]
        vcd = by_id[example_id]["vcd"]

        baseline_prediction = prediction_getter(baseline)
        vcd_prediction = prediction_getter(vcd)

        baseline_correct = bool(
            correctness_getter(baseline)
        )
        vcd_correct = bool(
            correctness_getter(vcd)
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
                "answer": normalize_text(
                    baseline.get("answer")
                ),
                "baseline_prediction": baseline_prediction,
                "vcd_prediction": vcd_prediction,
                "baseline_correct": baseline_correct,
                "vcd_correct": vcd_correct,
            })

    return {
        "num_paired_examples": len(paired_ids),
        "prediction_changed_count": prediction_changed,
        "prediction_changed_rate": (
            prediction_changed / len(paired_ids)
        ),
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only_correct,
        "vcd_only_correct": vcd_only_correct,
        "both_wrong": both_wrong,
        "net_vcd_correct_gain": (
            vcd_only_correct - baseline_only_correct
        ),
        "changed_examples": changed_examples,
    }


def summarize_predictions(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        groups[record["method"]].append(record)

    summary: dict[str, Any] = {
        "num_predictions": len(records),
        "methods": {},
    }

    for method, method_records in groups.items():
        generation_correctness = [
            is_correct(record)
            for record in method_records
        ]

        generation_scored = [
            value
            for value in generation_correctness
            if value is not None
        ]

        token_confidences = []
        correct_token_confidences = []
        wrong_token_confidences = []

        for record, correct in zip(
            method_records,
            generation_correctness,
        ):
            token_confidence = (
                record.get("confidence", {})
                .get("avg_token_prob")
            )

            if token_confidence is None:
                continue

            token_confidence = float(token_confidence)
            token_confidences.append(token_confidence)

            if correct is True:
                correct_token_confidences.append(
                    token_confidence
                )
            elif correct is False:
                wrong_token_confidences.append(
                    token_confidence
                )

        generation_confusion = _build_confusion_data(
            records=method_records,
            prediction_getter=lambda record: extract_choice(
                record.get("prediction", ""),
                record.get("choices"),
            ),
            correctness_getter=is_correct,
        )

        choice_correctness = [
            choice_is_correct(record)
            for record in method_records
        ]

        choice_scored = [
            value
            for value in choice_correctness
            if value is not None
        ]

        choice_confidences = []
        correct_choice_confidences = []
        wrong_choice_confidences = []
        answer_probabilities = []
        brier_scores = []

        for record, correct in zip(
            method_records,
            choice_correctness,
        ):
            choice_confidence = get_choice_confidence(
                record
            )

            if choice_confidence is not None:
                choice_confidences.append(
                    choice_confidence
                )

                if correct is True:
                    correct_choice_confidences.append(
                        choice_confidence
                    )
                elif correct is False:
                    wrong_choice_confidences.append(
                        choice_confidence
                    )

            answer_probability = get_answer_probability(
                record
            )

            if answer_probability is not None:
                answer_probabilities.append(
                    answer_probability
                )

            brier_score = multiclass_brier_score(
                record
            )

            if brier_score is not None:
                brier_scores.append(brier_score)

        choice_confusion = _build_confusion_data(
            records=method_records,
            prediction_getter=get_choice_prediction,
            correctness_getter=choice_is_correct,
        )

        parse_error_count = sum(
            extract_choice(
                record.get("prediction", ""),
                record.get("choices"),
            ) == "parse error"
            for record in method_records
        )

        summary["methods"][method] = {
            "num_predictions": len(method_records),

            "generation_metrics": {
                "num_scored": len(generation_scored),
                "num_correct": sum(generation_scored),
                "accuracy": (
                    sum(generation_scored)
                    / len(generation_scored)
                    if generation_scored
                    else None
                ),
                "parse_error_count": parse_error_count,
                "avg_token_confidence": _mean(
                    token_confidences
                ),
                "correct_answer_avg_token_confidence": _mean(
                    correct_token_confidences
                ),
                "wrong_answer_avg_token_confidence": _mean(
                    wrong_token_confidences
                ),
                **generation_confusion,
            },

            "choice_metrics": {
                "num_scored": len(choice_scored),
                "num_correct": sum(choice_scored),
                "accuracy": (
                    sum(choice_scored)
                    / len(choice_scored)
                    if choice_scored
                    else None
                ),
                "avg_choice_confidence": _mean(
                    choice_confidences
                ),
                "correct_answer_avg_choice_confidence": _mean(
                    correct_choice_confidences
                ),
                "wrong_answer_avg_choice_confidence": _mean(
                    wrong_choice_confidences
                ),
                "avg_ground_truth_probability": _mean(
                    answer_probabilities
                ),
                "multiclass_brier_score": _mean(
                    brier_scores
                ),
                "ece_10_bins": expected_calibration_error(
                    method_records,
                    num_bins=10,
                ),
                **choice_confusion,
            },
        }

    generation_paired = _build_paired_comparison(
        records=records,
        prediction_getter=lambda record: extract_choice(
            record.get("prediction", ""),
            record.get("choices"),
        ),
        correctness_getter=is_correct,
    )

    if generation_paired is not None:
        summary["generation_paired_comparison"] = (
            generation_paired
        )

    choice_paired = _build_paired_comparison(
        records=records,
        prediction_getter=get_choice_prediction,
        correctness_getter=choice_is_correct,
    )

    if choice_paired is not None:
        summary["choice_paired_comparison"] = (
            choice_paired
        )

    return summary


def write_jsonl(
    path: str | Path,
    records: list[dict[str, Any]],
) -> None:
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_json(
    path: str | Path,
    data: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )
