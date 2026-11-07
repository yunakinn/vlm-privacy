"""Build paired final metrics from chunked prediction files."""

from __future__ import annotations

import argparse
import sys
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scipy.stats import chi2, wilcoxon
except ImportError as exc:
    raise SystemExit(
        "scipy가 필요합니다. 먼저 `pip install scipy`를 실행하세요."
    ) from exc


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import (normalize_text, read_jsonl, expected_calibration_error, multiclass_brier_score)

RESULTS_ROOT = Path("results")
OUTPUT_JSON = RESULTS_ROOT / "final_metrics.json"
OUTPUT_CSV = RESULTS_ROOT / "final_metrics.csv"

METHODS = ("baseline", "vcd", "m3id", "icd")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/final_experiments.json"
EXPERIMENTS = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


normalize_label = normalize_text
load_jsonl = read_jsonl


def extract_choice_fields(record: dict[str, Any]) -> dict[str, Any]:
    confidence = record.get("confidence") or {}

    answer = normalize_label(record.get("answer"))
    prediction = normalize_label(
        confidence.get("choice_prediction") or record.get("prediction")
    )

    choice_confidence = confidence.get("choice_confidence")
    if choice_confidence is None:
        raise ValueError(
            f"Missing choice_confidence: id={record.get('id')}, "
            f"method={record.get('method')}"
        )

    raw_probabilities = confidence.get("choice_probabilities") or {}
    probabilities = {
        normalize_label(label): float(probability)
        for label, probability in raw_probabilities.items()
    }

    choices = {
        normalize_label(choice)
        for choice in record.get("choices", [])
    }
    classes = sorted(choices | set(probabilities) | {answer})

    return {
        "id": str(record.get("id")),
        "method": str(record.get("method")).lower(),
        "answer": answer,
        "prediction": prediction,
        "correct": prediction == answer,
        "choice_confidence": float(choice_confidence),
        "probabilities": probabilities,
        "classes": classes,
    }


def load_experiment(
    folders: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {
        method: {} for method in METHODS
    }

    for folder in folders:
        path = RESULTS_ROOT / folder / "pilot_predictions.jsonl"

        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")

        for raw_record in load_jsonl(path):
            record = extract_choice_fields(raw_record)
            method = record["method"]
            record_id = record["id"]

            if method not in by_method:
                raise ValueError(
                    f"Unexpected method '{method}' in {path}"
                )

            if record_id in by_method[method]:
                raise ValueError(
                    f"Duplicate id '{record_id}' for method '{method}' "
                    f"in experiment folders {folders}"
                )

            by_method[method][record_id] = record

    return by_method


def _evaluation_record(record):
    return {"answer": record["answer"], "choices": record["classes"], "confidence": {
        "choice_prediction": record["prediction"], "choice_confidence": record["choice_confidence"],
        "choice_probabilities": record["probabilities"],
    }}


def calculate_ece(records, num_bins=10):
    return expected_calibration_error([_evaluation_record(r) for r in records], num_bins)


def calculate_brier(records):
    return mean(multiclass_brier_score(_evaluation_record(r)) for r in records)


def calculate_method_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    correct_count = sum(record["correct"] for record in records)

    ground_truth_probabilities = [
        record["probabilities"].get(record["answer"], 0.0)
        for record in records
    ]

    return {
        "num_examples": len(records),
        "num_correct": correct_count,
        "accuracy": correct_count / len(records),
        "avg_choice_confidence": mean(
            record["choice_confidence"] for record in records
        ),
        "avg_ground_truth_probability": mean(
            ground_truth_probabilities
        ),
        "multiclass_brier_score": calculate_brier(records),
        "ece_10_bins": calculate_ece(records),
    }


def paired_comparison(
    baseline_records: list[dict[str, Any]],
    method_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {
        record["id"]: record for record in baseline_records
    }
    method_by_id = {
        record["id"]: record for record in method_records
    }

    paired_ids = sorted(set(baseline_by_id) & set(method_by_id))

    baseline_only_correct = 0
    method_only_correct = 0
    both_correct = 0
    both_wrong = 0

    baseline_confidences: list[float] = []
    method_confidences: list[float] = []

    for record_id in paired_ids:
        baseline_record = baseline_by_id[record_id]
        method_record = method_by_id[record_id]

        baseline_correct = baseline_record["correct"]
        method_correct = method_record["correct"]

        if baseline_correct and method_correct:
            both_correct += 1
        elif baseline_correct:
            baseline_only_correct += 1
        elif method_correct:
            method_only_correct += 1
        else:
            both_wrong += 1

        baseline_confidences.append(
            baseline_record["choice_confidence"]
        )
        method_confidences.append(
            method_record["choice_confidence"]
        )

    discordant = baseline_only_correct + method_only_correct

    if discordant:
        statistic = (
            abs(baseline_only_correct - method_only_correct) - 1
        ) ** 2 / discordant
        mcnemar_p = float(chi2.sf(statistic, df=1))
    else:
        mcnemar_p = 1.0

    differences = [
        method - baseline
        for baseline, method in zip(
            baseline_confidences,
            method_confidences,
        )
    ]

    if all(math.isclose(difference, 0.0) for difference in differences):
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0
    else:
        test = wilcoxon(
            method_confidences,
            baseline_confidences,
            zero_method="wilcox",
            alternative="two-sided",
        )
        wilcoxon_statistic = float(test.statistic)
        wilcoxon_p = float(test.pvalue)

    return {
        "num_paired_examples": len(paired_ids),
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only_correct,
        "method_only_correct": method_only_correct,
        "both_wrong": both_wrong,
        "accuracy_mcnemar_corrected_p": mcnemar_p,
        "confidence_wilcoxon_statistic": wilcoxon_statistic,
        "confidence_wilcoxon_p": wilcoxon_p,
    }


def build_results() -> dict[str, Any]:
    final: dict[str, Any] = {
        "description": (
            "Choice-based metrics calculated from paired prediction records."
        ),
        "experiments": {},
    }

    for experiment_name, config in EXPERIMENTS.items():
        by_method = load_experiment(config["folders"])

        id_sets = {
            method: set(records)
            for method, records in by_method.items()
        }

        paired_ids = set.intersection(
            *(id_sets[method] for method in METHODS)
        )

        if not paired_ids:
            raise ValueError(
                f"No common paired IDs for {experiment_name}"
            )

        paired_ids = set(sorted(paired_ids))

        experiment_result: dict[str, Any] = {
            "dataset": config["dataset"],
            "attribute": config["attribute"],
            "num_paired_examples": len(paired_ids),
            "source_folders": config["folders"],
            "excluded_unpaired_records": {
                method: len(id_sets[method] - paired_ids)
                for method in METHODS
            },
            "methods": {},
            "comparisons_vs_baseline": {},
        }

        if "note" in config:
            experiment_result["note"] = config["note"]

        paired_records: dict[str, list[dict[str, Any]]] = {}

        for method in METHODS:
            records = [
                by_method[method][record_id]
                for record_id in sorted(paired_ids)
            ]

            paired_records[method] = records
            experiment_result["methods"][method] = (
                calculate_method_metrics(records)
            )

        baseline_metrics = experiment_result["methods"]["baseline"]

        for method in METHODS[1:]:
            method_metrics = experiment_result["methods"][method]

            comparison = paired_comparison(
                paired_records["baseline"],
                paired_records[method],
            )

            comparison.update(
                {
                    "accuracy_delta": (
                        method_metrics["accuracy"]
                        - baseline_metrics["accuracy"]
                    ),
                    "confidence_delta": (
                        method_metrics["avg_choice_confidence"]
                        - baseline_metrics["avg_choice_confidence"]
                    ),
                    "ece_delta": (
                        method_metrics["ece_10_bins"]
                        - baseline_metrics["ece_10_bins"]
                    ),
                    "brier_delta": (
                        method_metrics["multiclass_brier_score"]
                        - baseline_metrics["multiclass_brier_score"]
                    ),
                }
            )

            experiment_result["comparisons_vs_baseline"][
                method
            ] = comparison

        final["experiments"][experiment_name] = experiment_result

    return final


def write_csv(final: dict[str, Any]) -> None:
    columns = [
        "experiment",
        "dataset",
        "attribute",
        "num_examples",
        "method",
        "accuracy",
        "avg_choice_confidence",
        "ece_10_bins",
        "multiclass_brier_score",
        "avg_ground_truth_probability",
        "accuracy_delta_vs_baseline",
        "accuracy_p_value",
        "confidence_delta_vs_baseline",
        "confidence_p_value",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for experiment_name, experiment in final["experiments"].items():
            for method, metrics in experiment["methods"].items():
                comparison = experiment[
                    "comparisons_vs_baseline"
                ].get(method, {})

                writer.writerow(
                    {
                        "experiment": experiment_name,
                        "dataset": experiment["dataset"],
                        "attribute": experiment["attribute"],
                        "num_examples": metrics["num_examples"],
                        "method": method,
                        "accuracy": metrics["accuracy"],
                        "avg_choice_confidence": metrics[
                            "avg_choice_confidence"
                        ],
                        "ece_10_bins": metrics["ece_10_bins"],
                        "multiclass_brier_score": metrics[
                            "multiclass_brier_score"
                        ],
                        "avg_ground_truth_probability": metrics[
                            "avg_ground_truth_probability"
                        ],
                        "accuracy_delta_vs_baseline": comparison.get(
                            "accuracy_delta", ""
                        ),
                        "accuracy_p_value": comparison.get(
                            "accuracy_mcnemar_corrected_p", ""
                        ),
                        "confidence_delta_vs_baseline": comparison.get(
                            "confidence_delta", ""
                        ),
                        "confidence_p_value": comparison.get(
                            "confidence_wilcoxon_p", ""
                        ),
                    }
                )


def print_summary(final: dict[str, Any]) -> None:
    for experiment in final["experiments"].values():
        print(
            f"\n{experiment['dataset']}–{experiment['attribute']} "
            f"(N={experiment['num_paired_examples']})"
        )

        for method in METHODS:
            metrics = experiment["methods"][method]

            if method == "baseline":
                suffix = ""
            else:
                comparison = experiment[
                    "comparisons_vs_baseline"
                ][method]

                suffix = (
                    f" | Δacc={comparison['accuracy_delta']:+.4f}"
                    f", p={comparison['accuracy_mcnemar_corrected_p']:.4g}"
                    f" | Δconf={comparison['confidence_delta']:+.4f}"
                    f", p={comparison['confidence_wilcoxon_p']:.4g}"
                )

            print(
                f"  {method:8s} "
                f"acc={metrics['accuracy']:.4f} "
                f"conf={metrics['avg_choice_confidence']:.4f} "
                f"ECE={metrics['ece_10_bins']:.4f}"
                f"{suffix}"
            )


def main() -> None:
    global RESULTS_ROOT, OUTPUT_JSON, OUTPUT_CSV, EXPERIMENTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    RESULTS_ROOT = args.results_root
    OUTPUT_JSON = RESULTS_ROOT / "final_metrics.json"
    OUTPUT_CSV = RESULTS_ROOT / "final_metrics.csv"
    EXPERIMENTS = json.loads(args.config.read_text(encoding="utf-8"))
    final = build_results()

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(final, file, indent=2, ensure_ascii=False)

    write_csv(final)
    print_summary(final)

    print(f"\n생성 완료: {OUTPUT_JSON}")
    print(f"생성 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
