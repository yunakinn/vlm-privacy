import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_predictions.py"


def record(**overrides):
    return {"id": "sample-1", "method": "baseline", "attribute": "age",
            "prediction": "young", "confidence": {
                "choice_prediction": "young", "choice_probabilities": {"young": .7, "old": .3}},
            **overrides}


def validate(tmp_path, records, *options):
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return subprocess.run([sys.executable, str(SCRIPT), "--predictions", str(path), *options],
                          capture_output=True, text=True)


@pytest.mark.parametrize("probabilities", [
    {"young": 1.2, "old": -.2}, {"young": float("nan"), "old": .3},
    {"young": float("inf"), "old": .3}, {"young": True, "old": False},
    {"young": "0.7", "old": .3}, {"young": 10**400, "old": .3},
    {}, {"young": .4, "old": .3}, [.7, .3],
])
def test_invalid_distribution_is_rejected(tmp_path, probabilities):
    result = validate(tmp_path, [record(confidence={"choice_prediction": "young",
                                                   "choice_probabilities": probabilities})])
    assert result.returncode == 1
    assert "Validation errors:" in result.stdout
    assert "Validation passed" not in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("field", ["id", "method"])
@pytest.mark.parametrize("value", [None, "", "  ", [], {}, True])
def test_missing_or_invalid_identity_is_rejected(tmp_path, field, value):
    item = record()
    item[field] = value
    result = validate(tmp_path, [item])
    assert result.returncode == 1
    assert f"invalid {field}" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("field", ["id", "method"])
def test_absent_identity_is_rejected(tmp_path, field):
    item = record()
    del item[field]
    result = validate(tmp_path, [item])
    assert result.returncode == 1
    assert f"invalid {field}" in result.stdout


@pytest.mark.parametrize("item", [[], None, record(confidence=[]), record(attribute=[]), record(prediction=[], confidence={})])
def test_malformed_records_report_validation_errors_instead_of_crashing(tmp_path, item):
    result = validate(tmp_path, [item])
    assert result.returncode == 1
    assert "Validation errors:" in result.stdout
    assert "Traceback" not in result.stderr


def test_valid_boundaries_integer_ids_and_expected_counts(tmp_path):
    rows = [record(id=0, confidence={"choice_prediction": "young", "choice_probabilities": {"young": 1, "old": 0}}),
            record(method="vcd")]
    result = validate(tmp_path, rows, "--expected-methods", "baseline", "vcd", "--expected-per-method", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation passed" in result.stdout


def test_free_form_predictions_without_choice_probabilities_remain_supported(tmp_path):
    result = validate(tmp_path, [record(confidence=None)])
    assert result.returncode == 0


@pytest.mark.parametrize("case", ["duplicate", "method", "count"])
def test_existing_duplicate_method_and_count_checks_are_preserved(tmp_path, case):
    rows, options = [record()], []
    if case == "duplicate":
        rows.append(record())
    elif case == "method":
        options = ["--expected-methods", "vcd"]
    else:
        options = ["--expected-per-method", "2"]
    result = validate(tmp_path, rows, *options)
    assert result.returncode == 1
    assert "Validation errors:" in result.stdout
