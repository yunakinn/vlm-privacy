import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from src import icd_inference as icd, choice_scoring
from src.baseline_inference import generation_seed, _parse_prediction
from src.evaluate import expected_calibration_error, multiclass_brier_score
from run_experiment import build_run_metadata, validate_resume_metadata, parse_args


def test_icd_forwards_custom_parameters(monkeypatch):
    generator = Mock(return_value=SimpleNamespace(sequences=torch.tensor([[1, 2]]), scores=[]))
    bundle = SimpleNamespace(device=torch.device("cpu"), model=SimpleNamespace(generate=generator), processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0)))
    scorer = Mock(return_value={})
    monkeypatch.setattr(icd, "load_image", lambda p: object())
    monkeypatch.setattr(icd, "prepare_inputs", lambda *a: {"input_ids": torch.tensor([[1]])})
    monkeypatch.setattr(icd, "decode_generated", lambda *a: "normal")
    monkeypatch.setattr(icd, "score_icd_choices", scorer)
    config = icd.ICDConfig(alpha=4, beta=.5, disturbance_instruction="custom", top_k=3)
    icd.run_icd_example(bundle, {"id": "a", "image": "a.png", "question": "test", "choices": ["normal", "abnormal"]}, ".", config)
    assert scorer.call_args.kwargs["alpha"] == 4
    assert scorer.call_args.kwargs["beta"] == .5
    assert scorer.call_args.kwargs["disturbance_instruction"] == "custom"
    assert generator.call_args.kwargs["top_k"] == 3


def test_seeded_sampling_is_independent_of_execution_order():
    before = torch.random.get_rng_state()
    with generation_seed("cpu", 42, "sample"):
        first = torch.rand(5)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(10)
    with generation_seed("cpu", 42, "sample"):
        second = torch.rand(5)
    assert torch.equal(first, second)


def test_parser_does_not_match_a_choice_inside_another_word():
    assert _parse_prediction("female", ["male", "female"]) == "female"
    assert _parse_prediction("I cannot determine this", ["male", "female"]) == "parse_error"
    assert _parse_prediction("B", ["male", "female"]) == "female"


def test_icd_disturbance_uses_real_newline(monkeypatch):
    prompts = []
    model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[[3., 2., 1.]]])))
    bundle = SimpleNamespace(model=model)
    def inputs(**kwargs):
        prompts.append(kwargs["prompt"])
        return {}
    monkeypatch.setattr(choice_scoring, "prepare_inputs", inputs)
    monkeypatch.setattr(choice_scoring, "_get_letter_probabilities", lambda **kw: torch.tensor([.6, .4]))
    choice_scoring.score_icd_choices(bundle, object(), "question", ["yes", "no"], disturbance_instruction="custom")
    assert prompts[1].startswith("custom\n")
    assert not prompts[1].startswith("custom\\n")


def test_final_aggregator_uses_shared_metrics():
    path = Path(__file__).resolve().parents[1] / "scripts/build_final_metrics.py"
    spec = importlib.util.spec_from_file_location("final_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = {"id": "a", "method": "baseline", "choices": ["young-adult", "old"], "answer": "young_adult", "confidence": {"choice_prediction": "young adult", "choice_confidence": .7, "choice_probabilities": {"young adult": .7, "old": .3}}}
    flat = module.extract_choice_fields(raw)
    assert module.calculate_ece([flat]) == expected_calibration_error([raw])
    assert module.calculate_brier([flat]) == multiclass_brier_score(raw)


def test_resume_records_m3id_and_rejects_different_implementation(monkeypatch, tmp_path):
    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]")
    monkeypatch.setattr("sys.argv", ["run_experiment.py", "--annotations", str(annotations), "--methods", "baseline", "m3id"])
    args = parse_args()
    metadata = build_run_metadata(args, 1, torch.device("cpu"), torch.float32, [], "running")
    assert metadata["method_implementations"]["m3id"]["step"] == args.m3id_step
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(metadata))
    validate_resume_metadata(path, args)
    metadata["implementation_revision"] = "old"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_metadata(path, args)
