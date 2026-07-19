import json
from pathlib import Path
from scipy.stats import chi2, wilcoxon

ROOT = Path(__file__).parent.parent

def load_predictions(*paths):
    """여러 jsonl 파일을 합쳐서 하나의 리스트로 반환"""
    all_preds = []
    for path in paths:
        full_path = ROOT / path
        with open(full_path, encoding="utf-8") as f:
            all_preds.extend(json.loads(line) for line in f)
    return all_preds

def split_by_method(predictions):
    by_method = {}
    for p in predictions:
        by_method.setdefault(p["method"], []).append(p)
    return by_method

def compute_accuracy_confidence(preds):
    correct = sum(
        1 for p in preds
        if p["confidence"]["choice_prediction"].strip().lower() == p["answer"].strip().lower()
    )
    total = len(preds)
    avg_conf = sum(p["confidence"]["choice_confidence"] for p in preds) / total
    return correct / total, avg_conf, total

def mcnemar_test(predictions_a, predictions_b, label_a="A", label_b="B"):
    by_id_a = {p["id"]: p for p in predictions_a}
    by_id_b = {p["id"]: p for p in predictions_b}
    common_ids = set(by_id_a.keys()) & set(by_id_b.keys())
    a_only, b_only = 0, 0
    for eid in common_ids:
        ca = by_id_a[eid]["confidence"]["choice_prediction"].strip().lower() == by_id_a[eid]["answer"].strip().lower()
        cb = by_id_b[eid]["confidence"]["choice_prediction"].strip().lower() == by_id_b[eid]["answer"].strip().lower()
        if ca and not cb:
            a_only += 1
        elif cb and not ca:
            b_only += 1
    n = a_only + b_only
    if n == 0:
        return None
    stat = (abs(a_only - b_only) - 1) ** 2 / n
    p_value = 1 - chi2.cdf(stat, df=1)
    return {"a_only": a_only, "b_only": b_only, "p_value": p_value}

def paired_confidence_test(predictions_a, predictions_b):
    by_id_a = {p["id"]: p for p in predictions_a}
    by_id_b = {p["id"]: p for p in predictions_b}
    common_ids = sorted(set(by_id_a.keys()) & set(by_id_b.keys()))
    conf_a = [by_id_a[eid]["confidence"]["choice_confidence"] for eid in common_ids]
    conf_b = [by_id_b[eid]["confidence"]["choice_confidence"] for eid in common_ids]
    stat, p_value = wilcoxon(conf_a, conf_b)
    return {"mean_a": sum(conf_a)/len(conf_a), "mean_b": sum(conf_b)/len(conf_b), "p_value": p_value}


DATASETS = {
    "UTKFace-Age": [
        "results/utkface_3k_part1/pilot_predictions.jsonl",
        "results/m3id_age_part1a/pilot_predictions.jsonl",
        "results/m3id_age_part1b/pilot_predictions.jsonl",
    ],
    "UTKFace-Race": [
        "results/utkface_3k_part2/pilot_predictions.jsonl",
        "results/m3id_race_part2a/pilot_predictions.jsonl",
        "results/m3id_race_part2b/pilot_predictions.jsonl",
    ],
    "PA-100K-Age": [f"results/pa100k_chunk{i}/pilot_predictions.jsonl" for i in range(5)],
}

print(f"{'Dataset':<15} {'Method':<10} {'N':<6} {'Accuracy':<10} {'Confidence':<10}")
print("-" * 55)

results_summary = {}

for name, paths in DATASETS.items():
    preds = load_predictions(*paths)
    by_method = split_by_method(preds)
    results_summary[name] = {}
    for method in ["baseline", "vcd", "m3id"]:
        if method not in by_method:
            continue
        acc, conf, n = compute_accuracy_confidence(by_method[method])
        results_summary[name][method] = {"accuracy": acc, "confidence": conf, "n": n, "preds": by_method[method]}
        print(f"{name:<15} {method:<10} {n:<6} {acc:<10.4f} {conf:<10.4f}")
    print()

print("\n=== 통계 검정 ===\n")
for name in DATASETS:
    if "baseline" not in results_summary[name]:
        continue
    baseline_preds = results_summary[name]["baseline"]["preds"]
    for method in ["vcd", "m3id"]:
        if method not in results_summary[name]:
            continue
        method_preds = results_summary[name][method]["preds"]
        mc = mcnemar_test(baseline_preds, method_preds)
        pc = paired_confidence_test(baseline_preds, method_preds)
        sig_acc = "유의미" if mc and mc["p_value"] < 0.05 else "유의미하지 않음"
        sig_conf = "유의미" if pc["p_value"] < 0.05 else "유의미하지 않음"
        print(f"{name} | Baseline vs {method.upper()}:")
        print(f"  Accuracy p={mc['p_value']:.4f} ({sig_acc})")
        print(f"  Confidence p={pc['p_value']:.10f} ({sig_conf})")
        print()