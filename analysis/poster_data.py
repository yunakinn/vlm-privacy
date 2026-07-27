"""
Shared data loading + statistics for the poster figures.

Reads the raw jsonl prediction files under results/ for all six real
dataset-attribute settings and computes accuracy / confidence / ECE per
method, plus significance vs. baseline (McNemar for accuracy, Wilcoxon
signed-rank for confidence). analysis/poster_figures.py consumes this.
"""
import json
from pathlib import Path
from scipy.stats import chi2, wilcoxon

ROOT = Path(__file__).parent.parent

# One entry per dataset-attribute setting. Each setting's predictions may be
# split across several chunked/method-specific run directories (e.g. ICD was
# run separately from baseline/VCD/M3ID for UTKFace and PA-100K); all files
# listed are simply concatenated and then split by the "method" field.
DATASETS = {
    "UTKFace-Age": [
        "results/utkface_3k_part1/pilot_predictions.jsonl",
        "results/m3id_age_part1a/pilot_predictions.jsonl",
        "results/m3id_age_part1b/pilot_predictions.jsonl",
        "results/icd_age_chunk0/pilot_predictions.jsonl",
        "results/icd_age_chunk1/pilot_predictions.jsonl",
        "results/icd_age_chunk2/pilot_predictions.jsonl",
    ],
    "UTKFace-Race": [
        "results/utkface_3k_part2/pilot_predictions.jsonl",
        "results/m3id_race_part2a/pilot_predictions.jsonl",
        "results/m3id_race_part2b/pilot_predictions.jsonl",
        "results/icd_race_chunk0/pilot_predictions.jsonl",
        "results/icd_race_chunk1/pilot_predictions.jsonl",
        "results/icd_race_chunk2/pilot_predictions.jsonl",
    ],
    "PA-100K-Age": [
        *[f"results/pa100k_chunk{i}/pilot_predictions.jsonl" for i in range(5)],
        *[f"results/icd_pa100k_chunk{i}/pilot_predictions.jsonl" for i in range(5)],
    ],
    "FairFace-Gender": [
        f"results/fairface_gender_chunk{i}/pilot_predictions.jsonl" for i in range(5)
    ],
    # Final balanced 7-category race set (~1800 examples). Supersedes the
    # earlier fairface_race_add_* / unreduced chunks, which were an
    # intermediate attempt at a 3000-example pool that was later abandoned
    # in favor of this smaller balanced set (see commit "FairFace race
    # reduced to 1800 (balanced 7 categories)").
    "FairFace-Race": [
        "results/fairface_race_reduced_chunk0/pilot_predictions.jsonl",
        "results/fairface_race_reduced_chunk1/pilot_predictions.jsonl",
        "results/fairface_race_chunk2_sub0/pilot_predictions.jsonl",
        "results/fairface_race_chunk2_sub1/pilot_predictions.jsonl",
        "results/fairface_race_chunk2_sub2/pilot_predictions.jsonl",
    ],
    "MSP60K-Gender": [
        f"results/msp60k_gender_chunk{i}/pilot_predictions.jsonl" for i in range(4)
    ],
}

METHODS = ["baseline", "vcd", "m3id", "icd"]
MITIGATION_METHODS = ["vcd", "m3id", "icd"]  # the 3 methods compared against baseline
METHOD_LABELS = {"baseline": "Baseline", "vcd": "VCD", "m3id": "M3ID", "icd": "ICD"}


def load_predictions(*paths):
    all_preds = []
    for path in paths:
        with open(ROOT / path, encoding="utf-8") as f:
            all_preds.extend(json.loads(line) for line in f)
    return all_preds


def split_by_method(predictions):
    by_method = {}
    for p in predictions:
        by_method.setdefault(p["method"], []).append(p)
    return by_method


def is_correct(p):
    return p["confidence"]["choice_prediction"].strip().lower() == p["answer"].strip().lower()


def compute_accuracy_confidence(preds):
    correct = sum(1 for p in preds if is_correct(p))
    total = len(preds)
    avg_conf = sum(p["confidence"]["choice_confidence"] for p in preds) / total
    return correct / total, avg_conf, total


def compute_reliability_bins(preds, n_bins=10):
    """Guo et al. (2017)-style equal-width confidence bins; empty bins kept as 0
    so bar positions stay aligned across methods."""
    bins = [[] for _ in range(n_bins)]
    for p in preds:
        conf = p["confidence"]["choice_confidence"]
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, is_correct(p)))

    bin_confs, bin_accs, bin_counts = [], [], []
    for b in bins:
        if not b:
            bin_confs.append(0.0)
            bin_accs.append(0.0)
            bin_counts.append(0)
            continue
        confs = [x[0] for x in b]
        corrects = [x[1] for x in b]
        bin_confs.append(sum(confs) / len(confs))
        bin_accs.append(sum(corrects) / len(corrects))
        bin_counts.append(len(b))
    return bin_confs, bin_accs, bin_counts


def compute_ece(bin_confs, bin_accs, bin_counts):
    n = sum(bin_counts)
    return sum(c / n * abs(a - cf) for cf, a, c in zip(bin_confs, bin_accs, bin_counts) if c > 0)


def mcnemar_p(preds_a, preds_b):
    """Paired accuracy-change significance (baseline vs. method) on shared ids."""
    by_id_a = {p["id"]: p for p in preds_a}
    by_id_b = {p["id"]: p for p in preds_b}
    common_ids = set(by_id_a) & set(by_id_b)
    a_only = b_only = 0
    for eid in common_ids:
        ca, cb = is_correct(by_id_a[eid]), is_correct(by_id_b[eid])
        if ca and not cb:
            a_only += 1
        elif cb and not ca:
            b_only += 1
    n = a_only + b_only
    if n == 0:
        return 1.0
    stat = (abs(a_only - b_only) - 1) ** 2 / n
    return 1 - chi2.cdf(stat, df=1)


def wilcoxon_p(preds_a, preds_b):
    """Paired confidence-change significance (baseline vs. method) on shared ids."""
    by_id_a = {p["id"]: p for p in preds_a}
    by_id_b = {p["id"]: p for p in preds_b}
    common_ids = sorted(set(by_id_a) & set(by_id_b))
    conf_a = [by_id_a[eid]["confidence"]["choice_confidence"] for eid in common_ids]
    conf_b = [by_id_b[eid]["confidence"]["choice_confidence"] for eid in common_ids]
    _, p_value = wilcoxon(conf_a, conf_b)
    return p_value


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_all():
    """Returns (raw_preds, stats, significance):
    raw_preds[dataset][method]      -> list of prediction dicts
    stats[dataset][method]          -> {"accuracy", "confidence", "ece", "n"}
    significance[dataset][method]   -> {"accuracy": stars, "accuracy_p": p,
                                         "confidence": stars, "confidence_p": p}
    """
    raw_preds, stats, significance = {}, {}, {}
    for name, paths in DATASETS.items():
        preds = load_predictions(*paths)
        by_method = split_by_method(preds)
        raw_preds[name] = by_method
        stats[name] = {}
        for method in METHODS:
            if method not in by_method:
                continue
            acc, conf, n = compute_accuracy_confidence(by_method[method])
            _, bin_accs, bin_counts = compute_reliability_bins(by_method[method])
            bin_confs, _, _ = compute_reliability_bins(by_method[method])
            ece = compute_ece(bin_confs, bin_accs, bin_counts)
            stats[name][method] = {"accuracy": acc, "confidence": conf, "ece": ece, "n": n}

        significance[name] = {}
        baseline_preds = by_method.get("baseline")
        if not baseline_preds:
            continue
        for method in MITIGATION_METHODS:
            method_preds = by_method.get(method)
            if not method_preds:
                continue
            acc_p = mcnemar_p(baseline_preds, method_preds)
            conf_p = wilcoxon_p(baseline_preds, method_preds)
            significance[name][method] = {
                "accuracy": significance_stars(acc_p),
                "accuracy_p": acc_p,
                "confidence": significance_stars(conf_p),
                "confidence_p": conf_p,
            }
    return raw_preds, stats, significance


if __name__ == "__main__":
    raw_preds, stats, significance = load_all()
    print(f"{'Dataset':<16} {'Method':<10} {'N':<6} {'Accuracy':<10} {'Confidence':<11} {'ECE':<8}")
    print("-" * 65)
    for name in DATASETS:
        for method in METHODS:
            if method not in stats[name]:
                continue
            s = stats[name][method]
            print(f"{name:<16} {method:<10} {s['n']:<6} {s['accuracy']:<10.4f} {s['confidence']:<11.4f} {s['ece']:<8.4f}")
        print()
