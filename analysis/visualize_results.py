import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, wilcoxon

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "analysis" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


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


def compute_accuracy_confidence(preds):
    correct = sum(
        1 for p in preds
        if p["confidence"]["choice_prediction"].strip().lower() == p["answer"].strip().lower()
    )
    total = len(preds)
    avg_conf = sum(p["confidence"]["choice_confidence"] for p in preds) / total
    return correct / total, avg_conf


def _is_correct(p):
    return p["confidence"]["choice_prediction"].strip().lower() == p["answer"].strip().lower()


def mcnemar_p(preds_a, preds_b):
    """Baseline vs method 쌍이 correct/incorrect를 바꾼 샘플만으로 accuracy 차이 유의성 검정"""
    by_id_a = {p["id"]: p for p in preds_a}
    by_id_b = {p["id"]: p for p in preds_b}
    common_ids = set(by_id_a) & set(by_id_b)
    a_only = b_only = 0
    for eid in common_ids:
        ca, cb = _is_correct(by_id_a[eid]), _is_correct(by_id_b[eid])
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
    """Baseline vs method의 paired confidence 차이 유의성 검정"""
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


DATASETS = {
    "UTKFace-Age": [
        "results/utkface_3k_part1/pilot_predictions.jsonl",
        "results/m3id_age_part1a/pilot_predictions.jsonl",
        "results/m3id_age_part1b/pilot_predictions.jsonl",
    ],
    "PA-100K-Age": [f"results/pa100k_chunk{i}/pilot_predictions.jsonl" for i in range(5)],
    "UTKFace-Race": [
        "results/utkface_3k_part2/pilot_predictions.jsonl",
        "results/m3id_race_part2a/pilot_predictions.jsonl",
        "results/m3id_race_part2b/pilot_predictions.jsonl",
    ],
}

METHOD_LABELS = {"baseline": "Baseline", "vcd": "VCD", "m3id": "M3ID"}
METHOD_COLORS = {
    "baseline": "#595959",  # tableau-colorblind10 gray
    "vcd": "#C85200",       # tableau-colorblind10 orange
    "m3id": "#006BA4",      # tableau-colorblind10 blue
}
GAP_COLOR = "#d03b3b"  # status red, CVD-validated against all three method colors

# 데이터 수집
results = {}
raw_preds = {}
for name, paths in DATASETS.items():
    preds = load_predictions(*paths)
    by_method = split_by_method(preds)
    raw_preds[name] = by_method
    results[name] = {}
    for method in ["baseline", "vcd", "m3id"]:
        if method in by_method:
            acc, conf = compute_accuracy_confidence(by_method[method])
            results[name][method] = {"accuracy": acc, "confidence": conf}

# baseline 대비 VCD/M3ID의 accuracy(McNemar)·confidence(Wilcoxon) 유의성 -
# 그래프의 "confidence는 항상 유의미하게 상승, accuracy는 일관되지 않음"이라는
# 핵심 주장을 막대 라벨에 직접 표시하기 위해 미리 계산해둔다.
SIGNIFICANCE = {}
for name in DATASETS:
    SIGNIFICANCE[name] = {}
    baseline_preds = raw_preds[name].get("baseline")
    if not baseline_preds:
        continue
    for method in ["vcd", "m3id"]:
        method_preds = raw_preds[name].get(method)
        if not method_preds:
            continue
        SIGNIFICANCE[name][method] = {
            "accuracy": significance_stars(mcnemar_p(baseline_preds, method_preds)),
            "confidence": significance_stars(wilcoxon_p(baseline_preds, method_preds)),
        }


def plot_main_comparison(datasets, filename, title):
    """Accuracy와 Confidence를 나란히 보여주는 그룹 막대그래프.
    VCD/M3ID 막대에는 baseline 대비 유의성(McNemar/Wilcoxon)을 별표로 표시."""
    methods = ["baseline", "vcd", "m3id"]
    fig, axes = plt.subplots(1, 2, figsize=(3.2 * len(datasets) + 3, 5.5))

    for ax_idx, metric in enumerate(["accuracy", "confidence"]):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.25

        for i, method in enumerate(methods):
            values = [results[d].get(method, {}).get(metric, 0) for d in datasets]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
            for bar, v, d in zip(bars, values, datasets):
                sig = SIGNIFICANCE.get(d, {}).get(method, {}).get(metric, "")
                label = f"{v:.2f}" + (f"\n{sig}" if sig else "")
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, label,
                        ha="center", va="bottom", fontsize=8, linespacing=1.3)

        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=0)
        ax.set_ylim(0, 1.2)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Significance vs. Baseline: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant",
              ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    print(f"저장됨: {OUTPUT_DIR / filename}")
    plt.close()


# 메인 그래프: age(2개) + race(1개) 데이터셋을 하나로 통합
plot_main_comparison(
    ["UTKFace-Age", "PA-100K-Age", "UTKFace-Race"],
    "main_comparison.png",
    "Age & Race Inference: Accuracy vs Confidence Across Datasets"
)

# 기존에 나눠 실었던 main_age_comparison.png / supplementary_race.png는 위 그래프로 통합됐으므로 정리
for stale in ["main_age_comparison.png", "supplementary_race.png"]:
    stale_path = OUTPUT_DIR / stale
    if stale_path.exists():
        stale_path.unlink()
        print(f"삭제됨(통합됨): {stale_path}")

def plot_delta_comparison(datasets, filename, title):
    """Baseline 대비 변화량(delta)만 보여주는 그래프 - 차이를 극대화해서 표시"""
    methods = ["vcd", "m3id"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax_idx, metric in enumerate(["accuracy", "confidence"]):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.35

        for i, method in enumerate(methods):
            deltas = [
                results[d][method][metric] - results[d]["baseline"][metric]
                for d in datasets
            ]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, deltas, width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
            for bar, v, d in zip(bars, deltas, datasets):
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                offset_amt = y_range * 0.03
                y_pos = v + offset_amt if v >= 0 else v - offset_amt
                sig = SIGNIFICANCE.get(d, {}).get(method, {}).get(metric, "")
                label = f"{v:+.3f}" + (f" {sig}" if sig else "")
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
        ax.margins(y=0.2)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Δ {metric.capitalize()} (vs Baseline)")
        ax.set_title(f"Change in {metric.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "Significance vs. Baseline: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant",
              ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    print(f"저장됨: {OUTPUT_DIR / filename}")
    plt.close()


plot_delta_comparison(
    ["UTKFace-Age", "PA-100K-Age", "UTKFace-Race"],
    "delta_comparison.png",
    "Change from Baseline: Accuracy vs Confidence"
)

def compute_reliability_bins(preds, n_bins=10):
    """confidence를 n_bins개 균등 구간으로 나눠서 bin별 accuracy/confidence/count를 계산.
    Guo et al. (2017) 표준을 따라 샘플이 없는 구간도 0으로 유지해 막대 위치를 정렬시킴."""
    bins = [[] for _ in range(n_bins)]
    for p in preds:
        conf = p["confidence"]["choice_confidence"]
        correct = p["confidence"]["choice_prediction"].strip().lower() == p["answer"].strip().lower()
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    bin_confs, bin_accs, bin_counts = [], [], []
    for b in bins:
        if len(b) == 0:
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


def plot_reliability_diagram(dataset_name, filename, n_bins=10):
    """Guo et al. (2017) 표준 reliability diagram: bin별 accuracy 막대 + confidence-accuracy
    gap을 위에 쌓은 빨간 막대 + 대각선(perfect calibration) + 하단 confidence histogram.
    baseline/VCD/M3ID를 각각 별도 열(column)로 나란히 배치해 overconfidence gap을 비교."""
    preds = load_predictions(*DATASETS[dataset_name])
    by_method = split_by_method(preds)
    methods = [m for m in ["baseline", "vcd", "m3id"] if m in by_method]

    bin_width = 1 / n_bins
    bin_centers = [(i + 0.5) * bin_width for i in range(n_bins)]

    fig, axes = plt.subplots(2, len(methods), figsize=(4.5 * len(methods), 6.5),
                              sharex=True, height_ratios=[3, 1])
    if len(methods) == 1:
        axes = axes.reshape(2, 1)

    max_pct = 0
    method_data = {}
    for method in methods:
        bin_confs, bin_accs, bin_counts = compute_reliability_bins(by_method[method], n_bins)
        total = sum(bin_counts)
        pct = [c / total * 100 for c in bin_counts]
        max_pct = max(max_pct, max(pct))
        acc, conf = compute_accuracy_confidence(by_method[method])
        ece = compute_ece(bin_confs, bin_accs, bin_counts)
        method_data[method] = dict(bin_confs=bin_confs, bin_accs=bin_accs, bin_counts=bin_counts,
                                    pct=pct, acc=acc, conf=conf, ece=ece)

    for col, method in enumerate(methods):
        d = method_data[method]
        ax_top, ax_bot = axes[0, col], axes[1, col]

        ax_top.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, zorder=1)
        nonzero = [c > 0 for c in d["bin_counts"]]
        centers = [x for x, nz in zip(bin_centers, nonzero) if nz]
        accs = [x for x, nz in zip(d["bin_accs"], nonzero) if nz]
        confs = [x for x, nz in zip(d["bin_confs"], nonzero) if nz]
        gaps = [cf - a for cf, a in zip(confs, accs)]

        ax_top.bar(centers, accs, width=bin_width * 0.95, color=METHOD_COLORS[method],
                   edgecolor="black", linewidth=0.5, label="Accuracy", zorder=2)
        # GAP_COLOR is a status red validated against baseline/vcd/m3id (esp. vs. the
        # VCD orange, where plain red gave CVD ΔE 3.7 — invisible under protanopia)
        ax_top.bar(centers, gaps, width=bin_width * 0.95, bottom=accs, color=GAP_COLOR, alpha=0.55,
                   hatch="//", edgecolor="black", linewidth=0.5, label="Gap (Conf. − Acc.)", zorder=2)

        ax_top.text(0.05, 0.95,
                    f"ECE = {d['ece']:.3f}\nAcc = {d['acc']:.3f}\nConf = {d['conf']:.3f}",
                    transform=ax_top.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

        ax_top.set_title(METHOD_LABELS[method], fontweight="bold")
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.grid(alpha=0.3)
        if col == 0:
            ax_top.set_ylabel("Accuracy")
            ax_top.legend(loc="lower right", fontsize=8)

        ax_bot.bar(bin_centers, d["pct"], width=bin_width * 0.95, color=METHOD_COLORS[method],
                   edgecolor="black", linewidth=0.5)
        ax_bot.axvline(d["conf"], color="black", linestyle="--", linewidth=1.2,
                       label=f"Avg conf.={d['conf']:.2f}")
        ax_bot.set_xlim(0, 1)
        ax_bot.legend(loc="upper left", fontsize=7)
        ax_bot.set_xlabel("Confidence")
        if col == 0:
            ax_bot.set_ylabel("% of Samples")
        ax_bot.grid(alpha=0.3)

    for col in range(len(methods)):
        axes[1, col].set_ylim(0, max_pct * 1.15)

    fig.suptitle(f"Reliability Diagram: {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    print(f"저장됨: {OUTPUT_DIR / filename}")
    plt.close()


for dataset_name in DATASETS:
    safe_name = dataset_name.lower().replace("-", "_")
    plot_reliability_diagram(dataset_name, f"reliability_{safe_name}.png")

print("\n완료")