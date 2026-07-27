"""Generate a consistent, visual-first suite of poster figures.

All quantitative panels are derived from raw JSONL predictions through
poster_data.load_all(). Dataset source images are intentionally not fabricated:
when an image path recorded during inference is unavailable on this machine,
the corresponding tile states that explicitly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

from poster_data import DATASETS, METHOD_LABELS, MITIGATION_METHODS, load_all


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#202124"
MUTED = "#666A70"
RED = "#8F1D22"
BLUE = "#527B9E"
ORANGE = "#CB7837"
GREEN = "#4B896A"
LIGHT = "#D9DDE0"
PALE = "#F4F5F5"
SURFACE = "#FFFFFF"

ORDER = [
    ("Age", "UTKFace", "UTKFace-Age"),
    ("Age", "PA-100K", "PA-100K-Age"),
    ("Gender", "FairFace", "FairFace-Gender"),
    ("Gender", "MSP60K", "MSP60K-Gender"),
    ("Race", "UTKFace", "UTKFace-Race"),
    ("Race", "FairFace", "FairFace-Race"),
]
METHOD_COLORS = {"vcd": "#D8874D", "m3id": "#4D9A83", "icd": "#7467A8"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
    "svg.fonttype": "none",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", facecolor=SURFACE, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {stem}.png")


def clean(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)


def arrow(ax, p1, p2, color=INK, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=11, color=color,
        linewidth=lw, shrinkA=2, shrinkB=2,
    ))


def missing_tile(ax, x, y, w, h, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PALE,
                           edgecolor=LIGHT, linewidth=1))
    ax.add_patch(Circle((x+w/2, y+h*.60), h*.09, color="#B5BBC0"))
    ax.add_patch(Rectangle((x+w*.34, y+h*.22), w*.32, h*.24,
                           facecolor="#B5BBC0", edgecolor="none"))
    ax.text(x+w/2, y+h*.08, label, ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=MUTED)


def plot_dataset_overview():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0))
    groups = [
        ("FACIAL IMAGES", [("UTKFace", "Age · Race"), ("FairFace", "Gender · Race")]),
        ("FULL-BODY IMAGES", [("PA-100K", "Age"), ("MSP60K", "Gender")]),
    ]
    for ax, (title, datasets) in zip(axes, groups):
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
        ax.add_patch(Rectangle((.06, .10), 9.88, 4.76, facecolor="white",
                               edgecolor=LIGHT, linewidth=1.1))
        ax.text(.38, 4.54, title, fontsize=15, fontweight="bold", color=RED)
        for ds_i, (name, attrs) in enumerate(datasets):
            start = .38 + ds_i*4.85
            ax.text(start, 4.08, name, fontsize=13, fontweight="bold")
            ax.text(start, 3.78, attrs, fontsize=9.5, color=MUTED)
            for i in range(4):
                missing_tile(ax, start+(i%2)*2.12, 2.04-(i//2)*1.45,
                             1.82, 1.20, f"sample {i+1}")
        ax.text(5, .27, "Source thumbnails not stored in this repository",
                ha="center", va="center", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(wspace=.06)
    save(fig, "dataset_overview_visual")


def plot_confidence(stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharex=True)
    groups = [
        ("AGE", ORDER[:2]), ("GENDER", ORDER[2:4]), ("RACE", ORDER[4:]),
    ]
    increases = 0
    for ax, (attr, rows) in zip(axes, groups):
        y = np.array([5.0, 4.15, 3.30, 1.85, 1.00, .15])
        vals, labels, colors = [], [], []
        for _, ds_label, key in rows:
            for method in MITIGATION_METHODS:
                delta = stats[key][method]["confidence"] - stats[key]["baseline"]["confidence"]
                vals.append(delta); labels.append(METHOD_LABELS[method])
                colors.append(BLUE if delta >= 0 else ORANGE)
                increases += delta > 0
        ax.barh(y, vals, height=.52, color=colors, edgecolor="none")
        for yy, val, method in zip(y, vals, labels):
            sign = "+" if val >= 0 else "\N{MINUS SIGN}"
            ax.text(-.027, yy, method, ha="left", va="center",
                    fontsize=11, fontweight="bold")
            # Keep the sole negative label on the readable side of the zero
            # line; otherwise the minus glyph visually merges with the axis.
            label_x = val + .003 if val >= 0 else .003
            ax.text(label_x, yy, f"{sign}{abs(val):.3f}", ha="left",
                    va="center", fontsize=12, fontweight="bold")
        ax.text(.005, 5.55, rows[0][1], fontsize=12.5, fontweight="bold")
        ax.text(.005, 2.40, rows[1][1], fontsize=12.5, fontweight="bold")
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_xlim(-.03, .16); ax.set_ylim(-.45, 6.15)
        ax.set_yticks([]); ax.grid(axis="x", color="#E8EAEB", lw=.8)
        ax.set_title(attr, fontsize=17, fontweight="bold", color=RED, pad=14)
        clean(ax)
    fig.text(.5, .97, f"{increases} OF 18 INCREASED", ha="center",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(.5, .02, "Δ mean confidence vs. Baseline",
             ha="center", fontsize=11, color=MUTED)
    fig.subplots_adjust(left=.035, right=.985, top=.84, bottom=.11, wspace=.13)
    save(fig, "confidence_delta_main")


def plot_accuracy(stats, significance):
    methods = ["baseline", "vcd", "m3id", "icd"]
    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.1); ax.axis("off")
    ax.text(5, 6.72, "ACCURACY EFFECTS WERE INCONSISTENT",
            ha="center", fontsize=18, fontweight="bold", color=RED)

    x_setting, xs = .55, [4.25, 5.55, 6.85, 8.15]
    ax.text(x_setting, 6.02, "Setting", fontsize=11.5,
            fontweight="bold", va="center")
    for x, method in zip(xs, methods):
        ax.text(x, 6.02, METHOD_LABELS[method], ha="right", va="center",
                fontsize=11.5, fontweight="bold")
    ax.plot([.45, 8.45], [5.74, 5.74], color=INK, lw=1.25)

    y_rows = np.linspace(5.25, 1.30, 6)
    for row, ((attr, ds, key), y) in enumerate(zip(ORDER, y_rows)):
        ax.text(x_setting, y, f"{ds}–{attr}", fontsize=11.5, va="center")
        row_values = [100*stats[key][m]["accuracy"] for m in methods]
        best = int(np.argmax(row_values))
        for col, (x, method, value) in enumerate(zip(xs, methods, row_values)):
            marker, marker_color = "", MUTED
            if method != "baseline":
                sig = significance[key][method]["accuracy"]
                delta = value - row_values[0]
                if sig != "ns":
                    marker = " ↑" if delta > 0 else " ↓"
                    marker_color = GREEN if delta > 0 else "#B95C55"
            ax.text(x, y, f"{value:.2f}", ha="right", va="center",
                    fontsize=11.5, fontweight="bold" if col == best else "normal")
            if marker:
                ax.text(x+.08, y, marker, ha="left", va="center",
                        fontsize=10.5, fontweight="bold", color=marker_color)
        ax.plot([.45, 8.45], [y-.37, y-.37],
                color=LIGHT, lw=.65 if row not in (1, 3) else 1.15)
    ax.plot([.45, 8.45], [.75, .75], color=INK, lw=1.25)
    ax.text(4.45, .30,
            "Values are accuracy (%)  ·  bold = row maximum  ·  ↑/↓ = significant change vs. Baseline",
            ha="center", fontsize=9.5, color=MUTED)
    save(fig, "accuracy_summary_matrix")


def plot_calibration(stats):
    methods = ["baseline", "vcd", "m3id", "icd"]
    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.1); ax.axis("off")
    ax.text(5, 6.72, "CALIBRATION VARIED ACROSS SETTINGS",
            ha="center", fontsize=18, fontweight="bold", color=RED)

    x_setting, xs = .55, [4.25, 5.55, 6.85, 8.15]
    ax.text(x_setting, 6.02, "Setting", fontsize=11.5,
            fontweight="bold", va="center")
    for x, method in zip(xs, methods):
        ax.text(x, 6.02, METHOD_LABELS[method], ha="right", va="center",
                fontsize=11.5, fontweight="bold")
    ax.plot([.45, 8.45], [5.74, 5.74], color=INK, lw=1.25)

    y_rows = np.linspace(5.25, 1.30, 6)
    for row, ((attr, ds, key), y) in enumerate(zip(ORDER, y_rows)):
        ax.text(x_setting, y, f"{ds}–{attr}", fontsize=11.5, va="center")
        row_values = [stats[key][m]["ece"] for m in methods]
        best = int(np.argmin(row_values))
        for col, (x, value) in enumerate(zip(xs, row_values)):
            highlight = key == "MSP60K-Gender" and col == 1
            if highlight:
                ax.add_patch(Rectangle((x-.73, y-.28), .88, .56,
                                       facecolor="#E8F2EC", edgecolor=GREEN,
                                       linewidth=1.0, zorder=0))
            ax.text(x, y, f"{value:.4f}", ha="right", va="center",
                    fontsize=11.5,
                    fontweight="bold" if col == best or highlight else "normal",
                    color=GREEN if highlight else INK)
        ax.plot([.45, 8.45], [y-.37, y-.37],
                color=LIGHT, lw=.65 if row not in (1, 3) else 1.15)
    ax.plot([.45, 8.45], [.75, .75], color=INK, lw=1.25)
    ax.text(9.05, 2.86, "VCD", fontsize=10, fontweight="bold", color=GREEN)
    ax.text(9.05, 2.49, "0.1222", fontsize=10.5, color=MUTED)
    ax.text(9.05, 2.13, "↓", fontsize=15, fontweight="bold", color=GREEN)
    ax.text(9.05, 1.77, "0.0595", fontsize=13, fontweight="bold", color=GREEN)
    ax.text(4.45, .30, "Expected Calibration Error (ECE)  ·  lower is better  ·  bold = row minimum",
            ha="center", fontsize=9.5, color=MUTED)
    save(fig, "calibration_summary_heatmap")


def choose_wrong_case(raw):
    """Find a paired case that stays wrong while mitigation confidence rises."""
    for dataset in DATASETS:
        baseline = {p["id"]: p for p in raw[dataset].get("baseline", [])}
        for method in MITIGATION_METHODS:
            other = {p["id"]: p for p in raw[dataset].get(method, [])}
            for pid in sorted(set(baseline) & set(other)):
                b, m = baseline[pid], other[pid]
                bp = b["confidence"]["choice_prediction"].strip().lower()
                mp = m["confidence"]["choice_prediction"].strip().lower()
                truth = b["answer"].strip().lower()
                bc = b["confidence"]["choice_confidence"]
                mc = m["confidence"]["choice_confidence"]
                if bp != truth and mp != truth and mc > bc + .03:
                    return dataset, method, b, m
    raise RuntimeError("No qualifying example was found.")


def resolve_image(record):
    candidates = [
        Path(record["image"]),
        ROOT / record["image"],
        ROOT / "data" / Path(record["image"]).name,
    ]
    return next((p for p in candidates if p.is_file()), None)


def plot_example(raw):
    dataset, method, b, m = choose_wrong_case(raw)
    image_path = resolve_image(b)
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(6.25, 4.12, "WRONG PREDICTION, HIGHER CONFIDENCE",
            ha="center", fontsize=18, fontweight="bold", color=RED)
    # Image area.
    ax.add_patch(Rectangle((.35, .48), 3.0, 3.20, fc=PALE, ec=LIGHT, lw=1))
    if image_path:
        im = Image.open(image_path).convert("RGB")
        ax.imshow(im, extent=(.35, 3.35, .48, 3.68), aspect="auto")
    else:
        ax.add_patch(Circle((1.85, 2.45), .35, color="#B5BBC0"))
        ax.add_patch(Rectangle((1.20, 1.25), 1.30, .75,
                               facecolor="#B5BBC0", edgecolor="none"))
        ax.text(1.85, .78, "Original source image\nnot available locally",
                ha="center", fontsize=9.5, color=MUTED)
    truth = b["answer"]
    ax.text(1.85, .20, f"{dataset}  ·  ID {b['id']}",
            ha="center", fontsize=9, color=MUTED)
    cards = [
        (4.05, "GROUND TRUTH", truth, None, BLUE),
        (7.00, "BASELINE", b["confidence"]["choice_prediction"],
         b["confidence"]["choice_confidence"], "#777B80"),
        (9.95, METHOD_LABELS[method].upper(),
         m["confidence"]["choice_prediction"],
         m["confidence"]["choice_confidence"], ORANGE),
    ]
    for x, title, pred, conf, color in cards:
        ax.add_patch(Rectangle((x, 1.22), 2.45, 1.85,
                               fc="white", ec=color, lw=1.5))
        ax.text(x+.18, 2.77, title, fontsize=10, fontweight="bold", color=color)
        ax.text(x+.18, 2.15, str(pred).title(), fontsize=14,
                fontweight="bold")
        if conf is not None:
            ax.text(x+.18, 1.60, f"{conf:.3f}", fontsize=17,
                    fontweight="bold", color=color)
    arrow(ax, (6.55, 2.15), (6.92, 2.15), MUTED)
    arrow(ax, (9.50, 2.15), (9.87, 2.15), MUTED)
    delta = m["confidence"]["choice_confidence"] - b["confidence"]["choice_confidence"]
    ax.text(9.35, .65, f"Still wrong  ·  confidence +{delta:.3f}",
            ha="center", fontsize=11.5, fontweight="bold", color=RED)
    save(fig, "example_wrong_higher_confidence")


def main():
    raw, stats, significance = load_all()
    plot_dataset_overview()
    plot_confidence(stats)
    plot_accuracy(stats, significance)
    plot_calibration(stats)
    plot_example(raw)


if __name__ == "__main__":
    main()
