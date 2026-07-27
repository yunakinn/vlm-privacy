"""Render the poster accuracy table from recomputed JSONL metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from poster_data import METHOD_LABELS, load_all


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "figures"
ORDER = [
    ("Age", "UTKFace–Age", "UTKFace-Age"),
    ("Age", "PA-100K–Age", "PA-100K-Age"),
    ("Gender", "FairFace–Gender", "FairFace-Gender"),
    ("Gender", "MSP60K–Gender", "MSP60K-Gender"),
    ("Race", "UTKFace–Race", "UTKFace-Race"),
    ("Race", "FairFace–Race", "FairFace-Race"),
]
METHODS = ["baseline", "vcd", "m3id", "icd"]
REFERENCE = {
    "UTKFace-Age": [64.97, 60.60, 65.67, 67.93],
    "PA-100K-Age": [45.13, 46.03, 45.67, 43.70],
    "FairFace-Gender": [94.53, 94.57, 94.83, 94.67],
    "MSP60K-Gender": [58.50, 63.67, 61.00, 58.17],
    "UTKFace-Race": [78.53, 77.70, 78.40, 77.50],
    "FairFace-Race": [54.19, 57.03, 54.74, 54.22],
}

INK = "#171717"
LIGHT = "#C8C8C8"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
})


def render_accuracy_table():
    _, stats, significance = load_all()

    for _, _, key in ORDER:
        computed = [100 * stats[key][m]["accuracy"] for m in METHODS]
        reference = REFERENCE[key]
        for method, actual, expected in zip(METHODS, computed, reference):
            if round(actual, 2) != round(expected, 2):
                print(
                    f"DISCREPANCY accuracy {key}/{method}: "
                    f"reference={expected:.2f}, recomputed={actual:.2f}; using recomputed"
                )

    fig, ax = plt.subplots(figsize=(8.0, 3.95))
    ax.set_xlim(.16, 4.70)
    ax.set_ylim(1.38, 6.52)
    ax.axis("off")

    x_setting = .38
    xs = [2.05, 2.77, 3.49, 4.21]

    # Booktabs-like header with no vertical rules.
    ax.plot([.30, 4.57], [6.40, 6.40], color=INK, lw=1.15)
    ax.text(x_setting, 6.03, "Setting", fontsize=18,
            fontweight="bold", va="center")
    for x, method in zip(xs, METHODS):
        ax.text(x, 6.03, METHOD_LABELS[method], ha="center", va="center",
                fontsize=18, fontweight="bold")
    ax.plot([.30, 4.57], [5.70, 5.70], color=INK, lw=.95)

    ys = [5.24, 4.56, 3.88, 3.20, 2.52, 1.84]
    for index, ((group, label, key), y) in enumerate(zip(ORDER, ys)):
        ax.text(x_setting, y, label, fontsize=17, va="center")
        values = [100 * stats[key][m]["accuracy"] for m in METHODS]
        best = int(np.argmax(values))
        for col, (x, method, value) in enumerate(zip(xs, METHODS, values)):
            marker = ""
            if method != "baseline" and significance[key][method]["accuracy"] != "ns":
                delta = value - values[0]
                marker = "↑" if delta > 0 else "↓"
            ax.text(x, y, f"{value:.2f}", ha="center", va="center",
                    fontsize=17, fontweight="bold" if col == best else "normal",
                    color=INK)
            if marker:
                ax.text(x + .195, y + .01, marker, ha="center", va="center",
                        fontsize=12.5, color=INK)

        separator_y = y - .34
        if index < 5:
            if index in (1, 3):
                ax.plot([.30, 4.57], [separator_y, separator_y],
                        color=INK, lw=.80)
            else:
                ax.plot([.30, 4.57], [separator_y, separator_y],
                        color=LIGHT, lw=.42)

    ax.plot([.30, 4.57], [1.49, 1.49], color=INK, lw=1.15)
    fig.subplots_adjust(left=.025, right=.985, top=.985, bottom=.025)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "table1_accuracy.png", facecolor="white",
                bbox_inches="tight", pad_inches=0, dpi=300)
    fig.savefig(OUT / "table1_accuracy.svg", transparent=True,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("created table1_accuracy.png and tightly cropped table1_accuracy.svg")


if __name__ == "__main__":
    render_accuracy_table()
