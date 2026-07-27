"""Generate the poster-ready ECE delta heatmap as a JPG."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from poster_data import METHOD_LABELS, load_all


ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "analysis" / "figures" / "ece_delta_heatmap.jpg"

DATASETS = [
    ("UTKFace", "Age", "UTKFace-Age"),
    ("PA-100K", "Age", "PA-100K-Age"),
    ("FairFace", "Gender", "FairFace-Gender"),
    ("MSP60K", "Gender", "MSP60K-Gender"),
    ("UTKFace", "Race", "UTKFace-Race"),
    ("FairFace", "Race", "FairFace-Race"),
]
METHODS = ["vcd", "m3id", "icd"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "sans-serif"],
})

# Negative = better calibration (blue), positive = worse calibration (red).
CMAP = LinearSegmentedColormap.from_list(
    "ece_delta",
    ["#4F88BD", "#A4C4DF", "#fcfcfb", "#EDB5AE", "#C96C66"],
)


def generate_ece_heatmap():
    _, stats, _ = load_all()
    values = np.array([
        [
            stats[dataset_key][method]["ece"]
            - stats[dataset_key]["baseline"]["ece"]
            for _, _, dataset_key in DATASETS
        ]
        for method in METHODS
    ])

    limit = 0.10
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(figsize=(11.2, 4.25))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(values, cmap=CMAP, norm=norm, aspect="auto")

    ax.set_xticks(
        np.arange(len(DATASETS)),
        [f"{dataset}\n{attribute}" for dataset, attribute, _ in DATASETS],
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    ax.set_yticks(
        np.arange(len(METHODS)),
        [METHOD_LABELS[method] for method in METHODS],
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0, pad=12)

    # Thin white gutters keep cells distinct at poster viewing distance.
    ax.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text_color = INK
            ax.text(
                col, row, f"{value:+.3f}",
                ha="center", va="center",
                fontsize=14.5, fontweight="bold", color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal",
                            fraction=0.08, pad=0.24, aspect=38)
    colorbar.set_ticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    colorbar.ax.tick_params(labelsize=10, length=0, colors=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.set_label(
        "Calibration improves  ←   ΔECE   →  Calibration worsens",
        fontsize=12.5, fontweight="bold", color=INK, labelpad=8,
    )

    fig.subplots_adjust(left=0.12, right=0.985, top=0.96, bottom=0.29)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(
        OUTPUT, dpi=300, bbox_inches="tight",
        facecolor=SURFACE,
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    generate_ece_heatmap()
