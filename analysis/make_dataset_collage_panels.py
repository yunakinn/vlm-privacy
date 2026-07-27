"""Arrange the supplied face and pedestrian collages as a two-panel figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "analysis" / "figures"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "svg.fonttype": "none",
})


def main():
    panels = [
        ("(a) Facial Images", FIGURES / "collage_faces.png"),
        ("(b) Pedestrian Images", FIGURES / "collage_context.png"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.25))
    for ax, (title, path) in zip(axes, panels):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(title, fontsize=23, fontweight="bold", pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#202124")
            spine.set_linewidth(1.0)

    fig.subplots_adjust(left=.012, right=.988, top=.92, bottom=.015, wspace=.18)
    fig.savefig(
        FIGURES / "dataset_overview_collages.png",
        dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.03,
    )
    fig.savefig(
        FIGURES / "dataset_overview_collages.svg",
        transparent=True, bbox_inches="tight", pad_inches=.03,
    )
    plt.close(fig)
    print("created dataset_overview_collages.png and SVG")


if __name__ == "__main__":
    main()
