"""Generate the poster-ready accuracy comparison table as a JPG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from poster_data import METHOD_LABELS, load_all


ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "analysis" / "figures" / "accuracy_table.jpg"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d3"
SURFACE = "#fcfcfb"
UP = "#378B6D"
DOWN = "#BE5F5F"
NEUTRAL = "#77756f"
METHOD_COLORS = {
    "baseline": "#B0B7C3",
    "vcd": "#E8895B",
    "m3id": "#4CB394",
    "icd": "#7568B3",
}

ROWS = [
    ("Age", "UTKFace", "UTKFace-Age"),
    ("Age", "PA-100K", "PA-100K-Age"),
    ("Gender", "FairFace", "FairFace-Gender"),
    ("Gender", "MSP60K", "MSP60K-Gender"),
    ("Race", "UTKFace", "UTKFace-Race"),
    ("Race", "FairFace", "FairFace-Race"),
]
METHODS = ["baseline", "vcd", "m3id", "icd"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "sans-serif"],
})


def accuracy_marker(stats, significance, dataset, method):
    if method == "baseline":
        return "", INK
    if significance[dataset][method]["accuracy"] == "ns":
        return "–", NEUTRAL
    delta = stats[dataset][method]["accuracy"] - stats[dataset]["baseline"]["accuracy"]
    return ("↑", UP) if delta > 0 else ("↓", DOWN)


def generate_accuracy_table():
    _, stats, significance = load_all()

    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.035, 0.985
    top, bottom = 0.92, 0.15
    header_h = 0.13
    row_h = (top - bottom - header_h) / len(ROWS)
    widths = [0.14, 0.20, 0.165, 0.165, 0.165, 0.165]
    xs = [left]
    for width in widths:
        xs.append(xs[-1] + width * (right - left))

    headers = ["Attribute", "Dataset"] + [METHOD_LABELS[m] for m in METHODS]
    for col, header in enumerate(headers):
        x0, x1 = xs[col], xs[col + 1]
        color = "#31302d" if col < 2 else METHOD_COLORS[METHODS[col - 2]]
        ax.add_patch(Rectangle((x0, top - header_h), x1 - x0, header_h,
                               facecolor=color, edgecolor=SURFACE, linewidth=1.5))
        ax.text((x0 + x1) / 2, top - header_h / 2, header,
                ha="center", va="center", fontsize=14.5, fontweight="bold",
                color="white" if col < 2 or col == 5 else INK)

    group_fills = ["#f4f7fb", "#f8f5fb", "#f4f8f5"]
    for row_idx, (attribute, dataset_label, dataset_key) in enumerate(ROWS):
        y1 = top - header_h - row_idx * row_h
        y0 = y1 - row_h
        fill = group_fills[row_idx // 2]
        ax.add_patch(Rectangle((left, y0), right - left, row_h,
                               facecolor=fill, edgecolor="none"))

        # Attribute appears once at the visual center of its two-row group.
        if row_idx % 2 == 0:
            group_y0 = y0 - row_h
            ax.add_patch(Rectangle((xs[0], group_y0), xs[1] - xs[0], row_h * 2,
                                   facecolor=fill, edgecolor=GRID, linewidth=0.8))
            ax.text((xs[0] + xs[1]) / 2, group_y0 + row_h,
                    attribute, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=INK)

        ax.text((xs[1] + xs[2]) / 2, (y0 + y1) / 2,
                dataset_label, ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=INK)

        for method_idx, method in enumerate(METHODS):
            col = method_idx + 2
            value = stats[dataset_key][method]["accuracy"]
            marker, marker_color = accuracy_marker(stats, significance, dataset_key, method)
            center_x = (xs[col] + xs[col + 1]) / 2
            if marker:
                ax.text(center_x - 0.008, (y0 + y1) / 2, f"{value:.3f}",
                        ha="right", va="center", fontsize=13.5, fontweight="bold", color=INK)
                ax.text(center_x + 0.016, (y0 + y1) / 2, marker,
                        ha="left", va="center", fontsize=15.5, fontweight="bold",
                        color=marker_color)
            else:
                ax.text(center_x, (y0 + y1) / 2, f"{value:.3f}",
                        ha="center", va="center", fontsize=13.5, fontweight="bold", color=INK)

        # Inside an attribute pair, keep the row divider out of the merged
        # Attribute cell so it does not strike through the label.
        line_left = xs[1] if row_idx % 2 == 0 else left
        ax.plot([line_left, right], [y0, y0], color=GRID, linewidth=0.8)

    for x in xs:
        ax.plot([x, x], [bottom, top], color=GRID, linewidth=0.8)
    ax.plot([left, right], [top, top], color=GRID, linewidth=0.8)

    ax.text(
        (left + right) / 2, 0.075,
        "↑ significant increase     ↓ significant decrease     – not significant",
        ha="center", va="center", fontsize=12, color=INK_SECONDARY,
    )

    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(
        OUTPUT, dpi=300, bbox_inches="tight",
        facecolor=SURFACE,
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    generate_accuracy_table()
