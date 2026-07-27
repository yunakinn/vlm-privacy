"""Generate the vertical evaluation-pipeline diagram for the poster."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "analysis" / "figures" / "evaluation_pipeline.jpg"

TEAL = "#9FCFCB"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
BORDER = "#CDD2D8"
PANEL_FILL = WHITE = "#ffffff"
LIGHT_GREEN = "#E8F0DE"

FS_STAGE = 13
FS_PRIMARY = 15
FS_BODY = 10.5
FS_METHOD = 12.5
ARROW_GAP = 0.08

BASELINE, VCD, M3ID, ICD = "#B0B7C3", "#E8895B", "#4CB394", "#7568B3"
METHODS = [("Baseline", BASELINE), ("VCD", VCD), ("M3ID", M3ID), ("ICD", ICD)]

CHOICE_COLORS = ["#cde2fb", "#fbdac5", "#c9f0e2", "#fbe8b8", "#e6dcf5"]
CHOICE_LETTERS = ["A", "B", "C", "D", "E"]

OUTER_X, OUTER_W = 0.9, 5.6
CENTER = OUTER_X + OUTER_W / 2

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "sans-serif"],
})

fig, ax = plt.subplots(figsize=(7.6, 13.6))
ax.set_xlim(0, 7.4)
ax.set_ylim(0.3, 19.2)
ax.axis("off")


def panel(xy, w, h, title_text):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.3, edgecolor=BORDER, facecolor=PANEL_FILL, zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h - 0.32, title_text,
        ha="center", va="center", fontsize=FS_STAGE,
        fontweight="bold", color=INK, zorder=4,
    )


def box(xy, w, h, lines, facecolor, edgecolor=BORDER, zorder=3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor, zorder=zorder,
    )
    ax.add_patch(patch)
    step = h / (len(lines) + 1)
    for idx, (text, fontsize, weight, color) in enumerate(lines):
        ax.text(
            x + w / 2, y + h - step * (idx + 1), text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=color, zorder=zorder + 1,
        )


def circle_node(cx, cy, radius, label, color):
    ax.add_patch(Circle(
        (cx, cy), radius, facecolor=color,
        edgecolor=INK, linewidth=1.25, zorder=3,
    ))
    ax.text(
        cx, cy, label, ha="center", va="center",
        fontsize=11, fontweight="bold", color=INK, zorder=4,
    )


def arrow(start, end, color=INK_SECONDARY, lw=1.7):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, zorder=6, shrinkA=0, shrinkB=0,
    ))


# Input
box(
    (OUTER_X, 17.6), OUTER_W, 1.0,
    [("Input Image + Question", FS_PRIMARY, "bold", INK)],
    WHITE, BORDER,
)
arrow((CENTER, 17.6 - ARROW_GAP), (CENTER, 17.25 + ARROW_GAP))

# Multiple-choice options
panel((OUTER_X, 15.1), OUTER_W, 2.15, "Multiple-Choice Options (2–5-way)")
node_xs = [OUTER_X + 0.95 + idx * 1.0 for idx in range(5)]
for cx, letter, color in zip(node_xs, CHOICE_LETTERS, CHOICE_COLORS):
    circle_node(cx, 16.0, 0.4, letter, color)
arrow((CENTER, 15.1 - ARROW_GAP), (CENTER, 14.75 + ARROW_GAP))

# Cyclic permutation
panel((OUTER_X, 11.65), OUTER_W, 3.1, "I. Cyclic Answer-Order Permutation")
cell = 0.5
grid_x0 = 2.55
grid_y0 = 13.75
for row in range(3):
    ax.text(
        1.55, grid_y0 - row * (cell + 0.14), f"Order {row + 1}",
        ha="left", va="center", fontsize=11, color=INK_SECONDARY,
    )
    for col in range(5):
        idx = (col - row) % 5
        xx = grid_x0 + col * (cell + 0.08)
        yy = grid_y0 - row * (cell + 0.14) - cell / 2
        ax.add_patch(Rectangle(
            (xx, yy), cell, cell, facecolor=CHOICE_COLORS[idx],
            edgecolor=INK, linewidth=1.1, zorder=3,
        ))
        ax.text(
            xx + cell / 2, yy + cell / 2, CHOICE_LETTERS[idx],
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=4,
        )
arrow((CENTER, 11.65 - ARROW_GAP), (CENTER, 10.95 + ARROW_GAP))

# Vision-language inference
box(
    (OUTER_X, 9.65), OUTER_W, 1.6,
    [
        ("II. Vision–Language Inference", FS_STAGE, "bold", INK),
        ("LLaVA-1.5-7B", FS_PRIMARY, "bold", INK),
    ],
    WHITE, BORDER,
)

# Methods
method_w, gap = 1.25, 0.2
method_centers = []
for idx, (name, color) in enumerate(METHODS):
    xx = OUTER_X + idx * (method_w + gap)
    box((xx, 8.40), method_w, 0.85,
        [(name, FS_METHOD, "bold", INK)], color, color)
    arrow((CENTER, 9.65 - ARROW_GAP), (xx + method_w / 2, 9.25 + ARROW_GAP))
    method_centers.append(xx + method_w / 2)

# Aggregation and calibration
panel_y, panel_h = 3.45, 4.55
panel((OUTER_X, panel_y), OUTER_W, panel_h, "III. Aggregation & Calibration")

bus_y = 8.22
for cx in method_centers:
    ax.plot([cx, cx], [8.40 - ARROW_GAP, bus_y],
            color=INK_SECONDARY, linewidth=1.5, zorder=2)
ax.plot(
    [method_centers[0], method_centers[-1]], [bus_y, bus_y],
    color=INK_SECONDARY, linewidth=1.5, zorder=2,
)
arrow((CENTER, bus_y), (CENTER, panel_y + panel_h + ARROW_GAP))
arrow((CENTER, 7.42), (CENTER, 6.90 + ARROW_GAP))

box(
    (OUTER_X + 0.25, 6.10), OUTER_W - 0.5, 0.8,
    [("Aggregated Choice Probabilities", FS_METHOD, "bold", INK)],
    TEAL, TEAL,
)
arrow((CENTER, 6.10 - ARROW_GAP), (CENTER, 5.75 + ARROW_GAP))
box(
    (OUTER_X + 0.25, 4.95), OUTER_W - 0.5, 0.8,
    [("Prediction + Confidence", FS_METHOD, "bold", INK)],
    WHITE, BORDER,
)
arrow((CENTER, 4.95 - ARROW_GAP), (CENTER, 4.60 + ARROW_GAP))
box(
    (OUTER_X + 0.25, 3.80), OUTER_W - 0.5, 0.8,
    [("Accuracy · Confidence · ECE", FS_METHOD, "bold", INK)],
    LIGHT_GREEN, BORDER,
)

# Legend
legend_y = 2.95
legend_item_w = 1.25
legend_total_w = legend_item_w * len(METHODS)
legend_x0 = CENTER - legend_total_w / 2
for idx, (name, color) in enumerate(METHODS):
    xx = legend_x0 + idx * legend_item_w
    ax.add_patch(Rectangle(
        (xx, legend_y - 0.11), 0.22, 0.22,
        facecolor=color, edgecolor="none", zorder=3,
    ))
    ax.text(
        xx + 0.31, legend_y, name, ha="left", va="center",
        fontsize=FS_BODY + 1, color=INK,
    )

plt.tight_layout()
plt.savefig(
    OUTPUT, dpi=300, bbox_inches="tight", facecolor="white",
    pil_kwargs={"quality": 95, "subsampling": 0},
)
plt.close(fig)
print(f"saved {OUTPUT}")
