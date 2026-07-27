"""Poster infographic for the three contrastive decoding methods."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "analysis" / "figures" / "decoding_methods.jpg"

INK = "#111111"
MUTED = "#5b5a56"
SURFACE = "#fcfcfb"
PANEL = "#ffffff"
VCD = "#E8895B"
M3ID = "#4CB394"
ICD = "#7568B3"

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "sans-serif"],
})


def rounded(ax, x, y, w, h, face, edge="#d8d7d2", lw=1.3, radius=0.035, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=13, linewidth=1.4, color=MUTED, zorder=5,
    ))


def image_icon(ax, x, y, w, h, noisy=False, blank=False):
    face = "#181818" if blank else "#eaf2f7"
    rounded(ax, x, y, w, h, face, edge="#b8b7b2", lw=1.0, radius=0.02, z=3)
    if blank:
        ax.text(x + w / 2, y + h / 2, "blank", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=4)
        return
    ax.add_patch(Circle((x + w * 0.72, y + h * 0.72), h * 0.11,
                        facecolor="#f2b84b", edgecolor="none", zorder=4))
    ax.add_patch(plt.Polygon(
        [(x + w * 0.08, y + h * 0.12), (x + w * 0.42, y + h * 0.60),
         (x + w * 0.60, y + h * 0.35), (x + w * 0.90, y + h * 0.12)],
        closed=True, facecolor="#5b9f73", edgecolor="none", zorder=4,
    ))
    if noisy:
        rng = np.random.default_rng(7)
        for px, py in rng.uniform([x, y], [x + w, y + h], size=(42, 2)):
            ax.add_patch(Circle((px, py), 0.0035, facecolor="#8a5a9d",
                                edgecolor="none", alpha=0.55, zorder=5))


def prompt_icon(ax, x, y, w, h, disturbed=False):
    face = "#f5eef9" if disturbed else "#eef3f7"
    rounded(ax, x, y, w, h, face, edge="#b8b7b2", lw=1.0, radius=0.02, z=3)
    ax.text(x + w * 0.10, y + h * 0.67,
            "Confused instruction" if disturbed else "Standard instruction",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=INK, zorder=4)
    ax.plot([x + w * 0.10, x + w * 0.88], [y + h * 0.40] * 2,
            color="#aaa9a4", linewidth=1.2, zorder=4)
    ax.plot([x + w * 0.10, x + w * 0.68], [y + h * 0.23] * 2,
            color="#c2c1bd", linewidth=1.2, zorder=4)


def logit_box(ax, x, y, label, color):
    rounded(ax, x, y, 0.095, 0.105, "#f8f8f6", edge=color, lw=1.5, radius=0.018, z=3)
    ax.text(x + 0.0475, y + 0.0525, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK, zorder=4)


def method_card(ax, x, color, acronym, full_name, branch_labels, formula, takeaway, kind):
    w, y, h = 0.295, 0.16, 0.76
    rounded(ax, x, y, w, h, PANEL, edge="#d3d2cd", lw=1.2, radius=0.028)
    rounded(ax, x, y + h - 0.13, w, 0.13, color, edge=color, lw=0, radius=0.025, z=2)
    ax.add_patch(Rectangle((x, y + h - 0.13), w, 0.055, facecolor=color,
                           edgecolor="none", zorder=2))
    ax.text(x + 0.025, y + h - 0.052, acronym, ha="left", va="center",
            fontsize=19, fontweight="bold", color=INK, zorder=4)
    ax.text(x + 0.025, y + h - 0.098, full_name, ha="left", va="center",
            fontsize=10, fontweight="bold", color=INK, zorder=4)

    icon_x, icon_w, icon_h = x + 0.025, 0.105, 0.115
    top_y, bot_y = y + 0.445, y + 0.290
    if kind == "vcd":
        image_icon(ax, icon_x, top_y, icon_w, icon_h)
        image_icon(ax, icon_x, bot_y, icon_w, icon_h, noisy=True)
    elif kind == "m3id":
        image_icon(ax, icon_x, top_y, icon_w, icon_h)
        image_icon(ax, icon_x, bot_y, icon_w, icon_h, blank=True)
    else:
        prompt_icon(ax, icon_x, top_y, icon_w, icon_h)
        prompt_icon(ax, icon_x, bot_y, icon_w, icon_h, disturbed=True)

    ax.text(icon_x + icon_w / 2, top_y + icon_h + 0.018, branch_labels[0],
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.text(icon_x + icon_w / 2, bot_y + icon_h + 0.012, branch_labels[1],
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    lx = x + 0.175
    logit_box(ax, lx, top_y + 0.005, "L+", color)
    logit_box(ax, lx, bot_y + 0.005, "L−", color)
    arrow(ax, icon_x + icon_w + 0.008, top_y + icon_h / 2, lx - 0.008, top_y + icon_h / 2)
    arrow(ax, icon_x + icon_w + 0.008, bot_y + icon_h / 2, lx - 0.008, bot_y + icon_h / 2)

    rounded(ax, x + 0.025, y + 0.195, w - 0.05, 0.105,
            "#f7f7f4", edge="#deddd8", lw=1.0, radius=0.018, z=2)
    ax.text(x + w / 2, y + 0.247, formula, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=INK, zorder=4)

    ax.text(x + w / 2, y + 0.102, takeaway, ha="center", va="center",
            fontsize=10.5, color=MUTED, fontweight="bold", zorder=4)


fig, ax = plt.subplots(figsize=(14.2, 5.7))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

method_card(
    ax, 0.025, VCD, "VCD", "Visual Contrastive Decoding",
    ("Clean image", "Diffusion-noised image"),
    "L = (1 + α)L+ − αL−",
    "Suppress image-insensitive tokens",
    "vcd",
)
method_card(
    ax, 0.3525, M3ID, "M3ID", "Multi-Modal Mutual-Information Decoding",
    ("Image present", "Image removed"),
    "L = L+ + λ(L+ − L−)",
    "Amplify image-grounded evidence",
    "m3id",
)
method_card(
    ax, 0.68, ICD, "ICD", "Instruction Contrastive Decoding",
    ("Standard prompt", "Disturbed prompt"),
    "L = (1 + α)L+ − αL−",
    "Suppress instruction-induced confusion",
    "icd",
)

rounded(ax, 0.18, 0.035, 0.64, 0.075, "#f0f1ee", edge="#d4d3ce",
        lw=1.0, radius=0.02, z=2)
ax.text(
    0.5, 0.072,
    "Plausibility filtering  →  option-letter probabilities  →  cyclic-order aggregation",
    ha="center", va="center", fontsize=12, fontweight="bold", color=INK, zorder=4,
)
for center in [0.1725, 0.5, 0.8275]:
    arrow(ax, center, 0.16, center, 0.118)

plt.tight_layout(pad=0.5)
OUTPUT.parent.mkdir(exist_ok=True)
plt.savefig(
    OUTPUT, dpi=300, bbox_inches="tight", facecolor=SURFACE,
    pil_kwargs={"quality": 95, "subsampling": 0},
)
plt.close(fig)
print(f"saved {OUTPUT}")
