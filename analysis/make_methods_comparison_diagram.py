"""Wide, publication-style comparison of four VLM decoding methods."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "figures"
PNG = OUT / "methods_comparison_diagram.png"
SHARED_PHOTO = ROOT / "data" / "multipriv_en160" / "images" / "person_7_B.png"

INK = "#252525"
RED = "#8F1D22"
GRAY = "#666666"
LIGHT = "#D9D9D9"
PALE = "#F4F5F5"
BLUE = "#557B9D"
ORANGE = "#C9793C"
GREEN = "#4D8A69"

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
    "svg.fonttype": "none",
})


def box(ax, x, y, w, h, fc="white", ec=LIGHT, lw=0.9):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.1):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, shrinkA=1, shrinkB=1,
    ))


def portrait(ax, x, y, w=0.78, h=0.78, noisy=False):
    """Small neutral portrait icon. Noise is used only for the VCD auxiliary path."""
    box(ax, x, y, w, h, fc="#EAF0F3", ec="#BFC8CF")
    ax.add_patch(Circle((x + .50*w, y + .55*h), .16*h,
                        facecolor="#D4B29B", edgecolor="none"))
    ax.add_patch(Polygon([
        (x+.32*w, y+.60*h), (x+.42*w, y+.77*h),
        (x+.59*w, y+.77*h), (x+.69*w, y+.59*h),
    ], closed=True, facecolor="#4D535A", edgecolor="none"))
    ax.add_patch(Polygon([
        (x+.20*w, y+.08*h), (x+.28*w, y+.34*h),
        (x+.72*w, y+.34*h), (x+.80*w, y+.08*h),
    ], closed=True, facecolor="#8198AC", edgecolor="none"))
    if noisy:
        rng = np.random.default_rng(17)
        for px, py in rng.uniform([x, y], [x+w, y+h], (30, 2)):
            ax.add_patch(Circle((px, py), .010, color=ORANGE, alpha=.55))


def shared_real_photo(ax, x, y, w, h):
    """Crop the portrait region from a locally available real-person sample."""
    if not SHARED_PHOTO.is_file():
        portrait(ax, x, y, w, h)
        return
    image = Image.open(SHARED_PHOTO).convert("RGB")
    # The local sample is an ID layout; isolate its portrait rather than
    # exposing document text in the methods illustration.
    left = int(image.width * .65)
    crop = image.crop((left, 0, image.width, image.height))
    ax.imshow(crop, extent=(x, x+w, y, y+h), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                           edgecolor="#BFC8CF", linewidth=1, zorder=3))


def model(ax, x, y, label="LLaVA"):
    box(ax, x, y, .82, .42, fc=PALE, ec="#AEB4B8")
    ax.text(x+.41, y+.21, label, ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=INK)


def scores(ax, x, y, color=BLUE, label="Scores"):
    ax.text(x+.33, y+.42, label, ha="center", fontsize=6.3,
            fontweight="bold", color=GRAY)
    for i, width in enumerate((.50, .35, .22)):
        ax.add_patch(Rectangle((x, y+.27-i*.12), width, .055,
                               facecolor=color, edgecolor="none",
                               alpha=1-i*.15))


def output(ax, x, y, confidence):
    box(ax, x, y, 1.42, .68, fc="#F2F7F3", ec=GREEN, lw=1.1)
    ax.text(x+.12, y+.48, "PREDICTION", fontsize=5.7, color=GRAY,
            fontweight="bold", va="center")
    ax.text(x+.12, y+.28, "Young Adult", fontsize=8.2, color=INK,
            fontweight="bold", va="center")
    ax.text(x+.12, y+.10, f"Confidence  {confidence}", fontsize=6.4,
            color=GREEN, fontweight="bold", va="center")


def heading(ax, cx, name, subtitle):
    ax.text(cx, 5.26, name, ha="center", va="center", fontsize=14.5,
            fontweight="bold", color=RED)
    ax.text(cx, 4.93, subtitle, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color=INK)
    ax.plot([cx-1.25, cx+1.25], [4.77, 4.77], color=RED, lw=1.1)


fig, ax = plt.subplots(figsize=(18, 5.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 18)
ax.set_ylim(0, 5.8)
ax.axis("off")

# Five equal visual sections: shared input plus four decoding methods.
dividers = [2.8, 6.6, 10.4, 14.2]
for x in dividers:
    ax.plot([x, x], [.30, 5.48], color=LIGHT, lw=.9)

# Shared input
ax.text(1.40, 5.26, "SHARED INPUT", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=RED)
shared_real_photo(ax, .91, 3.93, .98, .98)
ax.text(1.40, 3.67, "Sensitive-attribute query", ha="center",
        fontsize=7.1, fontweight="bold", color=GRAY)
box(ax, .38, 2.45, 2.04, .92, fc=PALE, ec="#BFC3C5")
ax.text(1.40, 3.14, "What is the age group\nof this person?",
        ha="center", va="center", fontsize=8.2, fontweight="bold",
        color=INK, linespacing=1.25)
ax.text(.54, 2.75, "Choices", fontsize=6.1, fontweight="bold", color=RED)
ax.text(.54, 2.57, "Child · Teen · Young Adult\nMiddle-aged · Older Adult",
        fontsize=6.3, color=GRAY, va="top", linespacing=1.25)
ax.text(1.40, 1.93, "Same image + question\nfor every decoding method",
        ha="center", va="center", fontsize=7.1, color=GRAY, linespacing=1.35)
for y in (4.25, 3.55, 2.85, 2.15):
    arrow(ax, 2.51, y, 2.73, y, color=RED)

# Baseline
heading(ax, 4.70, "BASELINE", "Standard Decoding")
portrait(ax, 3.10, 3.73, .66, .66)
ax.text(3.43, 3.55, "Original", ha="center", fontsize=5.9, color=GRAY)
arrow(ax, 3.82, 4.06, 4.10, 4.06, BLUE)
model(ax, 4.14, 3.85, "LLaVA-1.5-7B")
arrow(ax, 5.00, 4.06, 5.26, 4.06, BLUE)
scores(ax, 5.31, 3.78, BLUE, "Choice Scores")
arrow(ax, 5.92, 4.06, 6.10, 4.06, BLUE)
output(ax, 4.00, 2.55, "0.50")
arrow(ax, 4.70, 3.58, 4.70, 3.28, BLUE)
ax.text(4.70, 1.88, "Directly ranks answer choices using\nthe image-conditioned model scores.",
        ha="center", va="center", fontsize=7.3, color=GRAY, linespacing=1.3)

# VCD
heading(ax, 8.50, "VCD", "Contrast with distorted image")
portrait(ax, 6.91, 4.02, .62, .62)
portrait(ax, 6.91, 3.12, .62, .62, noisy=True)
ax.text(7.22, 3.91, "Original", ha="center", fontsize=5.5, color=BLUE)
ax.text(7.22, 3.01, "Distorted", ha="center", fontsize=5.5, color=ORANGE)
arrow(ax, 7.58, 4.33, 7.82, 4.33, BLUE)
arrow(ax, 7.58, 3.43, 7.82, 3.43, ORANGE)
model(ax, 7.86, 4.12)
model(ax, 7.86, 3.22)
scores(ax, 8.82, 4.05, BLUE, "Original logits")
scores(ax, 8.82, 3.15, ORANGE, "Distorted logits")
arrow(ax, 9.46, 4.32, 9.73, 3.84, BLUE)
arrow(ax, 9.46, 3.42, 9.73, 3.75, ORANGE)
box(ax, 9.72, 3.52, .48, .48, fc=PALE, ec="#AEB4B8")
ax.text(9.96, 3.76, "−", ha="center", va="center",
        fontsize=15, fontweight="bold", color=RED)
arrow(ax, 9.96, 3.47, 9.96, 3.20, GREEN)
output(ax, 8.00, 2.31, "0.56")
ax.text(8.50, 1.63, "Suppresses outputs that remain likely\nwithout reliable visual evidence.",
        ha="center", va="center", fontsize=7.2, color=GRAY, linespacing=1.3)

# M3ID
heading(ax, 12.30, "M3ID", "Favor image-grounded outputs")
ax.text(10.75, 4.44, "Image-conditioned", fontsize=6.2,
        fontweight="bold", color=BLUE)
ax.text(10.75, 3.48, "Language-prior", fontsize=6.2,
        fontweight="bold", color=ORANGE)
scores(ax, 10.80, 3.88, BLUE, "Scores")
scores(ax, 10.80, 2.92, ORANGE, "Scores")
ax.text(11.66, 3.66, "vs.", fontsize=8.2, color=RED, fontweight="bold")
arrow(ax, 11.55, 4.10, 12.12, 3.72, BLUE)
arrow(ax, 11.55, 3.14, 12.12, 3.58, ORANGE)
box(ax, 12.12, 3.34, 1.30, .58, fc=PALE, ec="#AEB4B8")
ax.text(12.77, 3.63, "MI score\nadjustment", ha="center",
        va="center", fontsize=7.0, fontweight="bold", color=INK)
arrow(ax, 13.46, 3.63, 13.73, 3.63, GREEN)
output(ax, 11.59, 2.31, "0.55")
ax.text(12.30, 1.63, "Favors outputs supported by the visual\ninput rather than language priors.",
        ha="center", va="center", fontsize=7.2, color=GRAY, linespacing=1.3)

# ICD
heading(ax, 16.10, "ICD", "Contrastive instruction guidance")
box(ax, 14.55, 4.05, 1.18, .50, fc="#EEF3F7", ec=BLUE)
box(ax, 14.55, 3.10, 1.18, .50, fc="#FAF1EA", ec=ORANGE)
ax.text(15.14, 4.30, "Original instruction", ha="center", va="center",
        fontsize=6.3, fontweight="bold", color=BLUE)
ax.text(15.14, 3.35, "Contrastive instruction", ha="center", va="center",
        fontsize=6.0, fontweight="bold", color=ORANGE)
arrow(ax, 15.77, 4.30, 16.04, 4.30, BLUE)
arrow(ax, 15.77, 3.35, 16.04, 3.35, ORANGE)
scores(ax, 16.08, 4.02, BLUE, "Scores")
scores(ax, 16.08, 3.07, ORANGE, "Scores")
arrow(ax, 16.73, 4.28, 17.06, 3.78, BLUE)
arrow(ax, 16.73, 3.33, 17.06, 3.68, ORANGE)
box(ax, 17.04, 3.44, .58, .48, fc=PALE, ec="#AEB4B8")
ax.text(17.33, 3.68, "Δ", ha="center", va="center",
        fontsize=10, fontweight="bold", color=RED)
arrow(ax, 17.33, 3.39, 17.33, 3.16, GREEN)
output(ax, 15.39, 2.31, "0.53")
ax.text(16.10, 1.63, "Uses instruction-level contrast to suppress\nundesirable decoding behavior.",
        ha="center", va="center", fontsize=7.2, color=GRAY, linespacing=1.3)

# Shared scientific-figure footer.
ax.plot([.35, 17.65], [1.12, 1.12], color=LIGHT, lw=.8)
ax.text(9.0, .80,
        "All four methods operate at decoding time; only VCD uses a distorted-image reference.",
        ha="center", va="center", fontsize=7.4, color=GRAY)

OUT.mkdir(parents=True, exist_ok=True)
fig.subplots_adjust(left=.012, right=.988, top=.98, bottom=.06)
fig.savefig(PNG, dpi=300, facecolor="white")
plt.close(fig)
print(f"saved {PNG}")
