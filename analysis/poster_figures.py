"""
Poster result figures — built from real predictions via poster_data.load_all().

Produces, into analysis/figures/:
  fig0_confidence_main.jpg      "Confidence Across Dataset–Attribute Settings"
  fig1_confidence_increase.png  "Confidence Increased in 17 of 18 Comparisons"
  fig2_accuracy_inconsistent.png "Accuracy Effects Were Inconsistent"
  fig3_calibration_varied.png    "Calibration Varied by Setting"
  fig4_dataset_overview.png      4 dataset info cards

Palette (validated with dataviz-skill validator, light surface #fcfcfb):
  method identity : Baseline gray, VCD orange, M3ID aqua, ICD violet
  delta / status  : increase red, decrease blue, not-significant gray
Consistent across all 4 figures so a color always means the same thing.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from poster_data import load_all, METHOD_LABELS, MITIGATION_METHODS

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(exist_ok=True)

DATASET_ORDER = [
    "UTKFace-Age", "UTKFace-Race", "PA-100K-Age",
    "FairFace-Gender", "FairFace-Race", "MSP60K-Gender",
]

METHOD_COLORS = {
    "baseline": "#B0B7C3",
    "vcd": "#E8895B",
    "m3id": "#4CB394",
    "icd": "#7568B3",
}
INCREASE = "#D77878"
DECREASE = "#6698C8"
NEUTRAL = "#C7C8C4"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "sans-serif"],
    "font.size": 12,
    "text.color": INK,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _row_labels(stats):
    """18 (dataset, method) rows in fixed dataset/method order."""
    rows = []
    for ds in DATASET_ORDER:
        for m in MITIGATION_METHODS:
            if m in stats[ds]:
                rows.append((ds, m))
    return rows


def plot_confidence_main(stats, significance):
    """Large poster hero chart, grouped by attribute."""
    methods = ["baseline", "vcd", "m3id", "icd"]
    chart_order = [
        "UTKFace-Age", "PA-100K-Age",
        "FairFace-Gender", "MSP60K-Gender",
        "UTKFace-Race", "FairFace-Race",
    ]
    # A modest extra gap separates attributes without making the chart sparse.
    x = np.array([0.0, 1.0, 2.28, 3.28, 4.56, 5.56])
    width = 0.17
    bar_pitch = 0.22

    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    bars_by_method = {}
    for j, method in enumerate(methods):
        values = [stats[ds][method]["confidence"] for ds in chart_order]
        offsets = x + (j - 1.5) * bar_pitch
        bars = ax.bar(
            offsets, values, width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=3,
        )
        bars_by_method[method] = bars

        # One consistent clearance keeps every numeric label aligned.
        label_lifts = [0.015, 0.015, 0.015, 0.015]
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_lifts[j],
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.7,
                color=INK,
                fontweight="bold",
                zorder=5,
            )

    ax.set_xticks(
        x,
        ["UTKFace", "PA-100K", "FairFace", "MSP60K", "UTKFace", "FairFace"],
        fontsize=13.5,
        fontweight="bold",
    )
    for center, attribute in zip(
        [(x[0] + x[1]) / 2, (x[2] + x[3]) / 2, (x[4] + x[5]) / 2],
        ["Age", "Gender", "Race"],
    ):
        ax.text(
            center, -0.17, attribute,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=14, fontweight="bold", color=INK,
        )

    ax.set_ylabel("Mean Confidence", fontsize=14.5, fontweight="bold")
    ax.set_ylim(0, 0.84)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(axis="y", length=0, labelsize=13)
    ax.tick_params(axis="x", length=0, pad=9)

    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        fontsize=13.5,
        handlelength=1.4,
        columnspacing=2.2,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.24)
    output = OUT_DIR / "fig0_confidence_main.jpg"
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    print(f"saved {output}")
    plt.close(fig)


def _diverging_bar_chart(rows, values, sig_stars, filename, title, subtitle,
                          xlabel, legend_labels, highlight_key=None, highlight_note=None):
    """Horizontal diverging bar chart. Dataset names appear as their own bold
    header rows (no bar) interleaved with the method rows below them — avoids
    fragile manual text-transform placement entirely."""
    # build a flat row list: ("header", ds) then ("bar", ds, m) x3, per dataset
    row_kinds = []
    for ds in DATASET_ORDER:
        row_kinds.append(("header", ds, None))
        for m in MITIGATION_METHODS:
            if (ds, m) in rows:
                row_kinds.append(("bar", ds, m))
    val_lookup = {(ds, m): v for (ds, m), v in zip(rows, values)}

    n_total = len(row_kinds)
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * n_total + 1.0))
    y = np.arange(n_total)[::-1]  # top row = first dataset

    yticklabels, header_ys, bar_y, bar_vals, bar_colors, bar_meta = [], [], [], [], [], []
    for yy, (kind, ds, m) in zip(y, row_kinds):
        if kind == "header":
            yticklabels.append(ds)
            header_ys.append(yy)
        else:
            yticklabels.append("   " + METHOD_LABELS[m])
            v = val_lookup[(ds, m)]
            sig = sig_stars.get((ds, m), "ns")
            bar_y.append(yy)
            bar_vals.append(v)
            bar_colors.append(NEUTRAL if sig == "ns" else (INCREASE if v > 0 else DECREASE))
            bar_meta.append((ds, m))

    bar_y = np.array(bar_y)
    bar_vals = np.array(bar_vals)

    # zebra band behind each dataset's 4-row block (header + 3 methods)
    for gi, ds in enumerate(DATASET_ORDER):
        if gi % 2 == 1:
            idxs = [i for i, (k, d, m) in enumerate(row_kinds) if d == ds]
            top = y[idxs[0]] + 0.5
            bot = y[idxs[-1]] - 0.5
            ax.axhspan(bot, top, color="#f1f0ec", zorder=0)

    bars = ax.barh(bar_y, bar_vals, height=0.62, color=bar_colors, edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels, fontsize=9.5)
    for tick, yy in zip(ax.get_yticklabels(), y):
        if yy in header_ys:
            tick.set_fontweight("bold")
            tick.set_fontsize(10)
        else:
            tick.set_color(INK_SECONDARY)

    span = max(abs(bar_vals.min()), abs(bar_vals.max())) * 1.55 + 0.015
    ax.set_xlim(-span, span)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    for v, yy, meta in zip(bar_vals, bar_y, bar_meta):
        sig = sig_stars.get(meta, "ns")
        label = f"{v:+.3f}" + (f" {sig}" if sig != "ns" else "")
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), yy, label,
                ha="left" if v >= 0 else "right", va="center", fontsize=8, color=INK_SECONDARY, zorder=4)

    if highlight_key is not None and highlight_key in bar_meta:
        hi = bar_meta.index(highlight_key)
        bars[hi].set_edgecolor(INK)
        bars[hi].set_linewidth(1.8)
        if highlight_note:
            hy = bar_y[hi]
            tx = span * 0.42
            ax.annotate(highlight_note, xy=(span * 0.03, hy), xytext=(tx, hy),
                        fontsize=8.5, color=INK, ha="left", va="center", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1))

    ax.axvline(0, color=INK, linewidth=1.1, zorder=2)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-1, n_total)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)

    handles = [mpatches.Patch(color=c, label=l) for c, l in legend_labels]
    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.065), ncol=len(handles), fontsize=9.5)

    ax.set_title(subtitle, fontsize=9, color=INK_SECONDARY, pad=12, loc="center")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.28, right=0.95, top=0.90, bottom=0.10)
    fig.savefig(OUT_DIR / filename, dpi=220)
    plt.close(fig)
    print(f"saved {OUT_DIR / filename}")


def plot_confidence_increase(stats, significance):
    """Poster hero chart: three aligned attribute facets on one shared scale."""
    groups = [
        ("Age", [("UTKFace", "UTKFace-Age"), ("PA-100K", "PA-100K-Age")]),
        ("Gender", [("FairFace", "FairFace-Gender"), ("MSP60K", "MSP60K-Gender")]),
        ("Race", [("UTKFace", "UTKFace-Race"), ("FairFace", "FairFace-Race")]),
    ]
    minus = "\N{MINUS SIGN}"
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharex=True, sharey=True)
    y_positions = [5.0, 4.25, 3.5, 2.0, 1.25, 0.5]

    for panel_idx, (ax, (attribute, datasets)) in enumerate(zip(axes, groups)):
        values = []
        for _, dataset_key in datasets:
            values.extend([
                stats[dataset_key][method]["confidence"]
                - stats[dataset_key]["baseline"]["confidence"]
                for method in MITIGATION_METHODS
            ])

        colors = [INCREASE if value >= 0 else DECREASE for value in values]
        ax.barh(
            y_positions, values, height=0.52,
            color=colors, edgecolor="none", linewidth=0, zorder=3,
        )

        for yy, value in zip(y_positions, values):
            label = f"+{value:.3f}" if value >= 0 else f"{minus}{abs(value):.3f}"
            label_x = value + 0.003 if value >= 0 else 0.003
            ax.text(
                label_x, yy, label,
                ha="left",
                va="center", fontsize=11.5, fontweight="bold",
                color=INK, zorder=4,
            )
        for yy, method in zip(
            y_positions,
            ["VCD", "M3ID", "ICD", "VCD", "M3ID", "ICD"],
        ):
            ax.text(
                -0.029, yy, method,
                ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=INK, zorder=4,
            )

        ax.set_title(attribute, fontsize=18, fontweight="bold", color=INK, pad=18)

        # Dataset names act as compact sub-headings for each three-bar block.
        for dataset_label, yy in [(datasets[0][0], 5.58), (datasets[1][0], 2.58)]:
            ax.text(
                0.006, yy, dataset_label,
                ha="left", va="center", fontsize=12,
                fontweight="bold", color=INK_SECONDARY,
            )

        ax.set_xlim(-0.03, 0.16)
        ax.set_ylim(0.0, 6.25)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.2f}"))
        ax.xaxis.grid(True, color="#e8e7e2", linewidth=0.8, zorder=0)
        ax.axvline(0, color=INK, linewidth=1.25, zorder=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
        ax.tick_params(axis="x", labelsize=12, colors=INK)
        ax.tick_params(axis="y", length=0, pad=7)

        if panel_idx:
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_position(("axes", -0.10))
            ax.spines["left"].set_color("#deddd8")
            ax.spines["left"].set_linewidth(1.0)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.11, wspace=0.18)
    output = OUT_DIR / "fig1_confidence_increase.jpg"
    fig.savefig(
        output, dpi=300, bbox_inches="tight",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)
    print(f"saved {output}")


def plot_accuracy_inconsistent(stats, significance):
    rows = _row_labels(stats)
    values = np.array([stats[ds][m]["accuracy"] - stats[ds]["baseline"]["accuracy"] for ds, m in rows])
    sig_stars = {(ds, m): significance[ds][m]["accuracy"] for ds, m in rows}

    _diverging_bar_chart(
        rows, values, sig_stars, "fig2_accuracy_inconsistent.png",
        "Accuracy Effects Were Inconsistent",
        "Improves in some settings, declines in others, unchanged elsewhere",
        "Δ Accuracy vs. Baseline",
        [(INCREASE, "Increase (sig.)"), (DECREASE, "Decrease (sig.)"), (NEUTRAL, "Not significant")],
    )


def plot_calibration_varied(stats):
    methods = ["baseline", "vcd", "m3id", "icd"]
    n_ds = len(DATASET_ORDER)
    x = np.arange(n_ds)
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars_by_method = {}
    for j, m in enumerate(methods):
        vals = [stats[ds][m]["ece"] for ds in DATASET_ORDER]
        offs = x + (j - (len(methods) - 1) / 2) * width
        bars = ax.bar(offs, vals, width, label=METHOD_LABELS[m], color=METHOD_COLORS[m],
                      edgecolor="white", linewidth=0.6, zorder=3)
        bars_by_method[m] = bars

    # highlight the MSP60K-Gender / VCD headline result
    hi_ds = DATASET_ORDER.index("MSP60K-Gender")
    hi_bar = bars_by_method["vcd"][hi_ds]
    hi_bar.set_edgecolor(INK)
    hi_bar.set_linewidth(2.0)
    base_ece = stats["MSP60K-Gender"]["baseline"]["ece"]
    vcd_ece = stats["MSP60K-Gender"]["vcd"]["ece"]
    ax.annotate(
        f"VCD: {base_ece:.4f} → {vcd_ece:.4f}",
        xy=(hi_bar.get_x() + hi_bar.get_width() / 2, vcd_ece),
        xytext=(hi_bar.get_x() + hi_bar.get_width() / 2 - 1.15, vcd_ece + 0.075),
        fontsize=9, fontweight="bold", color=INK, ha="left",
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER, rotation=12, ha="right", fontsize=9.5)
    ax.set_ylabel("Expected Calibration Error (ECE) — lower is better")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9.5)

    fig.suptitle("Calibration Varied by Setting", fontsize=13.5, fontweight="bold", y=1.0)
    ax.set_title("VCD sharply improved calibration on MSP60K–Gender but worsened it elsewhere (e.g. FairFace–Race)",
                 fontsize=9.5, color=INK_SECONDARY, pad=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT_DIR / "fig3_calibration_varied.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'fig3_calibration_varied.png'}")


DATASET_CARDS = [
    {
        "name": "UTKFace",
        "source": "Face images, in-the-wild",
        "attrs": [("Age", "5-way", 3000), ("Race", "5-way", 3000)],
        "accent": "#2a78d6",
    },
    {
        "name": "FairFace",
        "source": "Face images, balanced by demographic group",
        "attrs": [("Gender", "2-way", 3000), ("Race", "7-way", 1799)],
        "accent": "#1baf7a",
    },
    {
        "name": "PA-100K",
        "source": "Pedestrian surveillance images",
        "attrs": [("Age", "3-way", 3000)],
        "accent": "#eda100",
    },
    {
        "name": "MSP60K",
        "source": "Pedestrian surveillance images",
        "attrs": [("Gender", "2-way", 600)],
        "accent": "#4a3aa7",
    },
]


def _silhouette(ax, cx, cy, r, color):
    """Small abstract person icon (head + shoulders) as a stand-in for a photo
    thumbnail — no sample images are available locally (source images live
    on the machine the model was run on)."""
    head = plt.Circle((cx, cy + r * 0.55), r * 0.42, color=color, zorder=3)
    body = mpatches.Wedge((cx, cy - r * 0.65), r * 0.95, 0, 180, color=color, zorder=3)
    ax.add_patch(head)
    ax.add_patch(body)


def plot_dataset_overview(stats):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    for ax, card in zip(axes, DATASET_CARDS):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        border = FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.01,rounding_size=0.05",
                                linewidth=1.4, edgecolor=card["accent"], facecolor="white", zorder=1)
        ax.add_patch(border)
        _silhouette(ax, 0.5, 0.74, 0.14, card["accent"])
        ax.text(0.5, 0.55, card["name"], ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(0.5, 0.47, card["source"], ha="center", va="center", fontsize=8, color=INK_SECONDARY, wrap=True)

        y = 0.34
        for attr_name, arity, n in card["attrs"]:
            ax.text(0.5, y, f"{attr_name}  ·  {arity}  ·  n={n:,}", ha="center", va="center",
                    fontsize=9.5, color=INK)
            y -= 0.1

    fig.suptitle("Dataset–Attribute Settings", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_dataset_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'fig4_dataset_overview.png'}")


if __name__ == "__main__":
    raw_preds, stats, significance = load_all()
    plot_confidence_main(stats, significance)
    plot_confidence_increase(stats, significance)
    plot_accuracy_inconsistent(stats, significance)
    plot_calibration_varied(stats)
    plot_dataset_overview(stats)
