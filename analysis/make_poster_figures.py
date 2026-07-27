"""Build real-data poster figures and report unavailable source photographs."""

from pathlib import Path
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageFilter, ImageOps

from poster_data import (
    DATASETS,
    METHOD_LABELS,
    MITIGATION_METHODS,
    compute_reliability_bins,
    load_all,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "figures"
SAMPLES = ROOT / "analysis" / "poster_samples"
ORDER = [
    ("Age", "UTKFace", "UTKFace-Age"),
    ("Age", "PA-100K", "PA-100K-Age"),
    ("Gender", "FairFace", "FairFace-Gender"),
    ("Gender", "MSP60K", "MSP60K-Gender"),
    ("Race", "UTKFace", "UTKFace-Race"),
    ("Race", "FairFace", "FairFace-Race"),
]

INK = "#202124"
MUTED = "#62676C"
GRID = "#E1E4E6"
POSITIVE = "#557F9F"
NEGATIVE = "#C9783A"
BASELINE = "#555A60"
MITIGATION = "#4B896A"
EXAMPLES = OUT / "dataset_examples"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "svg.fonttype": "none",
})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", facecolor="white",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"created {stem}.png")


def resolve_image(record):
    raw = Path(record["image"])
    candidates = [raw, ROOT / raw, ROOT / "data" / raw.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def report_images_and_select(raw):
    """Select four deterministic images per dataset, or report missing paths."""
    rng = random.Random(42)
    selected = {}
    dataset_keys = {
        "UTKFace": "UTKFace-Age",
        "FairFace": "FairFace-Gender",
        "PA-100K": "PA-100K-Age",
        "MSP60K": "MSP60K-Gender",
    }
    missing = {}
    for dataset, key in dataset_keys.items():
        records = raw[key]["baseline"]
        pool = list(records)
        rng.shuffle(pool)
        resolved = []
        missing_paths = []
        for record in pool:
            path = resolve_image(record)
            if path and path not in [p for _, p in resolved]:
                resolved.append((record, path))
                if len(resolved) == 4:
                    break
            elif len(missing_paths) < 4:
                missing_paths.append(record["image"])
        if len(resolved) < 4:
            missing[dataset] = missing_paths
        else:
            selected[dataset] = resolved

    if missing:
        print("MISSING REQUIRED DATASET IMAGES")
        for dataset, paths in missing.items():
            print(f"  {dataset}:")
            for path in paths:
                print(f"    {path}")
    return selected, missing


def copy_samples(selected):
    SAMPLES.mkdir(parents=True, exist_ok=True)
    copied = []
    for dataset, rows in selected.items():
        for i, (_, source) in enumerate(rows, 1):
            target = SAMPLES / f"{dataset.lower().replace('-', '')}_{i:02d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            copied.append(target)
            print(f"selected {dataset}: {source} -> {target}")
    return copied


def select_supplied_examples():
    """Select four real supplied samples per dataset with seed 42."""
    patterns = {
        "UTKFace": "utkface_*.jpg",
        "FairFace": "fairface_*.jpg",
        "PA-100K": "pa100k_*.jpg",
        "MSP60K": "msp60k_*.jpg",
    }
    selected, missing = {}, {}
    for offset, (dataset, pattern) in enumerate(patterns.items()):
        paths = sorted(EXAMPLES.glob(pattern))
        rng = random.Random(42 + offset)
        if len(paths) < 4:
            missing[dataset] = [str(EXAMPLES / pattern)]
            continue
        selected[dataset] = rng.sample(paths, 4)
    return selected, missing


def copy_supplied_samples(selected):
    SAMPLES.mkdir(parents=True, exist_ok=True)
    copied = []
    for dataset, paths in selected.items():
        slug = dataset.lower().replace("-", "")
        for i, source in enumerate(paths, 1):
            target = SAMPLES / f"{slug}_{i:02d}.jpg"
            shutil.copy2(source, target)
            copied.append(target)
            print(f"selected {dataset}: {source} -> {target}")
    return copied


def add_cropped_image(
        ax, path, extent, face=True, distorted=False, border_color="#BFC4C7"):
    image = Image.open(path).convert("RGB")
    w = int((extent[1] - extent[0]) * 240)
    h = int((extent[3] - extent[2]) * 240)
    if face:
        image = ImageOps.fit(image, (max(w, 40), max(h, 40)),
                             method=Image.Resampling.LANCZOS)
    else:
        image.thumbnail((max(w, 40), max(h, 40)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max(w, 40), max(h, 40)), "white")
        canvas.paste(image, ((canvas.width-image.width)//2,
                             (canvas.height-image.height)//2))
        image = canvas
    if distorted:
        image = image.filter(ImageFilter.GaussianBlur(radius=4))
        noise = np.random.default_rng(42).normal(0, 22, np.asarray(image).shape)
        image = Image.fromarray(
            np.clip(np.asarray(image).astype(float) + noise, 0, 255).astype("uint8")
        )
    ax.imshow(image, extent=extent, aspect="auto", zorder=2)
    ax.add_patch(Rectangle((extent[0], extent[2]),
                           extent[1]-extent[0], extent[3]-extent[2],
                           fill=False, edgecolor=border_color, lw=.8, zorder=3))


def flow_arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        color=color, linewidth=lw, shrinkA=2, shrinkB=2, zorder=5,
    ))


def score_bars(ax, x, y, color):
    for i, width in enumerate((.58, .40, .25)):
        ax.add_patch(Rectangle((x, y-i*.13), width, .065,
                               facecolor=color, edgecolor="none"))


def plot_pipeline(selected):
    """Readable CVPR-style overview with visual cues and restrained text."""
    input_path = EXAMPLES / "utkface_young0.jpg"
    fig, ax = plt.subplots(figsize=(7.6, 9.7))
    ax.set_xlim(0, 10); ax.set_ylim(2.08, 16); ax.axis("off")

    accent = {
        "Baseline": ("#3F4851", "#D8DEE2"),
        "VCD": ("#C94F00", "#F0CBB2"),
        "M3ID": ("#0E765E", "#BFDFD3"),
        "ICD": ("#54369B", "#D2C5E6"),
    }
    option_colors = ["#BFD8EC", "#EDC2A4", "#B8DCCC", "#ECD28A", "#D2C1E3"]
    arrow_color = "#62686D"
    arrow_lw = 1.15
    shape_edge = "#202124"

    def section(y, h):
        # Whitespace and arrows separate stages; no enclosing UI-style card.
        return None

    # 1) Input and choices share a single visual block.
    section(13.05, 2.68)
    ax.text(.52, 15.42, "Input Image, Question, and Choices", fontsize=17,
            fontweight="bold", color=INK, va="center")
    add_cropped_image(
        ax, input_path, (.58, 3.12, 13.35, 15.10),
        face=True, border_color=shape_edge,
    )
    ax.text(3.62, 14.72, "What is this person’s age group?",
            fontsize=16.5, fontweight="bold", color=INK, va="center")
    choices = [("A", "Child"), ("B", "Teen"), ("C", "Young Adult"),
               ("D", "Middle-aged Adult"), ("E", "Older Adult")]
    for i, ((letter, label), fill) in enumerate(zip(choices, option_colors)):
        x = 4.00 + i*1.16
        ax.add_patch(Ellipse((x, 14.10), .43, .43,
                             facecolor=fill, edgecolor=shape_edge, lw=.8))
        ax.text(x, 14.10, letter, ha="center", va="center",
                fontsize=11.5, fontweight="bold")
    ax.text(6.30, 13.58,
            "Child · Teen · Young Adult · Middle-aged Adult · Older Adult",
            ha="center", va="center", fontsize=11.5, color="#454A4E")
    flow_arrow(ax, 5, 13.00, 5, 12.54, color=arrow_color, lw=arrow_lw)

    # 2) Only three example permutations; compact matrix is the visual.
    section(10.15, 2.38)
    ax.text(.52, 12.22, "I. Cyclic Answer-Order Permutations", fontsize=16,
            fontweight="bold", color=INK, va="center")
    ax.text(9.48, 12.22, "Examples (3 of 5 orders)", fontsize=12.5,
            color=MUTED, ha="right", va="center")
    orders = [["A", "B", "C", "D", "E"],
              ["C", "D", "E", "A", "B"],
              ["E", "A", "B", "C", "D"]]
    for row, order in enumerate(orders):
        y = 11.58-row*.58
        ax.text(1.02, y, str(row+1), fontsize=14.5, color=MUTED,
                ha="center", va="center")
        for col, letter in enumerate(order):
            idx = "ABCDE".index(letter)
            x = 2.22 + col*1.08
            ax.add_patch(Rectangle(
                (x, y-.19), .78, .38, facecolor=option_colors[idx],
                edgecolor=shape_edge, lw=.8,
            ))
            ax.text(x+.39, y, letter, ha="center", va="center",
                    fontsize=13.5, fontweight="bold")
    flow_arrow(ax, 5, 10.11, 5, 9.64, color=arrow_color, lw=arrow_lw)

    # 3) Shared model branches into four light method lanes.
    section(5.65, 3.98)
    ax.text(.52, 9.31, "II. Decoding with LLaVA-1.5-7B", fontsize=16,
            fontweight="bold", color=INK, va="center")
    ax.add_patch(FancyBboxPatch(
        (3.48, 8.45), 3.04, .62,
        boxstyle="round,pad=.015,rounding_size=.09",
        facecolor="#F1F3F4", edgecolor=shape_edge, linewidth=.9,
    ))
    ax.text(5, 8.76, "LLaVA-1.5-7B", ha="center", va="center",
            fontsize=15, fontweight="bold", color=INK)
    centers = [1.45, 3.82, 6.18, 8.55]
    names = ["Baseline", "VCD", "M3ID", "ICD"]
    junction_y = 8.14
    ax.plot([5, 5], [8.45, junction_y], color=arrow_color, lw=arrow_lw)
    for x, name in zip(centers, names):
        edge, fill = accent[name]
        ax.plot([5, x], [junction_y, 7.78], color=arrow_color, lw=arrow_lw)
        flow_arrow(ax, x, 7.78, x, 7.58, color=arrow_color, lw=arrow_lw)
        ax.add_patch(FancyBboxPatch(
            (x-.91, 5.74), 1.82, 1.69,
            boxstyle="round,pad=.015,rounding_size=.10",
            facecolor="white", edgecolor=shape_edge, linewidth=.85,
        ))
        ax.text(x, 7.21, name, ha="center", va="center",
                fontsize=15.5, fontweight="bold",
                color=INK if name == "Baseline" else edge)

        if name == "Baseline":
            for j, (letter, fill) in enumerate(zip("ABC", option_colors[:3])):
                cx = x-.43+j*.43
                ax.add_patch(Ellipse((cx, 6.73), .31, .31,
                                     facecolor=fill, edgecolor=shape_edge, lw=.75))
                ax.text(cx, 6.73, letter, ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=INK)
            label = "Standard"
        elif name == "VCD":
            add_cropped_image(
                ax, input_path, (x-.66, x-.12, 6.48, 6.91),
                face=True, border_color=shape_edge,
            )
            add_cropped_image(ax, input_path, (x+.12, x+.66, 6.48, 6.91),
                              face=True, distorted=True, border_color=shape_edge)
            ax.text(x, 6.69, "−", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=edge)
            label = "Image vs.\nnoised image"
        elif name == "M3ID":
            for cx, text, color in [
                    (x-.43, "p(y|I)", edge), (x+.43, "p(y)", "#7E8790")]:
                ax.add_patch(FancyBboxPatch(
                    (cx-.31, 6.51), .62, .37,
                    boxstyle="round,pad=.01,rounding_size=.08",
                    facecolor="white", edgecolor=shape_edge, linewidth=.8,
                ))
                ax.text(cx, 6.695, text, ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=INK)
            ax.text(x, 6.69, "vs.", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=INK)
            label = "Image-present\nvs. absent"
        else:
            ax.add_patch(FancyBboxPatch(
                (x-.78, 6.48), .68, .40,
                boxstyle="round,pad=.01,rounding_size=.06",
                facecolor="white", edgecolor=shape_edge, linewidth=.85,
            ))
            ax.add_patch(FancyBboxPatch(
                (x+.10, 6.48), .68, .40,
                boxstyle="round,pad=.01,rounding_size=.06",
                facecolor="white", edgecolor=shape_edge, linewidth=.85,
            ))
            ax.text(x-.44, 6.68, "Original", ha="center", va="center",
                    fontsize=6.8, fontweight="bold", color=INK)
            ax.text(x+.44, 6.68, "Contrastive", ha="center", va="center",
                    fontsize=6.3, fontweight="bold", color=INK)
            ax.text(x, 6.68, "−", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=edge)
            label = "Standard vs.\nconfused instruction"
        label_size = 8.8 if name == "ICD" else 10.2
        ax.text(x, 6.02, label, ha="center", va="center",
                fontsize=label_size, color=INK, fontweight="bold",
                linespacing=.94)

    # Explicit many-to-one aggregation connector.
    join_y = 5.14
    for x in centers:
        ax.plot([x, 5], [5.74, join_y], color=arrow_color, lw=arrow_lw)
    flow_arrow(ax, 5, join_y, 5, 4.74, color=arrow_color, lw=arrow_lw)

    # 4) One shared aggregation and output; no repeated placeholder results.
    section(2.23, 2.74)
    ax.text(.52, 4.43, "III. Probability Aggregation and Prediction", fontsize=16,
            fontweight="bold", color=INK, va="center")
    ax.add_patch(FancyBboxPatch(
        (.78, 3.38), 8.44, .65,
        boxstyle="round,pad=.015,rounding_size=.09",
        facecolor="#C5E2E0", edgecolor=shape_edge, linewidth=.9,
    ))
    ax.text(3.72, 3.705, "Per-Method Aggregation Across Answer Orders",
            ha="center", va="center", fontsize=14.2, fontweight="bold")
    for i, (letter, fill) in enumerate(zip("ABCDE", option_colors)):
        x = 7.08 + i*.35
        ax.add_patch(Ellipse((x, 3.705), .25, .25,
                             facecolor=fill, edgecolor=shape_edge, lw=.7))
        ax.text(x, 3.705, letter, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=INK)
    flow_arrow(ax, 5, 3.32, 5, 2.96, color=arrow_color, lw=arrow_lw)
    ax.add_patch(FancyBboxPatch(
        (1.72, 2.26), 6.56, .61,
        boxstyle="round,pad=.015,rounding_size=.09",
        facecolor="white", edgecolor=shape_edge, linewidth=.9,
    ))
    ax.text(5, 2.565, "Method-Specific Prediction + Confidence", ha="center",
            va="center", fontsize=16.5, fontweight="bold", color=INK)

    fig.subplots_adjust(left=.012, right=.988, top=.992, bottom=.012)
    fig.savefig(
        OUT / "fig1_evaluation_pipeline.png", facecolor="white",
        bbox_inches="tight", pad_inches=.03, dpi=300,
    )
    fig.savefig(
        OUT / "fig1_evaluation_pipeline.svg", transparent=True,
        bbox_inches="tight", pad_inches=.03,
    )
    plt.close(fig)
    print("created fig1_evaluation_pipeline.png and transparent SVG")
    print(f"pipeline input: {input_path}")


def plot_dataset_overview(selected):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    panels = [
        ("(a)  Facial Images", [("UTKFace", True), ("FairFace", True)]),
        ("(b)  Full-Body Images", [("PA-100K", False), ("MSP60K", False)]),
    ]
    for ax, (title, rows) in zip(axes, panels):
        ax.set_xlim(0, 4); ax.set_ylim(0, 2.65); ax.axis("off")
        ax.text(0, 2.48, title, fontsize=22, fontweight="bold")
        for row_idx, (dataset, face) in enumerate(rows):
            y0 = 1.28-row_idx*1.18
            ax.text(0, y0+.98, dataset, fontsize=16, fontweight="bold")
            for col, path in enumerate(selected[dataset]):
                extent = (col+.03, col+.91, y0, y0+.84)
                add_cropped_image(ax, path, extent, face=face)
        ax.plot([0, 3.94], [1.22, 1.22], color="#D7DADD", lw=.8)
    fig.subplots_adjust(left=.03, right=.99, top=.97, bottom=.04, wspace=.08)
    save(fig, "fig2_dataset_overview")


def plot_confidence(stats):
    increase_color = "#4E79A7"
    decrease_color = "#E07A3F"
    groups = [
        ("Age", ORDER[:2]), ("Gender", ORDER[2:4]), ("Race", ORDER[4:]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.05), sharex=True)
    positions = [4.80, 4.03, 3.26, 1.68, .91, .14]
    for panel, (ax, (attribute, datasets)) in enumerate(zip(axes, groups)):
        vals = []
        for _, _, key in datasets:
            for method in MITIGATION_METHODS:
                value = stats[key][method]["confidence"] - stats[key]["baseline"]["confidence"]
                vals.append(value)
        colors = [increase_color if v >= 0 else decrease_color for v in vals]
        ax.barh(
            positions, vals, height=.44, color=colors,
            edgecolor="white", linewidth=.35, zorder=3,
        )
        for yy, value, method in zip(positions, vals, ["VCD", "M3ID", "ICD"] * 2):
            ax.text(-.006, yy, method, ha="right", va="center",
                    fontsize=15, fontweight="bold", color=INK)
            label_x = value + .0035 if value >= 0 else .004
            sign = "+" if value >= 0 else "\N{MINUS SIGN}"
            ax.text(label_x, yy, f"{sign}{abs(value):.3f}",
                    ha="left",
                    va="center", fontsize=15.7,
                    fontweight="bold", color=INK)
        ax.text(.004, 5.33, datasets[0][1], ha="left", fontsize=16,
                fontweight="bold", color=INK)
        ax.text(.004, 2.21, datasets[1][1], ha="left", fontsize=16,
                fontweight="bold", color=INK)
        ax.hlines(
            2.69, 0, .158, colors="#E5E7E9",
            linewidth=.75, zorder=1,
        )
        ax.set_title(
            f"({chr(97 + panel)})  {attribute}",
            fontsize=21, fontweight="bold", color=INK, pad=18,
        )
        ax.axvline(0, color="#333333", lw=1.25, zorder=2)
        ax.grid(axis="x", color="#E5E7E9", lw=.75, zorder=0)
        ax.set_xlim(-.01, .16)
        ax.set_ylim(-.35, 5.72)
        ax.set_yticks([])
        ax.set_xticks([-.01, .05, .10, .15])
        ax.tick_params(axis="x", labelsize=13.5, length=0, pad=7, colors=INK)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#7B8084")
        ax.spines["bottom"].set_linewidth(.8)
    fig.subplots_adjust(left=.035, right=.99, top=.88, bottom=.15, wspace=.12)
    fig.savefig(
        OUT / "fig3_confidence_change.png", facecolor="white",
        bbox_inches="tight", pad_inches=.04, dpi=300,
    )
    fig.savefig(
        OUT / "fig3_confidence_change.svg", facecolor="white",
        bbox_inches="tight", pad_inches=.04,
    )
    plt.close(fig)
    print("created fig3_confidence_change.png and SVG")


def plot_reliability(raw, stats):
    # Determine the strongest actual ECE worsening across all mitigation pairs.
    worsening = []
    for _, _, key in ORDER:
        for method in MITIGATION_METHODS:
            delta = stats[key][method]["ece"] - stats[key]["baseline"]["ece"]
            worsening.append((delta, key, method))
    worst_delta, worst_key, worst_method = max(worsening)
    requested = ("PA-100K-Age", "icd")
    if (worst_key, worst_method) != requested:
        print(
            "reliability worsening panel changed: "
            f"requested={requested[0]}/{requested[1]}, "
            f"strongest={worst_key}/{worst_method}, ΔECE={worst_delta:+.4f}"
        )
    panels = [
        ("(a)  MSP60K-Gender", "MSP60K-Gender", "vcd"),
        ("(b)  FairFace-Race", worst_key, worst_method),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.35), sharex=True, sharey=False)
    for ax, (panel_title, key, method) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], ls="--", color="#A8ADB1", lw=1.45)
        for m, color, marker in [
            ("baseline", BASELINE, "o"),
            (method, NEGATIVE, "s"),
        ]:
            confs, accs, counts = compute_reliability_bins(raw[key][m])
            x = [c for c, n in zip(confs, counts) if n]
            y = [a for a, n in zip(accs, counts) if n]
            ax.plot(x, y, color=color, marker=marker, ms=7.2, lw=2.4,
                    label=METHOD_LABELS[m])
        ax.set_title(panel_title.replace("-", "–"), fontsize=21,
                     fontweight="bold", y=1.175, pad=0)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.08)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=.8)
        ax.set_xticks(np.arange(0, 1.01, .2))
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.tick_params(labelsize=15, labelleft=True)
        ax.legend(
            frameon=False, fontsize=13.5, loc="lower center",
            bbox_to_anchor=(.5, 1.025), ncol=2,
            handlelength=2.0, handletextpad=.70, columnspacing=1.75,
        )
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes:
        ax.set_xlabel("Mean Confidence", fontsize=18)
        ax.set_ylabel("Empirical Accuracy", fontsize=18)
    fig.subplots_adjust(left=.07, right=.99, top=.75, bottom=.12, wspace=.10)
    fig.savefig(OUT / "fig4_reliability_diagrams.png", facecolor="white",
                bbox_inches="tight", pad_inches=.04, dpi=300)
    fig.savefig(OUT / "fig4_reliability_diagrams.svg", transparent=True,
                bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    print("created fig4_reliability_diagrams.png and transparent SVG")


def print_summary(stats, significance):
    print("SUMMARY")
    for _, _, key in ORDER:
        for method in ["baseline", *MITIGATION_METHODS]:
            s = stats[key][method]
            sig = significance[key].get(method, {})
            print(
                f"  {key:17s} {method:8s} n={s['n']:4d} "
                f"acc={s['accuracy']:.4f} conf={s['confidence']:.4f} "
                f"ece={s['ece']:.4f} acc_sig={sig.get('accuracy', '-')}"
            )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, stats, significance = load_all()
    print_summary(stats, significance)
    selected, missing = select_supplied_examples()
    if missing:
        print("MISSING REQUIRED DATASET EXAMPLES")
        for dataset, paths in missing.items():
            print(f"  {dataset}: {', '.join(paths)}")
        raise FileNotFoundError("Need at least four supplied examples per dataset.")
    copy_supplied_samples(selected)
    plot_pipeline(selected)
    plot_confidence(stats)
    plot_reliability(raw, stats)


if __name__ == "__main__":
    main()
