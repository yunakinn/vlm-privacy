"""Plot the published aggregate metrics without loading a model or raw images."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline", "vcd", "m3id", "icd")
COLORS = ("#555555", "#376795", "#408477", "#A46B43")
MARKERS = ("o", "s", "^", "D")
METRICS = (
    ("accuracy", "Accuracy ↑", True),
    ("avg_choice_confidence", "Mean choice confidence", True),
    ("ece_10_bins", "ECE ↓ · 10 bins", False),
    ("multiclass_brier_score", "Multiclass Brier score ↓", False),
)


def load_metrics(path: Path) -> tuple[list[str], dict]:
    tasks, records = [], {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"experiment", "dataset", "attribute", "method"} | {
            key for key, _, _ in METRICS
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            task, method = row["experiment"], row["method"]
            if not task or method not in METHODS:
                raise ValueError(f"Line {line}: invalid experiment or method")
            if (task, method) in records:
                raise ValueError(f"Duplicate row: {task}/{method}")
            if task not in tasks:
                tasks.append(task)
            values = {key: float(row[key]) for key, _, _ in METRICS}
            for key, value in values.items():
                upper = 2 if key == "multiclass_brier_score" else 1
                if not math.isfinite(value) or not 0 <= value <= upper:
                    raise ValueError(f"Line {line}: invalid {key}: {value}")
            records[task, method] = {
                "label": f"{row['dataset']} · {row['attribute']}", **values
            }
    if not tasks:
        raise ValueError("Metrics CSV contains no rows")
    for task in tasks:
        for method in METHODS:
            if (task, method) not in records:
                raise ValueError(f"Missing row: {task}/{method}")
            if records[task, method]["label"] != records[task, "baseline"]["label"]:
                raise ValueError(f"Inconsistent dataset/attribute for {task}")
    return tasks, records


def plot_metrics(input_path: Path, output_dir: Path) -> list[Path]:
    tasks, records = load_metrics(input_path)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "svg.fonttype": "none", "svg.hashsalt": "vlm-privacy",
    }):
        fig, axes = plt.subplots(2, 2, figsize=(13, max(8, len(tasks) * .8 + 3)))
        for ax, (metric, title, percent) in zip(axes.flat, METRICS):
            for index, (method, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS)):
                positions = [i + (index - 1.5) * .15 for i in range(len(tasks))]
                values = [records[task, method][metric] for task in tasks]
                ax.scatter(values, positions, label=method.upper() if index else "Baseline",
                           color=color, marker=marker, s=34, zorder=3)
            ax.set_yticks(range(len(tasks)), [records[t, "baseline"]["label"] for t in tasks])
            ax.set_ylim(len(tasks) - .4, -.6)
            upper = 1 if percent else max(records[t, m][metric] for t in tasks for m in METHODS) * 1.15
            ax.set_xlim(0, max(upper, .01))
            if percent:
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_title(title, loc="left", fontweight="bold", pad=14)
            ax.grid(axis="x", color="#E5E5E5", linewidth=.7)
            ax.tick_params(axis="y", length=0)
            ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94),
                   ncol=4, frameon=False)
        fig.suptitle("VLM evaluation · Published aggregate results", x=.03,
                     ha="left", fontsize=16, fontweight="bold", y=.99)
        fig.text(.03, .025,
                 "Confidence is not a quality score. Arrows indicate preferred direction.\n"
                 "Original experiments; not rerun after code fixes. No uncertainty intervals are available in the aggregate CSV.",
                 fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .09, 1, .88), h_pad=3, w_pad=3)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in ("svg", "png"):
            path = output_dir / f"metrics_overview.{extension}"
            fig.savefig(path, dpi=180, facecolor="white", metadata={"Date": None} if extension == "svg" else None)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            paths.append(path)
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/final_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/figures/generated")
    args = parser.parse_args()
    for path in plot_metrics(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
