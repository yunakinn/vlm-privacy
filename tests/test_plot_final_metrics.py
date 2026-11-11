import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plot_final_metrics", ROOT / "scripts/plot_final_metrics.py")
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)


def test_published_metrics_render_without_model_or_images(tmp_path):
    paths = plotter.plot_metrics(ROOT / "results/final_metrics.csv", tmp_path)
    assert {p.suffix for p in paths} == {".svg", ".png"}
    svg = (tmp_path / "metrics_overview.svg").read_text()
    for text in ("UTKFace", "FairFace", "PA-100K", "MSP60K", "Brier", "Confidence is not"):
        assert text in svg
    assert (tmp_path / "metrics_overview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nan"])
def test_incomplete_or_invalid_comparisons_fail_before_rendering(tmp_path, corruption):
    with (ROOT / "results/final_metrics.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["accuracy"] = "nan"
    source = tmp_path / "metrics.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        plotter.plot_metrics(source, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()
