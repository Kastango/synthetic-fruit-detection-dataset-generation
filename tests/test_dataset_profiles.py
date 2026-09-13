import importlib.util
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "profiles", Path(__file__).parents[1] / "scripts/measure_dataset_profiles.py"
)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


def test_scan_distinguishes_empty_missing_and_invalid_labels(tmp_path):
    images = tmp_path / "images/train"
    labels = tmp_path / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i, name in enumerate(["empty", "positive", "missing", "invalid"]):
        Image.new("RGB", (100, 200), (i * 40, 10, 20)).save(images / f"{name}.png")
    (labels / "empty.txt").write_text("")
    (labels / "positive.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    (labels / "invalid.txt").write_text("0 0.99 0.5 0.2 0.1\n")
    summary, _rows, boxes = profiles.scan(tmp_path, ("train",), "fixture")
    assert summary["empty_images"] == 1
    assert summary["boxes"] == 1
    assert len(summary["errors"]) == 2
    assert boxes[0]["width_px"] == boxes[0]["height_px"] == 20
    assert boxes[0]["max_side_norm"] == 0.2
    assert boxes[0]["sqrt_area_960"] == 96


def test_weighted_cdf_keeps_domain_weight_despite_sample_imbalance():
    result = profiles.weighted_description([1, 9, 9, 9], [0.5, 1 / 6, 1 / 6, 1 / 6])
    assert result["mean"] == pytest.approx(5)
    assert result["p25"] == 1
    assert result["p75"] == 9
