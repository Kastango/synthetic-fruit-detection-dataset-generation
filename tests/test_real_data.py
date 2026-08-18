from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from fruit_pipeline.real_data import (
    import_real_dataset,
    proportional_allocation,
    validate_yolo_text,
)


def test_device_allocation_matches_the_frozen_protocol() -> None:
    assert proportional_allocation({"iphone": 82, "pixel": 48}, 15) == {
        "iphone": 9,
        "pixel": 6,
    }


def test_yolo_validation_rejects_boxes_outside_image() -> None:
    try:
        validate_yolo_text("0 0.99 0.5 0.2 0.1\n", "sample.txt")
    except ValueError as error:
        assert "horizontal" in str(error)
    else:
        raise AssertionError("invalid label accepted")


def test_small_import_uses_metadata_and_keeps_unaugmented_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    for index in range(2):
        Image.new("RGB", (64, 96), (20 + index, 50, 80)).save(
            source / "images" / f"sample-{index}.jpg"
        )
        (source / "labels" / f"sample-{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "device"])
        writer.writerow(["sample-0.jpg", "iphone"])
        writer.writerow(["sample-1.jpg", "pixel"])
    target = tmp_path / "imported"
    summary = import_real_dataset(
        source,
        target,
        metadata_path=metadata,
        annotation_format="yolo",
        expected_images=2,
        expected_boxes=2,
        allow_unknown_device=False,
        allow_processed_input=True,
        force=False,
    )
    assert summary["images"] == 2
    assert summary["boxes"] == 2
    assert summary["devices"] == {"iphone": 1, "pixel": 1}
    assert (target / "manifest.json").exists()
