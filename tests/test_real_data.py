from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from fruit_pipeline.preprocess import FRUIT_BBOX_MANIFEST
from fruit_pipeline.real_data import (
    import_real_dataset,
    materialize_controlled_dataset,
    proportional_allocation,
    split_real_dataset,
    validate_yolo_text,
)


def test_device_allocation_matches_the_frozen_protocol() -> None:
    assert proportional_allocation({"iphone": 82, "pixel": 48}, 26) == {
        "iphone": 16,
        "pixel": 10,
    }


def test_yolo_validation_rejects_boxes_outside_image() -> None:
    try:
        validate_yolo_text("0 0.99 0.5 0.2 0.1\n", "sample.txt")
    except ValueError as error:
        assert "horizontal" in str(error)
    else:
        raise AssertionError("invalid label accepted")


def test_small_import_and_split_keep_unaugmented_records(
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

    split = split_real_dataset(
        {
            "project": {"seed": 42, "class_names": ["poncan"]},
            "paths": {
                "real_source": str(target),
                "artifacts": str(tmp_path / "artifacts"),
            },
            "real_dataset": {
                "train_images": 1,
                "val_images": 1,
                "output": str(tmp_path / "real_yolo_confirmatory"),
                "artifact": "real_split.json",
            },
        }
    )
    assert split["counts"] == {"train": 1, "val": 1}
    assert not (tmp_path / "real_yolo_confirmatory" / "images" / "test").exists()


def test_materialize_controlled_dataset_derives_boxes_and_negatives(
    tmp_path: Path,
) -> None:
    raw_fruits = tmp_path / "raw" / "fruits"
    regenerated = tmp_path / "assets"
    raw_fruits.mkdir(parents=True)
    (regenerated / "backgrounds").mkdir(parents=True)

    bboxes = {}
    for index in range(4):
        name = f"fruit{index}"
        Image.new("RGB", (100, 80), (10, 20, 30)).save(raw_fruits / f"{name}.jpg")
        bboxes[name] = {"bbox": [10, 5, 60, 55], "image_size": [100, 80]}
    (regenerated / FRUIT_BBOX_MANIFEST).write_text(json.dumps(bboxes), encoding="utf-8")
    for index in range(6):
        Image.new("RGB", (64, 48), (5, 5, 5)).save(
            regenerated / "backgrounds" / f"bg{index}.jpg"
        )

    config = {
        "paths": {
            "raw": str(tmp_path / "raw"),
            "regenerated_assets": str(regenerated),
            "real_controlled": str(tmp_path / "real_controlled"),
            "artifacts": str(tmp_path / "artifacts"),
        },
        "asset_split": {"train_ratio": 0.8, "seed": 42},
        "project": {"class_names": ["poncan"]},
    }
    summary = materialize_controlled_dataset(config)
    assert summary["fruits"]["train"] + summary["fruits"]["val"] == 4
    assert summary["backgrounds"]["train"] + summary["backgrounds"]["val"] == 6

    output = tmp_path / "real_controlled"
    fruit_labels = list((output / "labels" / "train").glob("fruit-*.txt")) + list(
        (output / "labels" / "val").glob("fruit-*.txt")
    )
    assert len(fruit_labels) == 4
    for label_path in fruit_labels:
        parts = label_path.read_text(encoding="utf-8").split()
        assert parts[0] == "0"
        cx, cy, width, height = (float(value) for value in parts[1:])
        assert abs(cx - 0.35) < 1e-6
        assert abs(cy - 0.375) < 1e-6
        assert abs(width - 0.5) < 1e-6
        assert abs(height - 0.625) < 1e-6

    background_labels = list((output / "labels" / "train").glob("bg-*.txt")) + list(
        (output / "labels" / "val").glob("bg-*.txt")
    )
    assert len(background_labels) == 6
    assert all(path.read_text(encoding="utf-8") == "" for path in background_labels)

    # Reexecutar sem --force reutiliza o resultado congelado (fingerprint igual).
    cached = materialize_controlled_dataset(config)
    assert cached == summary
