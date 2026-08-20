from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image

from fruit_pipeline.external_test import import_external_test


def test_nested_coco_archive_is_imported_without_reencoding(tmp_path: Path) -> None:
    inner_root = tmp_path / "inner"
    images = inner_root / "test" / "images"
    images.mkdir(parents=True)
    image = images / "sample.jpg"
    Image.new("RGB", (40, 20), (20, 90, 30)).save(image, quality=91)
    original_bytes = image.read_bytes()
    annotations = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 40, "height": 20}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [5, 4, 10, 8]}
        ],
        "categories": [{"id": 2, "name": "fruit"}],
    }
    (inner_root / "test" / "annotations.json").write_text(json.dumps(annotations))
    inner_zip = tmp_path / "CitDet-test.zip"
    with zipfile.ZipFile(inner_zip, "w") as archive:
        for path in sorted((inner_root / "test").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(inner_root))
    outer_zip = tmp_path / "UTA_CSE_Dataset.zip"
    with zipfile.ZipFile(outer_zip, "w") as archive:
        archive.write(inner_zip, inner_zip.name)

    summary = import_external_test(
        "citdet",
        outer_zip,
        tmp_path / "external",
        ["poncan"],
        annotation_format="coco",
        nested_archive="CitDet-test.zip",
        expected_images=1,
        expected_boxes=1,
    )

    target = tmp_path / "external" / "citdet"
    assert summary["images"] == 1
    assert summary["boxes"] == 1
    assert (target / "images" / "test" / "sample.jpg").read_bytes() == original_bytes
    assert (target / "labels" / "test" / "sample.txt").read_text().startswith("0 ")
