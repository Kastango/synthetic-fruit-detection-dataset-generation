#!/usr/bin/env python3
"""Exporta cenas existentes, selecionadas por quantis de contagem de caixas."""

import argparse
import json

from PIL import Image

from fruit_pipeline.common import ROOT, atomic_write_json, sha256_file
from render_detection_examples import draw_boxes, read_boxes, resize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/generated/paired_reference")
    args = parser.parse_args()
    dataset = ROOT / args.dataset
    manifest = dataset / "manifest.jsonl"
    rows = sorted(
        [json.loads(line) for line in manifest.read_text().splitlines() if line],
        key=lambda row: (row["annotations"], row["generation_index"]),
    )
    output = ROOT / "docs/figures/results/synthetic-examples"
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, quantile in enumerate((0.25, 0.5, 0.75, 0.97), 1):
        row = rows[round(quantile * (len(rows) - 1))]
        source = dataset / row["image"]
        label = dataset / row["label"]
        with Image.open(source) as image:
            shown = resize(image.convert("RGB"))
        shown.save(output / f"scene-{index}.jpg", quality=95, subsampling=0)
        boxes = read_boxes(label)
        assert len(boxes) == row["annotations"]
        draw_boxes(shown, boxes, width=5).save(
            output / f"scene-{index}-boxes.jpg", quality=95, subsampling=0
        )
        examples.append(
            dict(
                row,
                quantile=quantile,
                image_sha256=sha256_file(source),
                label_sha256=sha256_file(label),
            )
        )
    atomic_write_json(
        output / "provenance.json",
        {
            "dataset": args.dataset,
            "manifest_sha256": sha256_file(manifest),
            "selection": "round(q * (n - 1)), sorted by (annotations, generation_index), all splits",
            "examples": examples,
        },
    )
    print([(row["generation_index"], row["annotations"]) for row in examples])


if __name__ == "__main__":
    main()
