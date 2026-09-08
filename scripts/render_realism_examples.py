#!/usr/bin/env python3
"""Compara as mesmas cenas de duas receitas e registra a origem das figuras."""

import argparse
import json

from PIL import Image, ImageDraw

from fruit_pipeline.common import ROOT, atomic_write_json, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default="paired_reference")
    p.add_argument("--candidate", required=True)
    p.add_argument("--scene", type=int, action="append", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = ROOT / "docs/figures/results/realism"
    output.mkdir(parents=True, exist_ok=True)
    panel = Image.new("RGB", (960, 640 * len(args.scene)), "white")
    provenance = []
    for column, name in enumerate((args.reference, args.candidate)):
        root = ROOT / "data/generated" / name
        manifest = root / "manifest.jsonl"
        rows = {
            r["generation_index"]: r
            for r in map(json.loads, manifest.read_text().splitlines())
        }
        for row_index, index in enumerate(args.scene):
            row = rows[index]
            path = root / row["image"]
            with Image.open(path) as image:
                shown = image.convert("RGB")
                shown.thumbnail((480, 610), Image.Resampling.LANCZOS)
            panel.paste(shown, (480 * column, 640 * row_index + 30))
            ImageDraw.Draw(panel).text(
                (480 * column + 8, 640 * row_index + 8),
                f"{name} | cena {index} | {row['annotations']} caixas",
                fill="black",
            )
            provenance.append(
                {
                    "dataset": name,
                    "generation_index": index,
                    "seed": row["seed"],
                    "requested_objects": row["requested_objects"],
                    "annotations": row["annotations"],
                    "config_hash": row["config_hash"],
                    "generator_sha256": row["generator_sha256"],
                    "manifest_sha256": sha256_file(manifest),
                    "image": str(path.relative_to(ROOT)),
                    "image_sha256": sha256_file(path),
                    "label_sha256": sha256_file(root / row["label"]),
                }
            )
    panel.save(output / f"{args.output}.jpg", quality=95, subsampling=0)
    atomic_write_json(
        output / f"{args.output}.json",
        {
            "selection": "Índices fixados antes da avaliação dos detectores; ver protocolo.",
            "examples": provenance,
        },
    )


if __name__ == "__main__":
    main()
