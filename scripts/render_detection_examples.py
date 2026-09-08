#!/usr/bin/env python3
"""Renderiza exemplos reproduzíveis com checkpoints históricos, sem retreinar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from fruit_pipeline.common import ROOT, sha256_file, atomic_write_json

CONDITION_ORDER = [
    "manual-full",
    "controlled",
    "synthetic-1x",
    "synthetic-2x",
    "synthetic-3x",
    "synthetic-5x",
    "synthetic-10x",
]
CITDET_STEM = "ftp-6-60-43_fruit-drop-back-picture_1_2021-11-09-01-57-07_jpg"


def resize(image, max_side=1440):
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def draw_boxes(image, boxes, color="#ffe600", width=4):
    """Traço em pixels da saída, com contorno preto para qualquer fundo."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for x, y, bw, bh in boxes:
        rect = [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
        draw.rectangle(
            [rect[0] - 2, rect[1] - 2, rect[2] + 2, rect[3] + 2],
            outline="black",
            width=width + 4,
        )
        draw.rectangle(rect, outline=color, width=width)
    return image


def read_boxes(path):
    return [
        [float(v) for v in line.split()[1:]]
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["citdet", "manual_full_val"], default="citdet")
    p.add_argument(
        "--model", choices=["yolov8s", "yolo26s", "rtdetr-l"], default="yolo26s"
    )
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--image-stem")
    p.add_argument("--condition", action="append", choices=CONDITION_ORDER)
    p.add_argument("--device", default="0")
    args = p.parse_args()
    from ultralytics import YOLO

    if args.dataset == "citdet":
        root = ROOT / "data/external_tests/citdet"
        split = "test"
        stem = args.image_stem or CITDET_STEM
        output = ROOT / "docs/figures/results/examples"
    else:
        root = ROOT / "data/real_yolo_confirmatory"
        split = "val"
        # Primeira imagem em ordem lexicográfica: seleção independente das predições.
        stem = (
            args.image_stem or sorted((root / "images" / split).glob("*.jpg"))[0].stem
        )
        output = ROOT / "docs/figures/results/examples/manual-full-val" / args.model
    image_path = root / "images" / split / f"{stem}.jpg"
    label_path = root / "labels" / split / f"{stem}.txt"
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    shown = resize(original)
    gt = read_boxes(label_path)
    draw_boxes(shown, gt, width=5).save(
        output / "ground-truth.jpg", quality=95, subsampling=0
    )
    selection_path = ROOT / "artifacts/confirmatory/model_selection.json"
    selection = json.loads(selection_path.read_text())
    provenance = dict(
        dataset=args.dataset,
        image=image_path.relative_to(ROOT).as_posix(),
        image_sha256=sha256_file(image_path),
        label_sha256=sha256_file(label_path),
        ground_truth_boxes=len(gt),
        model=args.model,
        seed=args.seed,
        confidence=0.25,
        imgsz=960,
        max_det=1000,
        selection_sha256=sha256_file(selection_path),
        runs=[],
    )
    for condition in args.condition or CONDITION_ORDER:
        runs = selection["selected"][args.model][condition]["runs"]
        run = next(r for r in runs if r["seed"] == args.seed)
        checkpoint = Path(run["checkpoint"])
        if sha256_file(checkpoint) != run["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint alterado: {checkpoint}")
        result = YOLO(str(checkpoint)).predict(
            source=str(image_path),
            conf=0.25,
            imgsz=960,
            max_det=1000,
            device=args.device,
            verbose=False,
        )[0]
        boxes = result.boxes.xywhn.cpu().tolist()
        draw_boxes(shown, boxes, color="#22dfff", width=3).save(
            output / f"{condition}.jpg", quality=95, subsampling=0
        )
        provenance["runs"].append(
            dict(
                condition=condition,
                run_id=run["run_id"],
                checkpoint_sha256=run["checkpoint_sha256"],
                detections=len(boxes),
            )
        )
        print(args.dataset, args.model, condition, len(boxes))
    atomic_write_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
