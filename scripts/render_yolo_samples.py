#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from fruit_pipeline.common import image_files, project_path


def annotated_thumbnail(
    image_path: Path, label_path: Path, size: tuple[int, int]
) -> Image.Image:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(4, round(min(image.size) / 500))
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, x, y, width, height = line.split()
        x, y, width, height = map(float, (x, y, width, height))
        left = (x - width / 2) * image.width
        right = (x + width / 2) * image.width
        top = (y - height / 2) * image.height
        bottom = (y + height / 2) * image.height
        draw.rectangle(
            (left, top, right, bottom), outline=(255, 55, 40), width=line_width
        )
    image.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, "white")
    tile.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    ImageDraw.Draw(tile).text(
        (8, 8), image_path.name, fill=(0, 0, 0), stroke_width=2, stroke_fill="white"
    )
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renderiza amostras YOLO para inspeção visual."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/real_yolo"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/real_train_samples.jpg")
    )
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_test:
        raise SystemExit(
            "visualização do teste bloqueada; use somente após a seleção final"
        )
    dataset = project_path(args.dataset)
    paths = image_files(dataset / "images" / args.split)
    if not paths:
        raise FileNotFoundError(f"nenhuma imagem em {dataset / 'images' / args.split}")
    chosen = random.Random(args.seed).sample(paths, min(args.count, len(paths)))
    tile_size = (360, 480)
    columns = min(4, len(chosen))
    rows = math.ceil(len(chosen) / columns)
    sheet = Image.new(
        "RGB", (columns * tile_size[0], rows * tile_size[1]), (235, 235, 235)
    )
    for index, image_path in enumerate(chosen):
        label = dataset / "labels" / args.split / f"{image_path.stem}.txt"
        tile = annotated_thumbnail(image_path, label, tile_size)
        sheet.paste(
            tile, ((index % columns) * tile_size[0], (index // columns) * tile_size[1])
        )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(output)


if __name__ == "__main__":
    main()
