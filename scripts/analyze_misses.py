#!/usr/bin/env python3
"""Extrai o que o detector deixa de encontrar, para orientar o gerador.

Roda um checkpoint sobre um conjunto real, casa as predições com o gabarito
por IoU e separa as caixas que ficaram sem par. Emite recortes das falhas e
o perfil delas — tamanho, densidade local e posição —, que é o que permite
dizer ao gerador o que produzir mais.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fruit_pipeline.boxes import iou_matrix, read_boxes
from fruit_pipeline.common import ROOT

FONT = ROOT / "docs/figures/fonts/Geist[wght].ttf"


def neighbor_counts(boxes: np.ndarray, radius: float) -> np.ndarray:
    """Quantas outras caixas de gabarito há em volta de cada uma."""
    if len(boxes) == 0:
        return np.zeros(0)
    centers = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
    )
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    return (distances < radius).sum(axis=1) - 1


def contact_sheet(crops, path: Path, columns=8, cell=150):
    if not crops:
        return
    font = ImageFont.truetype(str(FONT), 15)
    rows = (len(crops) + columns - 1) // columns
    caption_height = 22
    sheet = Image.new(
        "RGB",
        (columns * (cell + 8) + 8, rows * (cell + caption_height + 8) + 8),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for i, (caption, crop) in enumerate(crops):
        column, row = i % columns, i // columns
        x, y = 8 + column * (cell + 8), 8 + row * (cell + caption_height + 8)
        draw.text((x, y), caption, font=font, fill="#233A33")
        crop = crop.resize((cell, cell), Image.LANCZOS)
        sheet.paste(crop, (x, y + caption_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--amostras", type=int, default=24)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    missed, found, crops = [], [], []
    for image_path in sorted(Path(args.images).glob("*.jpg")):
        label_path = Path(args.labels) / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            gt = read_boxes(label_path, W, H)
            if len(gt) == 0:
                continue
            result = model.predict(
                source=str(image_path), conf=args.conf, imgsz=960, max_det=1000,
                device="0", verbose=False,
            )[0]
            pred = result.boxes.xyxy.cpu().numpy()
            best_iou = iou_matrix(gt, pred).max(axis=1) if len(pred) else np.zeros(len(gt))
            density = neighbor_counts(gt, radius=0.08 * max(W, H))
            side = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
            for i in range(len(gt)):
                record = {
                    "lado": float(side[i] / W),
                    "densidade": int(density[i]),
                    "altura": float((gt[i, 1] + gt[i, 3]) / 2 / H),
                    "iou": float(best_iou[i]),
                }
                (missed if best_iou[i] < args.iou else found).append(record)
                if best_iou[i] < args.iou:
                    cx, cy = (gt[i, 0] + gt[i, 2]) / 2, (gt[i, 1] + gt[i, 3]) / 2
                    half = max(48, float(side[i]) * 2.4)
                    crop = im.crop(
                        (
                            max(0, int(cx - half)), max(0, int(cy - half)),
                            min(W, int(cx + half)), min(H, int(cy + half)),
                        )
                    ).copy()
                    scale = 150 / max(crop.size)
                    ImageDraw.Draw(crop).rectangle(
                        [gt[i, 0] - max(0, cx - half), gt[i, 1] - max(0, cy - half),
                         gt[i, 2] - max(0, cx - half), gt[i, 3] - max(0, cy - half)],
                        outline="#ff4d4d", width=max(1, int(2 / scale)),
                    )
                    crops.append((float(side[i] / W), f"{side[i]:.0f}px", crop))
    total = len(missed) + len(found)
    print(
        f"\ngabarito: {total} caixas | encontradas {len(found)} | "
        f"perdidas {len(missed)} ({100 * len(missed) / total:.1f}%)"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"total": total, "perdidas": len(missed), "detalhe": missed},
        (output / "misses.json").open("w"),
    )

    def miss_rate(title, key, ranges):
        print(f"\ntaxa de perda por {title}:")
        for lo, hi in ranges:
            missed_here = [r for r in missed if lo <= r[key] < hi]
            found_here = [r for r in found if lo <= r[key] < hi]
            n = len(missed_here) + len(found_here)
            if n < 20:
                continue
            print(f"  {lo:6.3f}–{hi:6.3f}: {100 * len(missed_here) / n:5.1f}% perdidas  (n={n})")

    miss_rate("tamanho (lado/largura)", "lado", [(0, .015), (.015, .025), (.025, .04), (.04, .07), (.07, 1)])
    miss_rate("densidade local (vizinhos)", "densidade", [(0, 3), (3, 8), (8, 15), (15, 100)])
    miss_rate("altura na imagem", "altura", [(0, .25), (.25, .5), (.5, .75), (.75, 1)])

    crops.sort(key=lambda c: c[0])
    n = args.amostras
    middle = len(crops) // 2
    for name, selection in (
        ("perdidas-menores", crops[:n]),
        ("perdidas-medianas", crops[middle - n // 2 : middle + n // 2]),
        ("perdidas-maiores", crops[-n:]),
    ):
        contact_sheet([(c[1], c[2]) for c in selection], output / f"{name}.jpg")


if __name__ == "__main__":
    main()
