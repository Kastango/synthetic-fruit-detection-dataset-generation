#!/usr/bin/env python3
"""Roda o checkpoint yolo26s selecionado de cada condição sobre uma mesma
imagem do teste CitDet e salva as detecções lado a lado com o gabarito, para
comparação visual qualitativa (complementa as tabelas de mAP em RESULTS.md).

Não participa do pipeline reprodutível (`run_pipeline.sh`); script de análise
executado manualmente sobre checkpoints já treinados.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "artifacts" / "confirmatory" / "model_selection.json"
IMAGE_STEM = "ftp-6-60-43_fruit-drop-back-picture_1_2021-11-09-01-57-07_jpg"
IMAGE_PATH = ROOT / "data" / "external_tests" / "citdet" / "images" / "test" / f"{IMAGE_STEM}.jpg"
LABEL_PATH = ROOT / "data" / "external_tests" / "citdet" / "labels" / "test" / f"{IMAGE_STEM}.txt"
OUT_DIR = ROOT / "docs" / "figures" / "results" / "examples"

MODEL = "yolo26s"
SEED = 41
CONDITION_ORDER = [
    "manual-full",
    "controlled",
    "synthetic-1x",
    "synthetic-2x",
    "synthetic-3x",
    "synthetic-5x",
    "synthetic-10x",
]
MAX_SIDE = 960


def checkpoint_for(condition: str) -> Path:
    entry = json.loads(SELECTION.read_text(encoding="utf-8"))
    runs = entry["selected"][MODEL][condition]["runs"]
    for run in runs:
        if run["seed"] == SEED:
            return Path(run["checkpoint"])
    return Path(runs[0]["checkpoint"])


def resize(image: Image.Image) -> Image.Image:
    ratio = MAX_SIDE / max(image.size)
    if ratio >= 1:
        return image
    return image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)


def render_ground_truth() -> Path:
    image = Image.open(IMAGE_PATH).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for line in LABEL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = (float(v) for v in line.split())
        left = (cx - w / 2) * width
        top = (cy - h / 2) * height
        right = (cx + w / 2) * width
        bottom = (cy + h / 2) * height
        draw.rectangle([left, top, right, bottom], outline=(27, 175, 122), width=3)
    out_path = OUT_DIR / "ground-truth.jpg"
    resize(image).save(out_path, quality=90)
    return out_path


def render_predictions() -> None:
    from ultralytics import YOLO

    for condition in CONDITION_ORDER:
        checkpoint = checkpoint_for(condition)
        detector = YOLO(str(checkpoint))
        result = detector.predict(source=str(IMAGE_PATH), conf=0.25, imgsz=960, device=0, verbose=False)[0]
        annotated = Image.fromarray(result.plot(line_width=2)[:, :, ::-1])
        out_path = OUT_DIR / f"{condition}.jpg"
        resize(annotated).save(out_path, quality=90)
        print(f"{condition}: {checkpoint.parent.parent.name} -> {out_path.relative_to(ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_path = render_ground_truth()
    print(f"gabarito -> {gt_path.relative_to(ROOT)}")
    render_predictions()


if __name__ == "__main__":
    main()
