"""Descritores transparentes de distribuição; não são uma medida de realismo."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .common import image_files

FEATURES = {
    "count": ("Caixas por imagem", 1),
    "size": ("Tamanho √(área) / imagem", 100),
    "center_y": ("Posição vertical", 100),
    "brightness": ("Luminância na caixa", 100),
    "saturation": ("Saturação na caixa", 100),
    "contrast": ("Contraste caixa / entorno", 1),
    "clipped": ("Pixels claros saturados", 100),
}


def read_boxes(path: Path) -> list[list[float]]:
    boxes = []
    for line in path.read_text().splitlines():
        values = [float(v) for v in line.split()]
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(f"Rótulo YOLO inválido: {path}")
        _, x, y, w, h = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Caixa fora dos limites: {path}")
        boxes.append([x, y, w, h])
    return boxes


def image_features(
    image: Image.Image, boxes: list[list[float]]
) -> dict[str, list[float]]:
    result = {key: [] for key in FEATURES}
    result["count"] = [len(boxes)]
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    h, w = lum.shape
    for x, y, bw, bh in boxes:
        left, right = max(0, int((x - bw / 2) * w)), min(
            w, int(np.ceil((x + bw / 2) * w))
        )
        top, bottom = max(0, int((y - bh / 2) * h)), min(
            h, int(np.ceil((y + bh / 2) * h))
        )
        patch = rgb[top:bottom, left:right]
        if not patch.size:
            continue
        light = lum[top:bottom, left:right]
        high, low = patch.max(axis=2), patch.min(axis=2)
        margin = max(2, round(min(right - left, bottom - top) * 0.5))
        l, r = max(0, left - margin), min(w, right + margin)
        t, b = max(0, top - margin), min(h, bottom + margin)
        region = lum[t:b, l:r]
        ring = np.ones(region.shape, dtype=bool)
        ring[top - t : bottom - t, left - l : right - l] = False
        ambient = float(np.mean(region[ring])) if ring.any() else float(light.mean())
        result["size"].append(float(np.sqrt(bw * bh)))
        result["center_y"].append(y)
        result["brightness"].append(float(light.mean()))
        result["saturation"].append(
            float(np.mean((high - low) / np.maximum(high, 1e-6)))
        )
        result["contrast"].append(
            float(np.log2((float(light.mean()) + 0.01) / (ambient + 0.01)))
        )
        result["clipped"].append(float((high >= 250 / 255).mean()))
    return result


def merge_features(items: list[dict]) -> dict:
    return {key: [v for item in items for v in item[key]] for key in FEATURES}


def dataset_features(images: Path, labels: Path, limit: int | None = None) -> dict:
    paths = image_files(images)
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images}")
    if limit is not None:
        paths = paths[:limit]
    features = []
    for path in paths:
        boxes = read_boxes(labels / f"{path.stem}.txt")
        with Image.open(path) as source:
            features.append(
                image_features(ImageOps.exif_transpose(source).convert("RGB"), boxes)
            )
    return merge_features(features)


def compare_features(synthetic: dict, real: dict) -> list[dict]:
    rows = []
    for key, (label, scale) in FEATURES.items():
        a, b = synthetic[key], real[key]
        if not a or not b:
            rows.append(
                dict(key=key, label=label, synthetic=None, real=None, distance=None)
            )
            continue
        # Distância entre quantis em unidades da própria variável; menor é
        # mais próximo. Sem soma de escalas incompatíveis nem "nota de realismo".
        q = np.linspace(0, 1, 101)
        distance = float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))) * scale
        rows.append(
            dict(
                key=key,
                label=label,
                synthetic=(np.quantile(a, [0.1, 0.5, 0.9]) * scale).tolist(),
                real=(np.quantile(b, [0.1, 0.5, 0.9]) * scale).tolist(),
                distance=distance,
            )
        )
    return rows
