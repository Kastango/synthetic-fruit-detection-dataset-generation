"""Caixas YOLO em pixels, IoU e correspondência um a um entre previsão e gabarito."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_boxes(path: Path, width: int, height: int) -> np.ndarray:
    """Lê um rótulo YOLO e devolve as caixas como `x0, y0, x1, y1` em pixels."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return np.zeros((0, 4))
    values = np.array([[float(value) for value in line.split()[1:5]] for line in lines])
    cx, cy = values[:, 0] * width, values[:, 1] * height
    w, h = values[:, 2] * width, values[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / np.maximum(
        area_a[:, None] + area_b[None, :] - intersection, 1e-9
    )


def match_counts(
    ground_truth: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    iou: float = 0.5,
) -> tuple[int, int, int]:
    """Conta acertos, falsos positivos e falsos negativos.

    As previsões acima do limiar de confiança são visitadas da mais confiante
    para a menos confiante. Cada uma fica com a caixa livre de maior IoU, desde
    que o IoU alcance `iou`.
    """
    boxes = boxes[scores >= threshold]
    scores = scores[scores >= threshold]
    overlaps = iou_matrix(boxes, ground_truth)
    used: set[int] = set()
    for index in np.argsort(-scores, kind="stable"):
        candidates = [
            j
            for j in range(len(ground_truth))
            if j not in used and overlaps[index, j] >= iou
        ]
        if candidates:
            used.add(max(candidates, key=lambda j: overlaps[index, j]))
    return len(used), len(boxes) - len(used), len(ground_truth) - len(used)
