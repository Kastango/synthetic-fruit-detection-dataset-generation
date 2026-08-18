from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .common import image_files


def _feature_extractor(device: str):
    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model, weights.transforms()


def extract_inception_features(
    paths: list[Path], device: str = "cpu", batch_size: int = 16
) -> np.ndarray:
    import torch

    model, preprocess = _feature_extractor(device)
    features = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        tensor = torch.stack(batch).to(device)
        with torch.inference_mode():
            output = model(tensor)
        features.append(output.cpu().numpy())
        batch.clear()

    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        batch.append(preprocess(image))
        if len(batch) >= batch_size:
            flush()
    flush()
    return np.concatenate(features, axis=0)


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    distance = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(distance)


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(
    real_paths: list[Path],
    synthetic_paths: list[Path],
    device: str = "cpu",
) -> float:
    real_features = extract_inception_features(real_paths, device=device)
    synthetic_features = extract_inception_features(synthetic_paths, device=device)
    real_mu, real_sigma = feature_statistics(real_features)
    synthetic_mu, synthetic_sigma = feature_statistics(synthetic_features)
    return frechet_distance(real_mu, real_sigma, synthetic_mu, synthetic_sigma)


def collect_real_images(real_yolo_root: Path) -> list[Path]:
    train = image_files(real_yolo_root / "images" / "train")
    val = image_files(real_yolo_root / "images" / "val")
    return sorted(train + val)
