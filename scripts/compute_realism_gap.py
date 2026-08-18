#!/usr/bin/env python3
"""Mede o gap de realismo (FID) entre um lote sintético e o corpus real.

Diagnóstico complementar ao mAP: acompanha se as cenas sintéticas estão se
aproximando da distribuição visual das fotos reais (treino+val, 115
imagens; o teste real nunca é usado). FID mais baixo é mais próximo do
real. Ferramenta interna do loop de otimização, não faz parte do
protocolo confirmatório.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fruit_pipeline.common import project_path
from fruit_pipeline.realism import (
    collect_real_images,
    extract_inception_features,
    feature_statistics,
    frechet_distance,
)


def _real_stats(
    real_yolo_root: Path, cache_path: Path, device: str, force: bool
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        return cached["mu"], cached["sigma"]
    real_paths = collect_real_images(real_yolo_root)
    features = extract_inception_features(real_paths, device=device)
    mu, sigma = feature_statistics(features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma, n_images=len(real_paths))
    return mu, sigma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--real-yolo-root", default="data/real_yolo", type=Path)
    parser.add_argument(
        "--real-stats-cache", default="artifacts/optimization/real_fid_stats.npz"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", required=True)
    parser.add_argument("--force-real-stats", action="store_true")
    args = parser.parse_args()

    real_yolo_root = project_path(args.real_yolo_root)
    cache_path = project_path(args.real_stats_cache)
    real_mu, real_sigma = _real_stats(
        real_yolo_root, cache_path, args.device, args.force_real_stats
    )

    synthetic_dir = args.synthetic_dir.expanduser().resolve()
    synthetic_paths = sorted(
        path
        for path in synthetic_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not synthetic_paths:
        raise SystemExit(f"nenhuma imagem sintética em {synthetic_dir}")
    synthetic_features = extract_inception_features(synthetic_paths, device=args.device)
    synthetic_mu, synthetic_sigma = feature_statistics(synthetic_features)
    fid = frechet_distance(real_mu, real_sigma, synthetic_mu, synthetic_sigma)

    result = {
        "name": args.name,
        "synthetic_dir": str(synthetic_dir),
        "synthetic_images": len(synthetic_paths),
        "real_images": 115,
        "fid": round(fid, 4),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
