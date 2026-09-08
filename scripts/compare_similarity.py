#!/usr/bin/env python3
"""Compara distribuições completas, sem usar o desempenho para ajustar imagens."""

import argparse
import json
from pathlib import Path
from fruit_pipeline.common import ROOT, atomic_write_json
from fruit_pipeline.similarity import dataset_features, compare_features

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    synthetic = dataset_features(
        a.dataset / "images" / a.split, a.dataset / "labels" / a.split
    )
    report = {
        "dataset": str(a.dataset),
        "split": a.split,
        "images": len(synthetic["count"]),
        "boxes": len(synthetic["size"]),
        "interpretation": "Descritores em caixas incluem folhagem; sem nota de realismo. CitDet usado no desenvolvimento.",
    }
    for name, root, split in [
        ("manual_train", ROOT / "data/real_yolo_confirmatory", "train"),
        ("citdet", ROOT / "data/external_tests/citdet", "test"),
    ]:
        real = dataset_features(root / "images" / split, root / "labels" / split)
        report[name] = {
            "images": len(real["count"]),
            "boxes": len(real["size"]),
            "features": compare_features(synthetic, real),
        }
    atomic_write_json(a.output, report)
    print(json.dumps(report, indent=2))
