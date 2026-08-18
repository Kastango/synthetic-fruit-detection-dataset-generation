#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import PIL

from fruit_pipeline.common import (
    atomic_write_json,
    load_yaml,
    project_path,
    stable_hash,
)
from fruit_pipeline.synthesis import generate_dataset


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("o valor deve ser positivo")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mede o throughput do gerador sem preservar os datasets temporários."
    )
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument(
        "--synthesis-config",
        default="configs/synthesis/confirmatory_pool.yaml",
        type=Path,
    )
    parser.add_argument("--train-images", type=_positive_int, default=24)
    parser.add_argument("--val-images", type=_positive_int, default=8)
    parser.add_argument(
        "--workers", type=_positive_int, action="append", dest="worker_counts"
    )
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    asset_root = args.asset_root.expanduser().resolve()
    config_path = (
        args.synthesis_config.expanduser().resolve()
        if args.synthesis_config.is_absolute()
        else project_path(args.synthesis_config)
    )
    config = copy.deepcopy(load_yaml(config_path))
    config["name"] = f"benchmark-{config['name']}"
    config["images"] = {"train": args.train_images, "val": args.val_images}
    worker_counts = args.worker_counts or [1]
    image_count = args.train_images + args.val_images
    rows = []

    with tempfile.TemporaryDirectory(prefix="fruit-synthesis-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for workers in worker_counts:
            elapsed_values = []
            summaries = []
            for repetition in range(args.repeats):
                output_root = temporary_root / f"w{workers}-r{repetition}"
                started = time.perf_counter()
                summary = generate_dataset(
                    asset_root,
                    output_root,
                    config,
                    train_ratio=0.8,
                    split_seed=42,
                    workers=workers,
                )
                elapsed_values.append(time.perf_counter() - started)
                summaries.append(summary)
            median_seconds = statistics.median(elapsed_values)
            rows.append(
                {
                    "workers": workers,
                    "seconds": [round(value, 6) for value in elapsed_values],
                    "median_seconds": round(median_seconds, 6),
                    "median_images_per_second": round(image_count / median_seconds, 6),
                    "annotations": summaries[-1]["annotations"],
                    "negative_images": summaries[-1]["negative_images"],
                }
            )

    report = {
        "schema_version": 1,
        "config": str(config_path),
        "config_hash": stable_hash(config, 24),
        "asset_root": str(asset_root),
        "images_per_repeat": image_count,
        "repeats": args.repeats,
        "environment": {
            "python": platform.python_version(),
            "pillow": PIL.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "results": rows,
    }
    if args.output:
        atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
