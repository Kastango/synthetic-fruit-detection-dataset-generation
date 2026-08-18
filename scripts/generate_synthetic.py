#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.synthesis import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera um dataset YOLO sintético determinístico e retomável."
    )
    parser.add_argument("--synthesis-config", required=True, type=Path)
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pipeline = load_yaml(project_path(args.pipeline_config))
    synthesis = load_yaml(args.synthesis_config.expanduser().resolve())
    asset_root = (
        args.asset_root.expanduser().resolve()
        if args.asset_root
        else project_path(pipeline["paths"]["assets"])
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else project_path(pipeline["paths"]["generated"]) / synthesis["name"]
    )
    split = pipeline["asset_split"]
    summary = generate_dataset(
        asset_root,
        output,
        synthesis,
        train_ratio=float(split["train_ratio"]),
        split_seed=int(split["seed"]),
        workers=max(1, args.workers),
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
