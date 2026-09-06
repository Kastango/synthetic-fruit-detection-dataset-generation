#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.synthesis import materialize_nested_subsets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recorta os subconjuntos aninhados 1x-10x do pool sintético confirmatório."
    )
    parser.add_argument(
        "--pool-root", type=Path, default=Path("data/generated/confirmatory_pool")
    )
    parser.add_argument("--target-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--multipliers", type=int, nargs="+")
    parser.add_argument("--base-size", type=int)
    parser.add_argument("--base-val-size", type=int)
    parser.add_argument("--prefix", default="synthetic-")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pipeline = load_yaml(project_path(args.pipeline_config))
    subset_config = pipeline["synthetic_subsets"]
    summary = materialize_nested_subsets(
        args.pool_root,
        args.target_root,
        args.multipliers or [int(value) for value in subset_config["multipliers"]],
        base_size=(
            args.base_size
            if args.base_size is not None
            else int(subset_config["base_train_images"])
        ),
        base_val_size=(
            args.base_val_size
            if args.base_val_size is not None
            else int(subset_config["base_val_images"])
        ),
        prefix=args.prefix,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
