#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.synthesis import create_asset_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Congela fundos e recortes disjuntos para treino/validação sintéticos."
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_yaml(project_path(args.config))
    asset_root = (
        args.asset_root.expanduser().resolve()
        if args.asset_root
        else project_path(config["paths"]["assets"])
    )
    split_config = config["asset_split"]
    result = create_asset_split(
        asset_root,
        train_ratio=float(split_config["train_ratio"]),
        seed=int(split_config["seed"]),
        force=args.force,
    )
    print(
        json.dumps(
            {
                "asset_root": str(asset_root),
                "fingerprint": result["source_fingerprint"],
                "counts": {
                    name: {key: len(values) for key, values in split.items()}
                    for name, split in result["splits"].items()
                },
                "orphan_depth_maps": len(result["orphans"]["depth_maps"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
