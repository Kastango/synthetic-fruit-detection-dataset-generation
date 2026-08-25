#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.synthesis import create_asset_catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cataloga todos os fundos, mapas e recortes usados na síntese."
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
    result = create_asset_catalog(asset_root, force=args.force)
    print(
        json.dumps(
            {
                "asset_root": str(asset_root),
                "fingerprint": result["source_fingerprint"],
                "counts": {
                    key: len(values) for key, values in result["assets"].items()
                },
                "orphan_depth_maps": len(result["orphans"]["depth_maps"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
