#!/usr/bin/env python3
from __future__ import annotations

import argparse

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.preprocess import preprocess_assets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstrói fundos normalizados, recortes DIS e mapas ZoeDepth."
    )
    parser.add_argument(
        "--stage", choices=("all", "normalize", "segment", "depth"), default="all"
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps ou cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preprocess_assets(
        load_yaml(project_path(args.config)),
        stage=args.stage,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
