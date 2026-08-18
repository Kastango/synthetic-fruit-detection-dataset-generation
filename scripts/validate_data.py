#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.validation import validate_project


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita dados, splits e rótulos antes do treino."
    )
    parser.add_argument(
        "--stage", choices=("all", "real", "assets", "generated"), default="all"
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    config = load_yaml(project_path(args.config))
    report = validate_project(
        config,
        stage=args.stage,
        asset_root=args.asset_root.expanduser().resolve() if args.asset_root else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
