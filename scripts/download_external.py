#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.download import obtain_external_archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa ou recebe e valida criptograficamente um teste externo."
    )
    parser.add_argument("name", help="nome em external_datasets, ex.: citdet")
    parser.add_argument("--source", type=Path, help="arquivo já baixado")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_yaml(project_path(args.config))
    archive = obtain_external_archive(
        config,
        args.name,
        source=args.source.expanduser() if args.source else None,
        force=args.force,
    )
    print(json.dumps({"name": args.name, "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
