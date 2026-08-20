#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.download import download_real_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa e valida o ZIP da base real anotada."
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--accept-data-terms",
        action="store_true",
        help="Confirma os termos de uso dos dados",
    )
    args = parser.parse_args()
    if not args.accept_data_terms:
        raise SystemExit(
            "use --accept-data-terms para confirmar os termos de uso dos dados"
        )
    config = load_yaml(project_path(args.config))
    archive = download_real_source(config, force=args.force)
    print(json.dumps({"archive": str(archive)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
