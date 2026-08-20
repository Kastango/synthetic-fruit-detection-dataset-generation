#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.real_data import materialize_controlled_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materializa a condição controlled (fotos de fruta + fundos negativos)."
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = materialize_controlled_dataset(
        load_yaml(project_path(args.config)), force=args.force
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
