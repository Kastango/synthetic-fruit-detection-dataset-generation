#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.real_data import import_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importa a base real anotada em YOLO, COCO ou CVAT e audita 130/2093."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--format", choices=("auto", "yolo", "coco", "cvat"), default="auto"
    )
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--allow-unknown-device", action="store_true")
    parser.add_argument(
        "--allow-processed-input",
        action="store_true",
        help="Permite imagens redimensionadas/Roboflow; não recomendado para o protocolo",
    )
    parser.add_argument("--skip-count-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_yaml(project_path(args.config))
    real = config["real_dataset"]
    target = project_path(config["paths"]["real_source"])
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = import_real_dataset(
        args.source.expanduser().resolve(),
        target,
        metadata_path=args.metadata_csv.expanduser().resolve()
        if args.metadata_csv
        else None,
        annotation_format=args.format,
        expected_images=None if args.skip_count_check else int(real["expected_images"]),
        expected_boxes=None if args.skip_count_check else int(real["expected_boxes"]),
        allow_unknown_device=args.allow_unknown_device,
        allow_processed_input=args.allow_processed_input,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
