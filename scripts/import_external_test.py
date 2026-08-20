#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.external_test import import_external_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Importa um conjunto de teste externo (diretório ou ZIP, YOLO/COCO/CVAT "
            "detectado automaticamente) para avaliação fora do domínio sintético/real."
        )
    )
    parser.add_argument("name", help="nome curto do conjunto, ex.: citdet")
    parser.add_argument("--source", required=True, type=Path, help="diretório ou ZIP")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_yaml(project_path(args.config))
    dataset = config["external_datasets"][args.name]
    target_root = project_path(config["paths"]["external_tests"])
    summary = import_external_test(
        args.name,
        args.source.expanduser().resolve(),
        target_root,
        config["project"]["class_names"],
        annotation_format=str(dataset["annotation_format"]),
        nested_archive=dataset.get("nested_archive"),
        collapse_to_single_class=bool(dataset["collapse_to_single_class"]),
        expected_images=dataset.get("expected_images"),
        expected_boxes=dataset.get("expected_boxes"),
        source_metadata={key: dataset[key] for key in ("landing_page", "license")},
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
