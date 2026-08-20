#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.reporting import generate_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera o Markdown consolidado dos detectores."
    )
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--external-name")
    args = parser.parse_args()
    experiment = load_yaml(project_path(args.config))
    if args.external_name:
        experiment["protocol"]["external_test"] = args.external_name
    destination = generate_markdown_report(
        experiment,
        load_yaml(project_path(args.pipeline_config)),
        output=args.output.expanduser().resolve() if args.output else None,
    )
    print(destination)


if __name__ == "__main__":
    main()
