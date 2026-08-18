#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.training import select_by_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seleciona hiperparâmetros pela média da validação real, sem ler o teste."
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    experiment = load_yaml(project_path(args.config))
    pipeline = load_yaml(project_path(args.pipeline_config))
    report = select_by_validation(
        experiment,
        project_path(pipeline["paths"]["runs"]),
        project_path(pipeline["paths"]["artifacts"]),
    )
    selected = {
        condition: {
            "model": item["model"],
            "validation_map50_95_mean": item["validation_map50_95_mean"],
            "candidate_id": item["candidate_id"],
        }
        for condition, item in report["selected"].items()
    }
    print(json.dumps(selected, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
