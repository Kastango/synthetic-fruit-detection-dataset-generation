#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.training import scoped_experiment_root, select_by_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seleciona checkpoints pela validação de origem."
    )
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--model",
        action="append",
        help="seleciona só estas arquiteturas; repita para várias. "
        "Todas as sementes de cada uma continuam obrigatórias.",
    )
    args = parser.parse_args()
    experiment = load_yaml(project_path(args.config))
    pipeline = load_yaml(project_path(args.pipeline_config))
    if args.model:
        experiment["models"] = [
            model
            for model in experiment["models"]
            if model["name"] in args.model or model["checkpoint"] in args.model
        ]
        if not experiment["models"]:
            raise SystemExit(f"nenhuma arquitetura casa com {args.model}")
    report = select_by_validation(
        experiment,
        scoped_experiment_root(
            project_path(pipeline["paths"]["runs"]), experiment, "runs"
        ),
        scoped_experiment_root(
            project_path(pipeline["paths"]["artifacts"]), experiment, "artifact"
        ),
    )
    selected = {
        model_name: {
            condition: {
                "model": item["model"],
                "validation_map50_95_mean": item["validation_map50_95_mean"],
                "candidate_id": item["candidate_id"],
            }
            for condition, item in by_condition.items()
        }
        for model_name, by_condition in report["selected"].items()
    }
    print(json.dumps(selected, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
