#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.training import run_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa a grade YOLO em processos isolados e retomáveis."
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    experiment = load_yaml(project_path(args.config))
    pipeline = load_yaml(project_path(args.pipeline_config))
    results = run_grid(
        experiment,
        runs_root=project_path(pipeline["paths"]["runs"]),
        condition_filters=args.condition,
        model_filters=args.model,
        max_runs=args.max_runs,
        device=args.device,
        dry_run=args.dry_run,
        force=args.force,
        continue_on_error=args.continue_on_error,
    )
    if results:
        print(json.dumps({"completed": len(results)}, indent=2))


if __name__ == "__main__":
    main()
