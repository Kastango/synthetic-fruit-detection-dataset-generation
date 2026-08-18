#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.training import evaluate_selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Abre uma única vez o teste real para os modelos já selecionados."
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.unlock_test:
        raise SystemExit(
            "teste bloqueado: finalize train_grid.py e select_models.py; depois confirme "
            "a avaliação final com --unlock-test"
        )
    experiment = load_yaml(project_path(args.config))
    pipeline = load_yaml(project_path(args.pipeline_config))
    report = evaluate_selected(
        experiment,
        project_path(pipeline["paths"]["runs"]),
        project_path(pipeline["paths"]["artifacts"]),
        device=args.device,
        force=args.force,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
