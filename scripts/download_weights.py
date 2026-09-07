#!/usr/bin/env python3
"""Baixa os 42 checkpoints confirmatórios já treinados e o `model_selection.json`
correspondente, sem precisar rodar `train`/`select`. Junto com os dados (`prepare`
ou `download-raw` + `import-real`) e o teste externo desejado (`prepare-test`),
isso é suficiente para `evaluate_test.py` / `generate_report.py` /
`scripts/plot_confirmatory_results.py` num computador novo.

Os caminhos de checkpoint gravados em `model_selection.json` são absolutos e
específicos da máquina onde o treino rodou; este script os reescreve para o
projeto local depois de extrair os pesos.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from fruit_pipeline.common import atomic_write_json, load_yaml, project_path
from fruit_pipeline.download import download_http

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pipeline = load_yaml(project_path(args.config))
    entry = pipeline["confirmatory_checkpoints"]

    archive = project_path(pipeline["paths"]["archives"]) / "confirmatory_checkpoints.zip"
    download_http(
        str(entry["download_url"]),
        archive,
        expected_bytes=int(entry["expected_bytes"]),
        expected_sha256=str(entry["sha256"]),
        force=args.force,
    )

    runs_root = project_path("runs/confirmatory/training")
    with zipfile.ZipFile(archive) as bundle:
        selection = json.loads(bundle.read("model_selection.json"))
        run_ids = [name[len("weights/") : -len(".pt")] for name in bundle.namelist() if name.startswith("weights/")]
        for run_id in run_ids:
            destination = runs_root / run_id / "weights" / "best.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not args.force:
                continue
            with bundle.open(f"weights/{run_id}.pt") as source, destination.open("wb") as handle:
                handle.write(source.read())

    for by_condition in selection["selected"].values():
        for candidate in by_condition.values():
            for run in candidate["runs"]:
                local = runs_root / run["run_id"] / "weights" / "best.pt"
                run["checkpoint"] = str(local.resolve())

    selection_path = project_path(pipeline["paths"]["artifacts"]) / "confirmatory" / "model_selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_path, selection)
    print(f"{len(run_ids)} checkpoints extraídos em {runs_root.relative_to(ROOT)}")
    print(f"model_selection.json escrito em {selection_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
