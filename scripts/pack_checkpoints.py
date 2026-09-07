#!/usr/bin/env python3
"""Empacota os 42 checkpoints confirmatórios (`best.pt` de cada execução) e o
`model_selection.json` num único ZIP para publicação como asset de release.

Contraparte de `download_weights.py`, que baixa e reconstrói essa mesma
estrutura em outra máquina. Não participa do `run_pipeline.sh`.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fruit_pipeline.common import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "artifacts" / "confirmatory" / "model_selection.json"
OUT_PATH = ROOT / "artifacts" / "confirmatory" / "confirmatory_checkpoints.zip"


def main() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    run_ids = sorted(
        run["run_id"]
        for by_condition in selection["selected"].values()
        for candidate in by_condition.values()
        for run in candidate["runs"]
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_STORED) as archive:
        archive.write(SELECTION, "model_selection.json")
        for run_id in run_ids:
            checkpoint = ROOT / "runs" / "confirmatory" / "training" / run_id / "weights" / "best.pt"
            archive.write(checkpoint, f"weights/{run_id}.pt")
    size = OUT_PATH.stat().st_size
    digest = sha256_file(OUT_PATH)
    print(f"{len(run_ids)} checkpoints empacotados em {OUT_PATH.relative_to(ROOT)}")
    print(f"expected_bytes: {size}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
