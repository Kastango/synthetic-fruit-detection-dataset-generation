#!/usr/bin/env python3
"""Avalia um checkpoint no split de desenvolvimento real (real-assisted).

Ferramenta interna do loop de otimização: usa apenas
data/real_yolo/images/val (15 imagens) ou, com --include-train,
val+train (115 imagens) para reduzir o ruído da métrica. Nunca referencia
o split de teste real, que permanece reservado para a avaliação
confirmatória final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fruit_pipeline.common import (
    atomic_write_json,
    atomic_write_text,
    image_files,
    project_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--real-yolo-root", default="data/real_yolo", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default="runs/optimization/real_val")
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="soma images/train ao conjunto de avaliação (115 imagens no total)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="grava o JSON de resultado neste caminho (stdout tem logs do YOLO misturados)",
    )
    args = parser.parse_args()

    import ultralytics
    from ultralytics import YOLO

    real_root = project_path(args.real_yolo_root)
    data_yaml_path = project_path(args.project) / f"{args.name}.yaml"
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if args.include_train:
        images = image_files(real_root / "images" / "val") + image_files(
            real_root / "images" / "train"
        )
        list_path = data_yaml_path.with_suffix(".txt")
        atomic_write_text(
            list_path, "\n".join(str(path.resolve()) for path in images) + "\n"
        )
        val_target = str(list_path)
    else:
        val_target = "images/val"
    value = {
        "path": str(real_root.resolve()),
        "train": "images/val",
        "val": val_target,
        "names": {0: "poncan"},
    }
    atomic_write_text(
        data_yaml_path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    )

    metrics = YOLO(str(args.checkpoint)).val(
        data=str(data_yaml_path),
        split="val",
        device=args.device,
        project=str(project_path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    result = {
        "name": args.name,
        "checkpoint": str(args.checkpoint),
        "ultralytics_version": ultralytics.__version__,
        "eval_set": "real_train_val_115" if args.include_train else "real_val_15",
        "real_val": {
            "precision": round(float(metrics.box.mp), 6),
            "recall": round(float(metrics.box.mr), 6),
            "map50": round(float(metrics.box.map50), 6),
            "map50_95": round(float(metrics.box.map), 6),
        },
    }
    if args.output:
        atomic_write_json(project_path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
