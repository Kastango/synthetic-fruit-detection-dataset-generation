#!/usr/bin/env python3
"""Gera um candidato sintético pequeno e treina uma rodada smoke/screen contra
sua própria validação sintética. Ferramenta interna do loop de otimização
guiado por $tune-synthetic-fruit-detector; não faz parte do protocolo
confirmatório e nunca lê validação ou teste reais.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import ultralytics

from fruit_pipeline.common import load_yaml, project_path, stable_hash
from fruit_pipeline.synthesis import generate_dataset
from fruit_pipeline.training import train_one


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roda um trial smoke/screen strict-synthetic."
    )
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--synthesis-config", required=True, type=Path)
    parser.add_argument("--asset-root", default="data/assets/regenerated", type=Path)
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--train-images", type=int, required=True)
    parser.add_argument("--val-images", type=int, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=8, help="workers da síntese")
    parser.add_argument("--generated-root", default="data/generated/optimization")
    parser.add_argument("--runs-root", default="runs/optimization")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pipeline = load_yaml(project_path(args.pipeline_config))
    asset_root = args.asset_root.expanduser().resolve()
    synthesis_config = copy.deepcopy(
        load_yaml(args.synthesis_config.expanduser().resolve())
    )
    synthesis_config["images"] = {"train": args.train_images, "val": args.val_images}
    synthesis_config["name"] = f"{synthesis_config['name']}-{args.trial_id}"

    output_root = project_path(args.generated_root) / args.trial_id
    split = pipeline["asset_split"]
    synth_summary = generate_dataset(
        asset_root,
        output_root,
        synthesis_config,
        train_ratio=float(split["train_ratio"]),
        split_seed=int(split["seed"]),
        workers=max(1, args.workers),
        force=args.force,
    )

    parameters = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.epochs,
        "optimizer": "SGD",
        "pretrained": True,
        "deterministic": True,
        "workers": 4,
        "device": args.device,
        "close_mosaic": 0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    train_dir = output_root / "images" / "train"
    val_dir = output_root / "images" / "val"
    candidate = {
        "condition": f"optimization_{args.trial_id}",
        "model": args.model,
        "parameters": parameters,
        "train": [str(train_dir.resolve())],
        "validation": str(val_dir.resolve()),
        "class_names": pipeline["project"]["class_names"],
        "dataset_fingerprint": synth_summary["config_hash"],
        "ultralytics_version": ultralytics.__version__,
    }
    candidate_id = stable_hash(candidate, 16)
    spec = {**candidate, "candidate_id": candidate_id, "seed": args.seed}
    spec["run_id"] = f"{args.trial_id}__s{args.seed}"

    result = train_one(spec, project_path(args.runs_root), force=args.force)
    report = {
        "trial_id": args.trial_id,
        "synthesis_summary": synth_summary,
        "training": result,
    }
    report_path = (
        project_path(args.runs_root) / "training" / spec["run_id"] / "trial_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
