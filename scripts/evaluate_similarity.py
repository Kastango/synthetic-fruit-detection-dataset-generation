#!/usr/bin/env python3
"""Avalia os checkpoints de um ciclo YOLOv8 nos dois cenários reais."""

from pathlib import Path
import argparse
import json
from fruit_pipeline.common import ROOT, atomic_write_json, load_yaml
from fruit_pipeline.training import expand_experiments, evaluate_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--config", default="configs/similarity_yolov8.yaml")
    a = p.parse_args()
    config = load_yaml(ROOT / a.config)
    specs = expand_experiments(config)
    root = ROOT / "runs" / config["protocol"]["runs_subdir"]
    artifacts = ROOT / "artifacts" / config["protocol"]["artifact_subdir"]
    selected = []
    for spec in specs:
        path = root / "training" / spec["run_id"] / "result.json"
        if not path.exists():
            raise SystemExit(f'Treino ainda não concluído: {spec["run_id"]}')
        result = json.loads(path.read_text())
        selected.append(result)
    atomic_write_json(artifacts / "paired_selection.json", selected)
    results = []
    for result in selected:
        for name, test in [
            ("manual_full_val", ROOT / "data/real_yolo_confirmatory/images/val"),
            ("citdet", ROOT / "data/external_tests/citdet/images/test"),
        ]:
            evaluation = {
                **result,
                "ultralytics_version": config["protocol"]["ultralytics_version"],
                "class_names": ["poncan"],
                "test": str(test),
                "device": a.device,
                "imgsz": 960,
                "batch": 8,
                "workers": 4,
                "max_det": 1000,
                "confidence_threshold": result["validation"].get("best_f1_confidence")
                or 0.25,
                "evaluation_root": str(root / "evaluation" / name),
                "output": str(artifacts / name / f'{result["run_id"]}.json'),
            }
            measured = evaluate_checkpoint(evaluation)
            results.append({"domain": name, **measured})
    atomic_write_json(artifacts / "paired_results.json", results)
    print(
        json.dumps(
            [
                {
                    "condition": r["condition"],
                    "seed": r["seed"],
                    "domain": r["domain"],
                    "metrics": r["test"],
                }
                for r in results
            ],
            indent=2,
        )
    )
