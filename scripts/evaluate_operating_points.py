#!/usr/bin/env python3
"""Precision/recall com matching 1:1 e limiares fixados antes do teste."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fruit_pipeline.boxes import match_counts, read_boxes
from fruit_pipeline.common import atomic_write_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    from ultralytics import YOLO

    for dataset in ("oranges_field", "manual_full_val"):
        report = json.loads((args.artifacts / f"test_results_{dataset}.json").read_text())
        images = Path(report["test_path"])
        if report["external_manifest_sha256"] != sha256_file(images.parents[1] / "manifest.json"):
            raise RuntimeError(
                f"{dataset}: relatório pertence a outro manifesto; "
                "reavalie antes de calcular os pontos operacionais"
            )
        labels = images.parent.parent / "labels" / images.name
        results = []
        for run in report["results"]:
            spec_path = args.artifacts / f"test_specs_{dataset}" / f"{run['run_id']}.json"
            spec = json.loads(spec_path.read_text())
            thresholds = {"fixed_025": 0.25, "validation_f1": float(spec["confidence_threshold"])}
            counts = {name: np.zeros(3, dtype=np.int64) for name in thresholds}
            model = YOLO(run["checkpoint"])
            n = 0
            predictions = model.predict(
                source=str(images), stream=True, conf=min(thresholds.values()),
                imgsz=960, max_det=1000, device=args.device, verbose=False,
            )
            for prediction in predictions:
                height, width = prediction.orig_shape
                label_path = labels / (Path(prediction.path).stem + ".txt")
                if not label_path.exists():
                    raise FileNotFoundError(label_path)
                gt = read_boxes(label_path, width, height)
                boxes = prediction.boxes.xyxy.cpu().numpy()
                scores = prediction.boxes.conf.cpu().numpy()
                for name, threshold in thresholds.items():
                    counts[name] += match_counts(gt, boxes, scores, threshold)
                n += 1
            for name, (tp, fp, fn) in counts.items():
                results.append({
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "run_id": run["run_id"],
                    "operating_point": name,
                    "confidence": thresholds[name],
                    "images": n,
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                    "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                    "fp_per_image": float(fp / n) if n else 0.0,
                })
            atomic_write_json(
                args.artifacts / f"operating_points_{dataset}.json",
                {
                    "dataset": dataset,
                    "matching": "confidence-greedy one-to-one IoU>=0.5",
                    "results": results,
                },
            )
            print(dataset, run["run_id"], "OK", flush=True)


if __name__ == "__main__":
    main()
