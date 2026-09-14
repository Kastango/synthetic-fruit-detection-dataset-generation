#!/usr/bin/env python3
"""Seleciona a maior diferença de acertos na validação com correspondência 1:1."""
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from analyze_misses import read_boxes
from evaluate_operating_points import match_counts
from fruit_pipeline.common import ROOT, atomic_write_json, sha256_file


def main():
    selection_path = ROOT / "artifacts/confirmatory/model_selection.json"
    selection = json.loads(selection_path.read_text())
    root = ROOT / "data/real_yolo_confirmatory"
    images = sorted((root / "images/val").glob("*.jpg"))
    conditions = ["manual-full"] + [f"synthetic-{n}x" for n in [1, 2, 3, 5, 10]]
    rows = {p.stem: {"image": p.stem, "hits": {}} for p in images}
    for condition in conditions:
        run = next(r for r in selection["selected"]["yolo26s"][condition]["runs"] if r["seed"] == 41)
        assert sha256_file(Path(run["checkpoint"])) == run["checkpoint_sha256"]
        model = YOLO(run["checkpoint"])
        for path in images:
            with Image.open(path) as im:
                gt = read_boxes(root / "labels/val" / f"{path.stem}.txt", *im.size)
            prediction = model.predict(str(path), conf=0.25, imgsz=960, max_det=1000, device="0", verbose=False)[0]
            tp, fp, fn = match_counts(gt, prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy(), 0.25)
            rows[path.stem]["ground_truth"] = len(gt)
            rows[path.stem]["hits"][condition] = {"tp": tp, "fp": fp, "fn": fn}
        print(condition, "OK", flush=True)
    for row in rows.values():
        row["best_synthetic_tp"] = max(row["hits"][c]["tp"] for c in conditions[1:])
        row["gap"] = row["hits"]["manual-full"]["tp"] - row["best_synthetic_tp"]
    ranked = sorted(rows.values(), key=lambda r: (-r["gap"], r["image"]))
    output = ROOT / "docs/figures/results/examples/manual-full-val/yolo26s/gap-audit.json"
    atomic_write_json(output, {"model": "yolo26s", "seed": 41, "confidence": 0.25, "iou": 0.5,
                               "matching": "confidence-greedy one-to-one", "selection_sha256": sha256_file(selection_path), "ranking": ranked})
    print(json.dumps(ranked[:3], indent=2))


if __name__ == "__main__":
    main()
