#!/usr/bin/env python3
"""Agrega os resultados de um candidato (2 seeds + FID) em um único JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from fruit_pipeline.common import atomic_write_json, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--runs-root", default="runs/optimization/training", type=Path)
    parser.add_argument(
        "--summary-dir", default="runs/optimization/candidate_summaries", type=Path
    )
    args = parser.parse_args()

    runs_root = project_path(args.runs_root)
    summary_dir = project_path(args.summary_dir)

    synthetic_values = []
    real_values = []
    training_seconds = []
    for seed in (41, 42):
        report = json.loads(
            (
                runs_root / f"{args.trial_id}-s{seed}__s{seed}" / "trial_report.json"
            ).read_text(encoding="utf-8")
        )
        synthetic_values.append(report["training"]["validation"]["map50_95"])
        training_seconds.append(report["training"]["training_seconds"])
        real = json.loads(
            (summary_dir / f"{args.trial_id}-s{seed}-real.json").read_text(
                encoding="utf-8"
            )
        )
        real_values.append(real["real_val"]["map50_95"])

    fid = json.loads(
        (summary_dir / f"{args.trial_id}-fid.json").read_text(encoding="utf-8")
    )["fid"]

    summary = {
        "trial_id": args.trial_id,
        "seeds": [41, 42],
        "synthetic_map50_95": {
            "values": synthetic_values,
            "mean": round(statistics.fmean(synthetic_values), 6),
        },
        "real_map50_95": {
            "values": real_values,
            "mean": round(statistics.fmean(real_values), 6),
            "std": round(statistics.stdev(real_values), 6),
        },
        "fid": fid,
        "training_seconds_total": round(sum(training_seconds), 2),
    }
    atomic_write_json(summary_dir / f"{args.trial_id}-summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
