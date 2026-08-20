from __future__ import annotations

import json
from pathlib import Path

from fruit_pipeline.reporting import generate_markdown_report


def test_report_highlights_only_positive_synthetic_delta(tmp_path: Path) -> None:
    experiment = {
        "protocol": {"artifact_subdir": "confirmatory", "external_test": "fixture"},
        "conditions": {"manual-full": {}, "synthetic-1x": {}},
    }
    pipeline = {"paths": {"artifacts": str(tmp_path / "artifacts")}}
    artifacts = tmp_path / "artifacts" / "confirmatory"
    artifacts.mkdir(parents=True)
    runs = {
        condition: [
            {
                "run_id": f"{condition}-s41",
                "seed": 41,
                "training_seconds": 60,
                "validation": {"map50_95": 0.4},
            }
        ]
        for condition in experiment["conditions"]
    }
    selection = {
        "selected": {
            "yolo": {
                condition: {"runs": condition_runs}
                for condition, condition_runs in runs.items()
            }
        }
    }
    (artifacts / "model_selection.json").write_text(json.dumps(selection))
    summaries = {
        "manual-full": {
            "precision_mean": 0.5,
            "recall_mean": 0.5,
            "f1_mean": 0.5,
            "map50_mean": 0.6,
            "map75_mean": 0.4,
            "map50_95_mean": 0.4,
            "delta_vs_manual_full": 0.0,
            "count_mae_mean": 2.0,
        },
        "synthetic-1x": {
            "precision_mean": 0.6,
            "recall_mean": 0.6,
            "f1_mean": 0.6,
            "map50_mean": 0.7,
            "map75_mean": 0.5,
            "map50_95_mean": 0.5,
            "delta_vs_manual_full": 0.1,
            "count_mae_mean": 1.0,
        },
    }
    evaluations = [
        {
            "run_id": condition_runs[0]["run_id"],
            "test": {"map50_95": summaries[condition]["map50_95_mean"]},
        }
        for condition, condition_runs in runs.items()
    ]
    results = {
        "selection_sha256": "a" * 64,
        "external_manifest_sha256": "b" * 64,
        "test_opened_after_selection": True,
        "summary": {"yolo": summaries},
        "results": evaluations,
    }
    (artifacts / "test_results_fixture.json").write_text(json.dumps(results))

    destination = generate_markdown_report(experiment, pipeline)
    text = destination.read_text()

    assert destination == artifacts / "RESULTS_fixture.md"
    assert "**synthetic-1x** ★" in text
    assert "superou descritivamente" in text
