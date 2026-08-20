from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image

from fruit_pipeline.reporting import generate_markdown_report


def _write_labels(image_directory: Path, *contents: str) -> None:
    image_directory.mkdir(parents=True)
    label_directory = image_directory.parent.parent / "labels" / image_directory.name
    label_directory.mkdir(parents=True)
    for index, text in enumerate(contents):
        (label_directory / f"sample-{index}.txt").write_text(text)


def _write_training_run(root: Path, condition: str, run: dict) -> None:
    directory = root / "training" / run["run_id"]
    directory.mkdir(parents=True)
    result = {
        "run_id": run["run_id"],
        "candidate_id": f"candidate-{condition}",
        "condition": condition,
        "model_name": "yolo",
        "seed": run["seed"],
        "parameters": {"epochs": 2, "imgsz": 960},
        "dataset_fingerprint": f"fingerprint-{condition}",
        "training_seconds": run["training_seconds"],
        "checkpoint": f"/checkpoints/{run['run_id']}.pt",
        "checkpoint_sha256": "c" * 64,
        "ultralytics_version": "fixture",
        "torch_version": "fixture",
        "hardware": {"gpu": "fixture"},
        "validation": run["validation"],
    }
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B),train/box_loss,val/box_loss\n"
        "1,0.30,1.0,1.1\n"
        "2,0.40,0.8,0.9\n"
    )


def test_report_highlights_only_positive_synthetic_delta(tmp_path: Path) -> None:
    manual = tmp_path / "manual"
    synthetic = tmp_path / "synthetic"
    external = tmp_path / "external"
    experiment = {
        "protocol": {
            "artifact_subdir": "confirmatory",
            "external_test": "fixture",
            "validation": str(manual / "images" / "val"),
        },
        "conditions": {
            "manual-full": {"train": [str(manual / "images" / "train")]},
            "synthetic-1x": {
                "train": [str(synthetic / "images" / "train")],
                "validation": str(synthetic / "images" / "val"),
            },
        },
    }
    pipeline = {
        "paths": {
            "artifacts": str(tmp_path / "artifacts"),
            "external_tests": str(external),
            "runs": str(tmp_path / "runs"),
        }
    }
    artifacts = tmp_path / "artifacts" / "confirmatory"
    artifacts.mkdir(parents=True)
    _write_labels(
        manual / "images" / "train",
        "0 0.5 0.5 0.4 0.4\n0 0.25 0.25 0.1 0.1\n",
    )
    _write_labels(manual / "images" / "val", "0 0.6 0.6 0.2 0.2\n")
    _write_labels(synthetic / "images" / "train", "0 0.5 0.5 0.3 0.3\n")
    _write_labels(synthetic / "images" / "val", "0 0.7 0.7 0.2 0.2\n")
    _write_labels(external / "fixture" / "images" / "test", "0 0.4 0.4 0.2 0.2\n")
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
    for condition, condition_runs in runs.items():
        _write_training_run(tmp_path / "runs", condition, condition_runs[0])
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
            "counting": {
                "confidence": 0.25,
                "mae": summaries[condition]["count_mae_mean"],
                "per_image": [
                    {
                        "id": "sample-0",
                        "target": 1,
                        "predicted": 1,
                        "error": 0,
                        "absolute_error": 0,
                    }
                ],
            },
        }
        for condition, condition_runs in runs.items()
    ]
    results = {
        "selection_sha256": "a" * 64,
        "external_manifest_sha256": "b" * 64,
        "test_opened_after_selection": True,
        "external_dataset": "fixture",
        "summary": {"yolo": summaries},
        "results": evaluations,
    }
    (artifacts / "test_results_fixture.json").write_text(json.dumps(results))

    destination = generate_markdown_report(experiment, pipeline)
    text = destination.read_text()

    assert destination == artifacts / "RESULTS_fixture.md"
    assert "**synthetic-1x** ★" in text
    assert "superou descritivamente" in text
    assert "## Distribuição espacial das anotações" in text
    assert "2 imagens e 3 caixas (treino + validação)" in text
    assert "annotation_heatmaps/manual-full.png" in text
    assert "## Curvas de treinamento" in text
    assert "training_curves/validation_map50_95_yolo.svg" in text
    assert "[Baixar pacote de análise](analysis_csv.zip)" in text
    with zipfile.ZipFile(artifacts / "analysis_csv.zip") as bundle:
        assert set(bundle.namelist()) == {
            "analysis_csv/annotations.csv",
            "analysis_csv/counting_by_image.csv",
            "analysis_csv/datasets.csv",
            "analysis_csv/protocol.csv",
            "analysis_csv/provenance.csv",
            "analysis_csv/result_summary.csv",
            "analysis_csv/run_metrics.csv",
            "analysis_csv/training_history.csv",
        }
        assert (
            "metrics/mAP50-95(B)"
            in bundle.read("analysis_csv/training_history.csv").decode()
        )
        assert (
            "validation.map50_95"
            in bundle.read("analysis_csv/run_metrics.csv").decode()
        )
        assert (
            "annotation_index" in bundle.read("analysis_csv/annotations.csv").decode()
        )
    for name in ("manual-full", "synthetic-1x", "fixture"):
        with Image.open(artifacts / "annotation_heatmaps" / f"{name}.png") as image:
            assert image.size == (512, 512)
            assert image.getbbox() is not None
