import json

from fruit_pipeline.common import ROOT, load_yaml
from fruit_pipeline.training import expand_experiments, select_by_validation


def test_confirmatory_matrix_has_42_runs_and_50_epoch_limit() -> None:
    config = load_yaml(ROOT / "configs" / "confirmatory.yaml")
    specs = expand_experiments(config, allow_missing=True)
    assert len(specs) == 42
    assert len({spec["candidate_id"] for spec in specs}) == 21
    assert all("test" not in spec for spec in specs)
    assert {spec["model_name"] for spec in specs} == {
        "yolo26s",
        "yolov8s",
        "rtdetr-l",
    }
    assert all(spec["parameters"]["epochs"] == 50 for spec in specs)
    assert all(spec["parameters"]["patience"] == 30 for spec in specs)
    assert all(spec["parameters"]["close_mosaic"] == 5 for spec in specs)
    assert {
        spec["validation"]
        for spec in specs
        if spec["condition"].startswith("synthetic-")
    } == {
        str(
            (
                ROOT
                / "data"
                / "generated"
                / f"synthetic-{multiplier}x"
                / "images"
                / "val"
            ).resolve()
        )
        for multiplier in (1, 2, 3, 5, 10)
    }


def test_confirmatory_selection_preserves_every_model_and_condition(tmp_path) -> None:
    config = load_yaml(ROOT / "configs" / "confirmatory.yaml")
    specs = expand_experiments(config, allow_missing=True)
    runs = tmp_path / "runs"
    for spec in specs:
        result = {
            **{
                key: spec[key]
                for key in (
                    "run_id",
                    "candidate_id",
                    "condition",
                    "model",
                    "model_name",
                    "seed",
                    "parameters",
                    "dataset_fingerprint",
                )
            },
            "training_seconds": 10.0,
            "checkpoint": f"/weights/{spec['run_id']}.pt",
            "checkpoint_sha256": "0" * 64,
            "validation": {"map50_95": 0.5 + spec["seed"] / 1000},
        }
        destination = runs / "training" / spec["run_id"] / "result.json"
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(result))

    report = select_by_validation(config, runs, tmp_path / "artifacts")

    assert set(report["selected"]) == {"yolo26s", "yolov8s", "rtdetr-l"}
    assert all(len(by_condition) == 7 for by_condition in report["selected"].values())
    assert (
        sum(
            len(item["runs"])
            for by_condition in report["selected"].values()
            for item in by_condition.values()
        )
        == 42
    )
