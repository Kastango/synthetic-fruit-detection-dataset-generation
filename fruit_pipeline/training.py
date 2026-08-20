from __future__ import annotations

import csv
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .common import (
    ROOT,
    atomic_write_json,
    atomic_write_text,
    automatic_workers,
    project_path,
    sha256_file,
    stable_hash,
)


def scoped_experiment_root(base: Path, config: dict, kind: str) -> Path:
    subdir = config.get("protocol", {}).get(f"{kind}_subdir")
    return base / str(subdir) if subdir else base


def _manifest_fingerprint(paths: list[Path], allow_missing: bool) -> str:
    values = []
    for path in paths:
        if not path.exists():
            if allow_missing:
                values.append((str(path), "MISSING"))
                continue
            raise FileNotFoundError(f"manifesto do dataset ausente: {path}")
        values.append(
            (
                str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                sha256_file(path),
            )
        )
    return stable_hash(values, 24)


def expand_experiments(config: dict, *, allow_missing: bool = False) -> list[dict]:
    protocol = config["protocol"]
    seeds = [int(seed) for seed in config["seeds"]]
    default_validation = project_path(protocol["validation"])
    specs = []
    for condition_name, condition in sorted(config["conditions"].items()):
        validation = (
            project_path(condition["validation"])
            if "validation" in condition
            else default_validation
        )
        if not allow_missing and not validation.is_dir():
            raise FileNotFoundError(
                f"validação ausente para {condition_name}: {validation}"
            )
        train_paths = [project_path(path) for path in condition["train"]]
        manifests = [project_path(path) for path in condition["manifests"]]
        if not allow_missing:
            missing = [str(path) for path in train_paths if not path.is_dir()]
            if missing:
                raise FileNotFoundError(
                    f"treino ausente para {condition_name}: {missing}"
                )
        dataset_fingerprint = _manifest_fingerprint(manifests, allow_missing)
        for model_config in config["models"]:
            parameters = dict(config["defaults"])
            parameters.update(model_config["parameters"])
            candidate = {
                "condition": condition_name,
                "model": str(model_config["checkpoint"]),
                "model_name": str(model_config["name"]),
                "parameters": parameters,
                "train": [str(path.resolve()) for path in train_paths],
                "validation": str(validation.resolve()),
                "class_names": protocol["class_names"],
                "dataset_fingerprint": dataset_fingerprint,
                "ultralytics_version": str(protocol["ultralytics_version"]),
            }
            candidate_id = stable_hash(candidate, 16)
            for seed in seeds:
                spec = {**candidate, "candidate_id": candidate_id, "seed": seed}
                spec["run_id"] = (
                    f"{condition_name}__{model_config['name']}__s{seed}__"
                    f"{stable_hash(spec, 10)}"
                )
                specs.append(spec)
    return specs


def training_yaml(spec: dict, destination: Path) -> None:
    value = {
        "train": spec["train"],
        "val": spec["validation"],
        "names": {index: name for index, name in enumerate(spec["class_names"])},
    }
    atomic_write_text(
        destination,
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
    )


def _metrics_dict(metrics, class_names: list[str]) -> dict:
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    map75 = getattr(metrics.box, "map75", None)
    result = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "map50": round(float(metrics.box.map50), 6),
        "map75": round(float(map75), 6) if map75 is not None else None,
        "map50_95": round(float(metrics.box.map), 6),
        "per_class": {},
    }
    all_ap = getattr(metrics.box, "all_ap", None)
    if all_ap is not None:
        array = np.asarray(all_ap, dtype=float)
        if array.ndim == 2 and array.shape[1] == 10:
            thresholds = np.arange(0.5, 1.0, 0.05)
            result["ap_by_iou"] = {
                f"{threshold:.2f}": round(float(value), 6)
                for threshold, value in zip(thresholds, array.mean(axis=0))
            }
    px = np.asarray(getattr(metrics.box, "px", []), dtype=float)
    f1_curve = np.asarray(getattr(metrics.box, "f1_curve", []), dtype=float)
    if px.size and f1_curve.ndim == 2 and f1_curve.shape[1] == px.size:
        mean_curve = f1_curve.mean(axis=0)
        index = int(mean_curve.argmax())
        result["best_f1"] = round(float(mean_curve[index]), 6)
        result["best_f1_confidence"] = round(float(px[index]), 6)
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        result["speed_ms_per_image"] = {
            str(key): round(float(value), 6) for key, value in speed.items()
        }
    for class_index, map50 in zip(metrics.box.ap_class_index, metrics.box.ap50):
        name = class_names[int(class_index)]
        result["per_class"][name] = {
            "map50": round(float(map50), 6),
            "map50_95": round(float(metrics.box.maps[int(class_index)]), 6),
        }
    return result


def _count_metrics(
    model,
    test_path: Path,
    *,
    confidence: float,
    imgsz: int,
    device: str,
    max_det: int,
) -> dict:
    label_root = test_path.parent.parent / "labels" / test_path.name
    if not label_root.is_dir():
        raise FileNotFoundError(f"rótulos de contagem ausentes: {label_root}")
    per_image = []
    predictions = model.predict(
        source=str(test_path),
        conf=confidence,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        stream=True,
        verbose=False,
        save=False,
    )
    for prediction in predictions:
        stem = Path(prediction.path).stem
        label = label_root / f"{stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"rótulo externo ausente: {label}")
        target_count = sum(
            bool(line.strip()) for line in label.read_text().splitlines()
        )
        predicted_count = len(prediction.boxes)
        error = predicted_count - target_count
        per_image.append(
            {
                "id": stem,
                "target": target_count,
                "predicted": predicted_count,
                "error": error,
                "absolute_error": abs(error),
            }
        )
    if not per_image:
        raise RuntimeError(f"nenhuma predição de contagem produzida para {test_path}")
    errors = [float(item["error"]) for item in per_image]
    return {
        "confidence": round(confidence, 6),
        "mae": round(statistics.fmean(abs(value) for value in errors), 6),
        "rmse": round(
            math.sqrt(statistics.fmean(value * value for value in errors)), 6
        ),
        "bias": round(statistics.fmean(errors), 6),
        "images": len(per_image),
        "per_image": sorted(per_image, key=lambda item: item["id"]),
    }


def train_one(spec: dict, runs_root: Path, force: bool = False) -> dict:
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("treino requer requirements.txt (Ultralytics)") from error
    expected_version = spec["ultralytics_version"]
    if ultralytics.__version__ != expected_version:
        raise RuntimeError(
            f"protocolo requer ultralytics=={expected_version}; encontrado "
            f"{ultralytics.__version__}"
        )
    run_id = spec["run_id"]
    run_dir = runs_root / "training" / run_id
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    spec_dir = runs_root / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = spec_dir / f"{run_id}.yaml"
    training_yaml(spec, data_yaml)
    atomic_write_json(spec_dir / f"{run_id}.json", spec)
    parameters = dict(spec["parameters"])
    device = parameters.pop("device", "cuda")
    seed = int(spec["seed"])
    parameters.update(
        {
            "seed": seed,
            "data": str(data_yaml),
            "project": str((runs_root / "training").resolve()),
            "name": run_id,
            "exist_ok": True,
            "verbose": False,
            "plots": True,
            "device": device,
        }
    )
    last_checkpoint = run_dir / "weights" / "last.pt"
    state_path = run_dir / "training_state.json"
    training_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists() and not force
        else {"training_seconds": 0.0}
    )
    started = time.monotonic()
    model = YOLO(
        str(last_checkpoint)
        if last_checkpoint.exists() and not force
        else spec["model"]
    )
    input_checkpoint = Path(spec["model"])
    if input_checkpoint.exists() and "input_checkpoint_sha256" not in training_state:
        training_state["input_checkpoint"] = str(input_checkpoint.resolve())
        training_state["input_checkpoint_sha256"] = sha256_file(input_checkpoint)

    def persist_training_state(_trainer=None) -> None:
        nonlocal started
        training_state["training_seconds"] = round(
            float(training_state.get("training_seconds", 0.0))
            + max(0.0, time.monotonic() - started),
            2,
        )
        started = time.monotonic()
        atomic_write_json(state_path, training_state)

    model.add_callback("on_model_save", persist_training_state)
    model.add_callback("on_train_end", persist_training_state)
    if last_checkpoint.exists() and not force:
        print(f"retomando {run_id} a partir de last.pt", flush=True)
        model.train(resume=True, device=device)
    else:
        print(f"treinando {run_id}", flush=True)
        model.train(**parameters)
    persist_training_state()
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"treino terminou sem best.pt: {run_dir}")
    validation_dir = runs_root / "validation"
    metrics = YOLO(str(best)).val(
        data=str(data_yaml),
        split="val",
        device=device,
        project=str(validation_dir.resolve()),
        name=run_id,
        exist_ok=True,
        verbose=False,
        plots=True,
        imgsz=int(spec["parameters"]["imgsz"]),
        batch=int(spec["parameters"].get("batch", 8)),
        workers=int(spec["parameters"].get("workers", automatic_workers())),
        max_det=int(spec["parameters"].get("max_det", 1000)),
    )
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if torch.cuda.is_available():
        hardware["gpu"] = torch.cuda.get_device_name(0)
    result = {
        "run_id": run_id,
        "candidate_id": spec["candidate_id"],
        "condition": spec["condition"],
        "model": spec["model"],
        "model_name": spec.get("model_name", Path(spec["model"]).stem),
        "seed": seed,
        "parameters": spec["parameters"],
        "dataset_fingerprint": spec["dataset_fingerprint"],
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "training_seconds": float(training_state["training_seconds"]),
        "hardware": hardware,
        "checkpoint": str(best.resolve()),
        "checkpoint_sha256": sha256_file(best),
        "input_checkpoint": training_state.get("input_checkpoint", spec["model"]),
        "input_checkpoint_sha256": training_state.get("input_checkpoint_sha256"),
        "validation": _metrics_dict(metrics, spec["class_names"]),
    }
    atomic_write_json(result_path, result)
    return result


def run_grid(
    config: dict,
    *,
    runs_root: Path,
    condition_filters: list[str] | None = None,
    model_filters: list[str] | None = None,
    max_runs: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    continue_on_error: bool = False,
) -> list[dict]:
    specs = expand_experiments(config, allow_missing=dry_run)
    if condition_filters:
        specs = [spec for spec in specs if spec["condition"] in condition_filters]
    if model_filters:
        specs = [
            spec
            for spec in specs
            if spec["model"] in model_filters or spec["model_name"] in model_filters
        ]
    # Evita confundir condição/arquitetura com ordem térmica ou temporal da GPU.
    import random

    random.Random(2027).shuffle(specs)
    if max_runs is not None:
        specs = specs[:max_runs]
    if device:
        for spec in specs:
            spec["parameters"]["device"] = device
    worker_count = max(1, workers) if workers is not None else automatic_workers()
    for spec in specs:
        spec["parameters"]["workers"] = worker_count
    print(f"grade selecionada: {len(specs)} execuções", flush=True)
    if dry_run:
        for spec in specs:
            print(
                f"{spec['run_id']} condition={spec['condition']} model={spec['model']} "
                f"seed={spec['seed']} params={spec['parameters']}"
            )
        return []

    specs_dir = runs_root / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for index, spec in enumerate(specs, 1):
        result_path = runs_root / "training" / spec["run_id"] / "result.json"
        if result_path.exists() and not force:
            print(f"[{index}/{len(specs)}] reutilizando {spec['run_id']}")
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        spec_path = specs_dir / f"{spec['run_id']}.json"
        atomic_write_json(spec_path, spec)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_one.py"),
            "--spec",
            str(spec_path),
            "--runs-root",
            str(runs_root),
        ]
        if force:
            command.append("--force")
        print(f"[{index}/{len(specs)}] {spec['run_id']}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            failures.append(
                {"run_id": spec["run_id"], "returncode": completed.returncode}
            )
            if not continue_on_error:
                raise subprocess.CalledProcessError(completed.returncode, command)
        elif result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
    if failures:
        atomic_write_json(runs_root / "training_failures.json", failures)
        raise RuntimeError(
            f"{len(failures)} treino(s) falharam; veja training_failures.json"
        )
    return results


def _all_training_results(runs_root: Path) -> list[dict]:
    results = []
    for path in sorted((runs_root / "training").glob("*/result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def select_by_validation(config: dict, runs_root: Path, artifacts: Path) -> dict:
    expected_seeds = set(map(int, config["seeds"]))
    expected_specs = expand_experiments(config, allow_missing=True)
    expected_candidate_ids = {spec["candidate_id"] for spec in expected_specs}
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for result in _all_training_results(runs_root):
        if result["candidate_id"] in expected_candidate_ids:
            grouped.setdefault(
                (
                    result.get("model_name", Path(result["model"]).stem),
                    result["condition"],
                    result["candidate_id"],
                ),
                [],
            ).append(result)
    candidates = []
    for (model_name, condition, candidate_id), results in grouped.items():
        seeds = {int(item["seed"]) for item in results}
        if seeds != expected_seeds:
            continue
        values = [float(item["validation"]["map50_95"]) for item in results]
        candidates.append(
            {
                "condition": condition,
                "model_name": model_name,
                "candidate_id": candidate_id,
                "model": results[0]["model"],
                "parameters": results[0]["parameters"],
                "dataset_fingerprint": results[0]["dataset_fingerprint"],
                "validation_map50_95_mean": round(statistics.fmean(values), 6),
                "validation_map50_95_std": round(statistics.stdev(values), 6)
                if len(values) > 1
                else 0.0,
                "runs": sorted(
                    [
                        {
                            "run_id": item["run_id"],
                            "seed": item["seed"],
                            "checkpoint": item["checkpoint"],
                            "checkpoint_sha256": item["checkpoint_sha256"],
                            "validation": item["validation"],
                            "training_seconds": item["training_seconds"],
                        }
                        for item in results
                    ],
                    key=lambda item: item["seed"],
                ),
            }
        )
    selected: dict[str, dict[str, dict]] = {}
    expected_conditions = set(config["conditions"])
    expected_models = sorted({spec["model_name"] for spec in expected_specs})
    for model_name in expected_models:
        selected[model_name] = {}
        for condition in sorted(expected_conditions):
            eligible = [
                item
                for item in candidates
                if item["condition"] == condition and item["model_name"] == model_name
            ]
            if not eligible:
                raise RuntimeError(
                    f"nenhum candidato completo para {model_name}/{condition}; "
                    "todas as sementes são obrigatórias"
                )
            selected[model_name][condition] = max(
                eligible,
                key=lambda item: (
                    item["validation_map50_95_mean"],
                    -item["validation_map50_95_std"],
                    item["candidate_id"],
                ),
            )
    report = {
        "selection_metric": "origin_validation_map50_95_mean",
        "test_was_read": False,
        "required_seeds": sorted(expected_seeds),
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=lambda item: (
                item["model_name"],
                item["condition"],
                item["candidate_id"],
            ),
        ),
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_write_json(artifacts / "model_selection.json", report)
    return report


def evaluate_checkpoint(spec: dict) -> dict:
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("avaliação requer requirements.txt") from error
    if ultralytics.__version__ != spec["ultralytics_version"]:
        raise RuntimeError(
            f"protocolo requer ultralytics=={spec['ultralytics_version']}; "
            f"encontrado {ultralytics.__version__}"
        )
    output = Path(spec["output"])
    if output.exists() and not spec.get("force", False):
        return json.loads(output.read_text(encoding="utf-8"))
    actual_checkpoint_hash = sha256_file(Path(spec["checkpoint"]))
    if actual_checkpoint_hash != spec["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint alterado desde a seleção: {spec['checkpoint']}")
    data_yaml = output.with_suffix(".yaml")
    test_path = str(Path(spec["test"]).resolve())
    value = {
        "train": test_path,
        "val": test_path,
        "test": test_path,
        "names": {index: name for index, name in enumerate(spec["class_names"])},
    }
    atomic_write_text(
        data_yaml, yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    )
    detector = YOLO(spec["checkpoint"])
    metrics = detector.val(
        data=str(data_yaml),
        split="test",
        device=spec["device"],
        project=str(Path(spec["evaluation_root"]).resolve()),
        name=spec["run_id"],
        exist_ok=True,
        verbose=False,
        plots=True,
        imgsz=int(spec["imgsz"]),
        batch=int(spec.get("batch", 8)),
        workers=int(spec.get("workers", automatic_workers())),
        max_det=int(spec.get("max_det", 1000)),
    )
    confidence = float(spec.get("confidence_threshold", 0.25))
    result = {
        "condition": spec["condition"],
        "model_name": spec["model_name"],
        "candidate_id": spec["candidate_id"],
        "run_id": spec["run_id"],
        "seed": spec["seed"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "test": _metrics_dict(metrics, spec["class_names"]),
        "counting": _count_metrics(
            detector,
            Path(spec["test"]),
            confidence=confidence,
            imgsz=int(spec["imgsz"]),
            device=str(spec["device"]),
            max_det=int(spec.get("max_det", 1000)),
        ),
    }
    atomic_write_json(output, result)
    return result


def evaluate_selected(
    config: dict,
    runs_root: Path,
    artifacts: Path,
    *,
    device: str,
    force: bool = False,
) -> dict:
    selection_path = artifacts / "model_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            "selecione os modelos pela validação antes de abrir o teste"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_hash = sha256_file(selection_path)
    external_name = str(config["protocol"].get("external_test", "test"))
    final_path = artifacts / f"test_results_{external_name}.json"
    test_path = project_path(config["protocol"]["test"])
    if not test_path.is_dir():
        raise FileNotFoundError(f"teste real ausente: {test_path}")
    external_manifest = test_path.parents[1] / "manifest.json"
    if not external_manifest.exists():
        raise FileNotFoundError(
            f"manifesto do teste externo ausente: {external_manifest}"
        )
    external_manifest_sha256 = sha256_file(external_manifest)
    if final_path.exists() and not force:
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            existing.get("selection_sha256") != selection_hash
            or existing.get("external_manifest_sha256") != external_manifest_sha256
        ):
            raise RuntimeError(
                "o teste já foi aberto para outra seleção ou outro manifesto; "
                "preserve o resultado ou use --force conscientemente"
            )
        return existing
    evaluations = artifacts / f"test_evaluations_{external_name}"
    specs = artifacts / f"test_specs_{external_name}"
    evaluations.mkdir(parents=True, exist_ok=True)
    specs.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name, by_condition in sorted(selection["selected"].items()):
        for condition, selected in sorted(by_condition.items()):
            for run in selected["runs"]:
                output = evaluations / f"{run['run_id']}.json"
                parameters = selected["parameters"]
                spec = {
                    "condition": condition,
                    "model_name": model_name,
                    "candidate_id": selected["candidate_id"],
                    "run_id": run["run_id"],
                    "seed": run["seed"],
                    "checkpoint": run["checkpoint"],
                    "checkpoint_sha256": run["checkpoint_sha256"],
                    "test": str(test_path),
                    "class_names": config["protocol"]["class_names"],
                    "ultralytics_version": str(
                        config["protocol"]["ultralytics_version"]
                    ),
                    "device": device,
                    "imgsz": int(parameters["imgsz"]),
                    "batch": int(parameters.get("batch", 8)),
                    "workers": int(parameters.get("workers", automatic_workers())),
                    "max_det": int(parameters.get("max_det", 1000)),
                    "confidence_threshold": float(
                        run["validation"].get("best_f1_confidence", 0.25)
                    ),
                    "evaluation_root": str(
                        (runs_root / "test" / external_name).resolve()
                    ),
                    "output": str(output),
                    "force": force,
                }
                spec_path = specs / f"{run['run_id']}.json"
                atomic_write_json(spec_path, spec)
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_one.py"),
                    "--spec",
                    str(spec_path),
                ]
                subprocess.run(command, cwd=ROOT, check=True)
                results.append(json.loads(output.read_text(encoding="utf-8")))
    summaries: dict[str, dict[str, dict]] = {}
    for model_name, by_condition in selection["selected"].items():
        summaries[model_name] = {}
        for condition in by_condition:
            matching = [
                item
                for item in results
                if item["condition"] == condition and item["model_name"] == model_name
            ]
            metric_summary = {}
            for metric in ("precision", "recall", "f1", "map50", "map75", "map50_95"):
                values = [
                    float(item["test"][metric])
                    for item in matching
                    if item["test"].get(metric) is not None
                ]
                if values:
                    metric_summary[f"{metric}_mean"] = round(
                        statistics.fmean(values), 6
                    )
                    metric_summary[f"{metric}_std"] = (
                        round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
                    )
            for metric in ("mae", "rmse", "bias"):
                values = [float(item["counting"][metric]) for item in matching]
                metric_summary[f"count_{metric}_mean"] = round(
                    statistics.fmean(values), 6
                )
                metric_summary[f"count_{metric}_std"] = (
                    round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
                )
            metric_summary["runs"] = len(matching)
            summaries[model_name][condition] = metric_summary
        baseline = summaries[model_name].get("manual-full", {}).get("map50_95_mean")
        for condition_summary in summaries[model_name].values():
            condition_summary["delta_vs_manual_full"] = (
                round(condition_summary["map50_95_mean"] - baseline, 6)
                if baseline is not None
                else None
            )
    final = {
        "selection_sha256": selection_hash,
        "test_opened_after_selection": True,
        "external_dataset": external_name,
        "external_manifest_sha256": external_manifest_sha256,
        "test_path": str(test_path),
        "summary": summaries,
        "results": results,
    }
    atomic_write_json(final_path, final)
    with (artifacts / f"test_comparison_{external_name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "condition",
                "map50_95_mean",
                "map50_95_std",
                "delta_vs_manual_full",
                "runs",
            ]
        )
        for model_name, by_condition in sorted(summaries.items()):
            for condition, summary in sorted(by_condition.items()):
                writer.writerow(
                    [
                        model_name,
                        condition,
                        summary["map50_95_mean"],
                        summary["map50_95_std"],
                        summary["delta_vs_manual_full"],
                        summary["runs"],
                    ]
                )
    return final
