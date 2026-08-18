from __future__ import annotations

import csv
import itertools
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .common import (
    ROOT,
    atomic_write_json,
    atomic_write_text,
    project_path,
    sha256_file,
    stable_hash,
)


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
    models = list(config["models"])
    seeds = [int(seed) for seed in config["seeds"]]
    grid = config.get("grid", {})
    grid_names = sorted(grid)
    grid_values = [grid[name] for name in grid_names]
    combinations = list(itertools.product(*grid_values)) if grid_names else [()]
    validation = project_path(protocol["validation"])
    if not allow_missing and not validation.is_dir():
        raise FileNotFoundError(f"validação real ausente: {validation}")
    specs = []
    for condition_name, condition in sorted(config["conditions"].items()):
        train_paths = [project_path(path) for path in condition["train"]]
        manifests = [project_path(path) for path in condition["manifests"]]
        if not allow_missing:
            missing = [str(path) for path in train_paths if not path.is_dir()]
            if missing:
                raise FileNotFoundError(
                    f"treino ausente para {condition_name}: {missing}"
                )
        dataset_fingerprint = _manifest_fingerprint(manifests, allow_missing)
        for model in models:
            for combination in combinations:
                parameters = dict(config["defaults"])
                parameters.update(dict(zip(grid_names, combination)))
                candidate = {
                    "condition": condition_name,
                    "model": model,
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
                        f"{condition_name}__{Path(model).stem}__s{seed}__"
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
    result = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "map50": round(float(metrics.box.map50), 6),
        "map50_95": round(float(metrics.box.map), 6),
        "per_class": {},
    }
    for class_index, map50 in zip(metrics.box.ap_class_index, metrics.box.ap50):
        name = class_names[int(class_index)]
        result["per_class"][name] = {
            "map50": round(float(map50), 6),
            "map50_95": round(float(metrics.box.maps[int(class_index)]), 6),
        }
    return result


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
    started = time.monotonic()
    if last_checkpoint.exists() and not force:
        print(f"retomando {run_id} a partir de last.pt", flush=True)
        YOLO(str(last_checkpoint)).train(resume=True, device=device)
    else:
        print(f"treinando {run_id}", flush=True)
        YOLO(spec["model"]).train(**parameters)
    elapsed = time.monotonic() - started
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
    )
    result = {
        "run_id": run_id,
        "candidate_id": spec["candidate_id"],
        "condition": spec["condition"],
        "model": spec["model"],
        "seed": seed,
        "parameters": spec["parameters"],
        "dataset_fingerprint": spec["dataset_fingerprint"],
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "training_seconds": round(elapsed, 2),
        "checkpoint": str(best.resolve()),
        "checkpoint_sha256": sha256_file(best),
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
    dry_run: bool = False,
    force: bool = False,
    continue_on_error: bool = False,
) -> list[dict]:
    specs = expand_experiments(config, allow_missing=dry_run)
    if condition_filters:
        specs = [spec for spec in specs if spec["condition"] in condition_filters]
    if model_filters:
        specs = [spec for spec in specs if spec["model"] in model_filters]
    # Evita confundir condição/arquitetura com ordem térmica ou temporal da GPU.
    import random

    random.Random(2027).shuffle(specs)
    if max_runs is not None:
        specs = specs[:max_runs]
    if device:
        for spec in specs:
            spec["parameters"]["device"] = device
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
    grouped: dict[tuple[str, str], list[dict]] = {}
    for result in _all_training_results(runs_root):
        grouped.setdefault((result["condition"], result["candidate_id"]), []).append(
            result
        )
    candidates = []
    for (condition, candidate_id), results in grouped.items():
        seeds = {int(item["seed"]) for item in results}
        if seeds != expected_seeds:
            continue
        values = [float(item["validation"]["map50_95"]) for item in results]
        candidates.append(
            {
                "condition": condition,
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
                        }
                        for item in results
                    ],
                    key=lambda item: item["seed"],
                ),
            }
        )
    selected = {}
    expected_conditions = set(config["conditions"])
    for condition in sorted(expected_conditions):
        eligible = [item for item in candidates if item["condition"] == condition]
        if not eligible:
            raise RuntimeError(
                f"nenhum candidato completo para {condition}; todas as sementes são obrigatórias"
            )
        selected[condition] = max(
            eligible,
            key=lambda item: (
                item["validation_map50_95_mean"],
                -item["validation_map50_95_std"],
                item["candidate_id"],
            ),
        )
    report = {
        "selection_metric": "real_validation_map50_95_mean",
        "test_was_read": False,
        "required_seeds": sorted(expected_seeds),
        "selected": selected,
        "candidates": sorted(
            candidates, key=lambda item: (item["condition"], item["candidate_id"])
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
    metrics = YOLO(spec["checkpoint"]).val(
        data=str(data_yaml),
        split="test",
        device=spec["device"],
        project=str(Path(spec["evaluation_root"]).resolve()),
        name=spec["run_id"],
        exist_ok=True,
        verbose=False,
        plots=True,
    )
    result = {
        "condition": spec["condition"],
        "candidate_id": spec["candidate_id"],
        "run_id": spec["run_id"],
        "seed": spec["seed"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "test": _metrics_dict(metrics, spec["class_names"]),
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
    final_path = artifacts / "test_results.json"
    if final_path.exists() and not force:
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing.get("selection_sha256") != selection_hash:
            raise RuntimeError(
                "o teste já foi aberto para outra seleção; preserve o resultado ou use --force conscientemente"
            )
        return existing
    test_path = project_path(config["protocol"]["test"])
    if not test_path.is_dir():
        raise FileNotFoundError(f"teste real ausente: {test_path}")
    evaluations = artifacts / "test_evaluations"
    specs = artifacts / "test_specs"
    evaluations.mkdir(parents=True, exist_ok=True)
    specs.mkdir(parents=True, exist_ok=True)
    results = []
    for condition, selected in sorted(selection["selected"].items()):
        for run in selected["runs"]:
            output = evaluations / f"{run['run_id']}.json"
            spec = {
                "condition": condition,
                "candidate_id": selected["candidate_id"],
                "run_id": run["run_id"],
                "seed": run["seed"],
                "checkpoint": run["checkpoint"],
                "checkpoint_sha256": run["checkpoint_sha256"],
                "test": str(test_path),
                "class_names": config["protocol"]["class_names"],
                "ultralytics_version": str(config["protocol"]["ultralytics_version"]),
                "device": device,
                "evaluation_root": str((runs_root / "test").resolve()),
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
    summaries = {}
    for condition in selection["selected"]:
        values = [
            float(item["test"]["map50_95"])
            for item in results
            if item["condition"] == condition
        ]
        summaries[condition] = {
            "map50_95_mean": round(statistics.fmean(values), 6),
            "map50_95_std": round(statistics.stdev(values), 6)
            if len(values) > 1
            else 0.0,
            "runs": len(values),
        }
    baseline = summaries.get("real_baseline", {}).get("map50_95_mean")
    for condition, summary in summaries.items():
        summary["delta_vs_real_baseline"] = (
            round(summary["map50_95_mean"] - baseline, 6)
            if baseline is not None
            else None
        )
    final = {
        "selection_sha256": selection_hash,
        "test_opened_after_selection": True,
        "test_path": str(test_path),
        "summary": summaries,
        "results": results,
    }
    atomic_write_json(final_path, final)
    with (artifacts / "test_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "map50_95_mean",
                "map50_95_std",
                "delta_vs_real_baseline",
                "runs",
            ]
        )
        for condition, summary in sorted(summaries.items()):
            writer.writerow(
                [
                    condition,
                    summary["map50_95_mean"],
                    summary["map50_95_std"],
                    summary["delta_vs_real_baseline"],
                    summary["runs"],
                ]
            )
    return final
