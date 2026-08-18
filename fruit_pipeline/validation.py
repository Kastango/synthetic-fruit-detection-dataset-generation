from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .common import atomic_write_json, image_files, project_path, sha256_file
from .real_data import validate_yolo_text
from .synthesis import find_depth_map


def validate_yolo_tree(
    root: Path, split_names: tuple[str, ...]
) -> tuple[dict, list[str]]:
    report: dict[str, Any] = {}
    errors = []
    for split in split_names:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = image_files(image_dir)
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        boxes = 0
        if image_stems != label_stems:
            errors.append(
                f"{root}: stems imagem/rótulo diferentes em {split} "
                f"(sem rótulo={len(image_stems - label_stems)}, sem imagem={len(label_stems - image_stems)})"
            )
        for path in labels:
            try:
                boxes += validate_yolo_text(path.read_text(encoding="utf-8"), str(path))
            except ValueError as error:
                errors.append(str(error))
        report[split] = {"images": len(images), "labels": len(labels), "boxes": boxes}
    return report, errors


def validate_real(config: dict) -> tuple[dict, list[str]]:
    errors = []
    source = project_path(config["paths"]["real_source"])
    output = project_path(config["paths"]["real_yolo"])
    artifact = project_path(config["paths"]["artifacts"]) / "real_split.json"
    required = (source / "manifest.json", artifact, output / "data.yaml")
    for path in required:
        if not path.exists():
            errors.append(f"artefato real ausente: {path}")
    if errors:
        return {"ready": False}, errors
    imported = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads(artifact.read_text(encoding="utf-8"))
    expected = config["real_dataset"]
    summary = imported["summary"]
    if summary["images"] != int(expected["expected_images"]):
        errors.append("contagem de imagens reais diferente do protocolo")
    if summary["boxes"] != int(expected["expected_boxes"]):
        errors.append("contagem de caixas reais diferente do protocolo")
    if not summary.get("augmentation_signals", {}).get("passed"):
        errors.append("a fonte real apresenta sinais de resize/augmentation")
    ids = frozen["splits"]
    split_sets = {name: set(values) for name, values in ids.items()}
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        errors.append("vazamento entre splits reais")
    all_ids = {item["id"] for item in imported["records"]}
    if set().union(*split_sets.values()) != all_ids:
        errors.append("split real não cobre exatamente a base importada")
    tree, tree_errors = validate_yolo_tree(output, ("train", "val", "test"))
    errors.extend(tree_errors)
    expected_counts = {
        "train": int(expected["train_images"]),
        "val": int(expected["val_images"]),
        "test": int(expected["test_images"]),
    }
    if {name: tree[name]["images"] for name in expected_counts} != expected_counts:
        errors.append("contagens materializadas 100/15/15 incorretas")
    return {
        "ready": not errors,
        "source_sha256": summary.get("source", {}).get("sha256"),
        "images": summary["images"],
        "boxes": summary["boxes"],
        "devices": summary["devices"],
        "splits": tree,
        "split_devices": frozen["summary"]["devices"],
        "augmentation_signals": summary["augmentation_signals"],
    }, errors


def validate_assets(config: dict, asset_root: Path) -> tuple[dict, list[str]]:
    errors = []
    backgrounds = image_files(asset_root / "backgrounds")
    depth_maps = image_files(asset_root / "backgrounds_map")
    cutouts = image_files(asset_root / "pictures_trimmed")
    paired = []
    mismatched_sizes = []
    for background in backgrounds:
        depth = find_depth_map(background, asset_root / "backgrounds_map")
        if depth is None:
            errors.append(f"mapa ausente para {background.name}")
            continue
        paired.append(depth)
        try:
            with Image.open(background) as image, Image.open(depth) as depth_image:
                if (
                    image.size != depth_image.size
                    and image.size != depth_image.size[::-1]
                ):
                    mismatched_sizes.append(
                        (background.name, image.size, depth_image.size)
                    )
        except OSError as error:
            errors.append(str(error))
    if not backgrounds:
        errors.append(f"nenhum fundo em {asset_root}")
    if not cutouts:
        errors.append(f"nenhum recorte em {asset_root}")
    invalid_cutouts = []
    for path in cutouts:
        try:
            with Image.open(path) as image:
                if (
                    "A" not in image.getbands()
                    or image.getchannel("A").getbbox() is None
                ):
                    invalid_cutouts.append(path.name)
        except OSError:
            invalid_cutouts.append(path.name)
    if invalid_cutouts:
        errors.append(f"recortes sem alpha válido: {invalid_cutouts[:10]}")
    if mismatched_sizes:
        errors.append(f"fundo/mapa com dimensões incompatíveis: {mismatched_sizes[:5]}")
    split_path = asset_root / "asset_split.json"
    split_report = None
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_backgrounds = {
            item["image"] for item in split["splits"]["train"]["backgrounds"]
        }
        val_backgrounds = {
            item["image"] for item in split["splits"]["val"]["backgrounds"]
        }
        train_cutouts = set(split["splits"]["train"]["cutouts"])
        val_cutouts = set(split["splits"]["val"]["cutouts"])
        if train_backgrounds & val_backgrounds or train_cutouts & val_cutouts:
            errors.append("vazamento de ativos entre treino e validação sintéticos")
        split_report = {
            "fingerprint": split["source_fingerprint"],
            "backgrounds": [len(train_backgrounds), len(val_backgrounds)],
            "cutouts": [len(train_cutouts), len(val_cutouts)],
        }
    else:
        errors.append(f"split de ativos ausente: {split_path}")
    return {
        "ready": not errors,
        "root": str(asset_root),
        "backgrounds": len(backgrounds),
        "paired_depth_maps": len(paired),
        "depth_maps": len(depth_maps),
        "cutouts": len(cutouts),
        "split": split_report,
    }, errors


def validate_generated(config: dict) -> tuple[dict, list[str]]:
    generated_root = project_path(config["paths"]["generated"])
    errors = []
    datasets = {}
    if not generated_root.exists():
        return {}, [f"nenhum dataset sintético em {generated_root}"]
    for directory in sorted(path for path in generated_root.iterdir() if path.is_dir()):
        required = (
            directory / "manifest.jsonl",
            directory / "summary.json",
            directory / "data.yaml",
        )
        missing = [path.name for path in required if not path.exists()]
        if missing:
            errors.append(f"{directory.name}: artefatos ausentes {missing}")
            continue
        tree, tree_errors = validate_yolo_tree(directory, ("train", "val"))
        errors.extend(tree_errors)
        records = [
            json.loads(line)
            for line in (directory / "manifest.jsonl").read_text().splitlines()
            if line
        ]
        record_counts = Counter(item["split"] for item in records)
        if any(
            tree[name]["images"] != record_counts[name] for name in ("train", "val")
        ):
            errors.append(f"{directory.name}: manifesto não corresponde às imagens")
        datasets[directory.name] = {
            "splits": tree,
            "manifest_records": len(records),
            "manifest_sha256": sha256_file(directory / "manifest.jsonl"),
        }
    if not datasets:
        errors.append("nenhum dataset sintético completo")
    return datasets, errors


def validate_project(
    config: dict, stage: str = "all", asset_root: Path | None = None
) -> dict:
    report: dict[str, Any] = {"stage": stage, "errors": []}
    if stage in {"all", "real"}:
        report["real"], errors = validate_real(config)
        report["errors"].extend(errors)
    if stage in {"all", "assets"}:
        chosen_root = asset_root or project_path(config["paths"]["assets"])
        report["assets"], errors = validate_assets(config, chosen_root)
        report["errors"].extend(errors)
    if stage in {"all", "generated"}:
        report["generated"], errors = validate_generated(config)
        report["errors"].extend(errors)
    report["ready"] = not report["errors"]
    artifact = project_path(config["paths"]["artifacts"]) / "project_status.json"
    atomic_write_json(artifact, report)
    return report
