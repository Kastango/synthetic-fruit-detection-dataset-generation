from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image, ImageOps

from .common import (
    atomic_write_json,
    image_files,
    project_path,
    sha256_file,
    stable_hash,
)


def normalize_device(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if "iphone" in normalized or "apple" in normalized:
        return "iphone"
    if "pixel" in normalized or "google" in normalized:
        return "pixel"
    return "unknown"


def read_device_metadata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"filename", "device"} <= set(rows[0]):
        raise ValueError("metadata deve ter as colunas filename,device")
    result = {}
    for row in rows:
        name = Path(row["filename"]).name.lower()
        device = normalize_device(row["device"])
        if device == "unknown":
            raise ValueError(f"dispositivo inválido em metadata: {row}")
        if name in result and result[name] != device:
            raise ValueError(f"dispositivo conflitante para {name}")
        result[name] = device
    return result


def infer_device(path: Path, metadata: dict[str, str]) -> tuple[str, str]:
    name = path.name.lower()
    if name in metadata:
        return metadata[name], "metadata"
    from_path = normalize_device("/".join(part.lower() for part in path.parts))
    if from_path != "unknown":
        return from_path, "path"
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            make = str(exif.get(271, ""))
            model = str(exif.get(272, ""))
        from_exif = normalize_device(f"{make} {model}")
        if from_exif != "unknown":
            return from_exif, "exif"
    except (OSError, ValueError):
        pass
    return "unknown", "unavailable"


@contextlib.contextmanager
def open_source(path: Path) -> Iterator[Path]:
    if path.is_dir():
        yield path
        return
    if not zipfile.is_zipfile(path):
        raise ValueError(f"fonte deve ser diretório ou ZIP: {path}")
    temporary = Path(tempfile.mkdtemp(prefix="fruit-real-import-"))
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or (mode & 0o170000) == 0o120000
                ):
                    raise ValueError(f"entrada insegura no ZIP: {info.filename}")
            archive.extractall(temporary)
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def validate_yolo_text(text: str, source: str) -> int:
    boxes = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"rótulo inválido {source}:{line_number}: esperado 5 campos"
            )
        class_id, *values = fields
        if class_id != "0":
            raise ValueError(f"classe diferente de 0 em {source}:{line_number}")
        try:
            x, y, width, height = map(float, values)
        except ValueError as error:
            raise ValueError(
                f"coordenada não numérica em {source}:{line_number}"
            ) from error
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"coordenada fora do intervalo em {source}:{line_number}")
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            raise ValueError(
                f"caixa horizontal fora da imagem em {source}:{line_number}"
            )
        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(f"caixa vertical fora da imagem em {source}:{line_number}")
        boxes += 1
    return boxes


def _label_map(root: Path) -> dict[str, Path]:
    labels: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if path.name.lower() in {
            "classes.txt",
            "obj.names",
            "train.txt",
            "valid.txt",
            "test.txt",
        }:
            continue
        if path.stem.lower() in labels:
            raise ValueError(f"mais de um rótulo para o stem {path.stem!r}")
        labels[path.stem.lower()] = path
    return labels


def load_yolo_records(root: Path) -> list[tuple[Path, str]]:
    labels = _label_map(root)
    records = []
    stems: set[str] = set()
    for image in image_files(root):
        stem = image.stem.lower()
        if stem in stems:
            raise ValueError(f"nomes de imagem repetidos (stem): {image.stem}")
        stems.add(stem)
        label = labels.get(stem)
        if label is None:
            raise FileNotFoundError(f"rótulo YOLO ausente para {image}")
        records.append((image, label.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(f"nenhuma imagem encontrada em {root}")
    return records


def _find_coco_json(root: Path) -> Path | None:
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and {"images", "annotations"} <= set(value):
            return path
    return None


def load_coco_records(root: Path, annotation_path: Path) -> list[tuple[Path, str]]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_name = {path.name: path for path in image_files(root)}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for item in coco["annotations"]:
        annotations[int(item["image_id"])].append(item)
    records = []
    for item in coco["images"]:
        path = by_name.get(Path(item["file_name"]).name)
        if path is None:
            raise FileNotFoundError(f"imagem COCO ausente: {item['file_name']}")
        width, height = float(item["width"]), float(item["height"])
        lines = []
        for annotation in annotations[int(item["id"])]:
            x, y, box_width, box_height = map(float, annotation["bbox"])
            lines.append(
                f"0 {(x + box_width / 2) / width:.8f} "
                f"{(y + box_height / 2) / height:.8f} "
                f"{box_width / width:.8f} {box_height / height:.8f}"
            )
        records.append((path, "\n".join(lines) + ("\n" if lines else "")))
    return records


def _find_cvat_xml(root: Path) -> Path | None:
    for path in sorted(root.rglob("*.xml")):
        try:
            if ET.parse(path).getroot().find("image") is not None:
                return path
        except ET.ParseError:
            continue
    return None


def load_cvat_records(root: Path, annotation_path: Path) -> list[tuple[Path, str]]:
    by_name = {path.name: path for path in image_files(root)}
    records = []
    for image_node in ET.parse(annotation_path).getroot().iterfind("image"):
        name = Path(str(image_node.attrib["name"])).name
        path = by_name.get(name)
        if path is None:
            raise FileNotFoundError(f"imagem CVAT ausente: {name}")
        width = float(image_node.attrib["width"])
        height = float(image_node.attrib["height"])
        lines = []
        for box in image_node.iterfind("box"):
            x1, y1 = float(box.attrib["xtl"]), float(box.attrib["ytl"])
            x2, y2 = float(box.attrib["xbr"]), float(box.attrib["ybr"])
            lines.append(
                f"0 {((x1 + x2) / 2) / width:.8f} {((y1 + y2) / 2) / height:.8f} "
                f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
            )
        records.append((path, "\n".join(lines) + ("\n" if lines else "")))
    return records


def detect_records(
    root: Path, annotation_format: str
) -> tuple[str, list[tuple[Path, str]]]:
    if annotation_format in {"auto", "coco"}:
        coco = _find_coco_json(root)
        if coco:
            return "coco", load_coco_records(root, coco)
        if annotation_format == "coco":
            raise FileNotFoundError("annotations COCO JSON não encontradas")
    if annotation_format in {"auto", "cvat"}:
        cvat = _find_cvat_xml(root)
        if cvat:
            return "cvat", load_cvat_records(root, cvat)
        if annotation_format == "cvat":
            raise FileNotFoundError("annotations.xml do CVAT não encontrado")
    return "yolo", load_yolo_records(root)


def import_real_dataset(
    source: Path,
    target: Path,
    *,
    metadata_path: Path | None,
    annotation_format: str,
    expected_images: int | None,
    expected_boxes: int | None,
    allow_unknown_device: bool,
    allow_processed_input: bool,
    force: bool,
) -> dict:
    if target.exists() and not force:
        raise FileExistsError(f"dataset importado já existe: {target}; use --force")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = read_device_metadata(metadata_path)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.import-", dir=target.parent)
    )
    try:
        images_target = temporary / "images"
        labels_target = temporary / "labels"
        images_target.mkdir(parents=True)
        labels_target.mkdir(parents=True)
        manifest_records = []
        with open_source(source) as root:
            detected_format, records = detect_records(root, annotation_format)
            if expected_images is not None and len(records) != expected_images:
                raise ValueError(
                    f"base real contém {len(records)} imagens; esperado {expected_images}"
                )
            stems: set[str] = set()
            total_boxes = 0
            raw_hashes: dict[str, str] = {}
            raw_dimensions = Counter()
            canonical_dimensions = Counter()
            processed_name_signals = []
            for input_path, label_text in records:
                stem = input_path.stem.lower()
                if stem in stems:
                    raise ValueError(f"stem duplicado: {input_path.stem}")
                stems.add(stem)
                boxes = validate_yolo_text(label_text, str(input_path))
                total_boxes += boxes
                device, device_source = infer_device(input_path, metadata)
                if device == "unknown" and not allow_unknown_device:
                    raise ValueError(
                        f"não foi possível identificar iPhone/Pixel para {input_path.name}; "
                        "forneça --metadata-csv filename,device"
                    )
                raw_hash = sha256_file(input_path)
                if raw_hash in raw_hashes:
                    raise ValueError(
                        f"imagens exatamente duplicadas: {raw_hashes[raw_hash]} e {input_path.name}"
                    )
                raw_hashes[raw_hash] = input_path.name
                if ".rf." in input_path.name.lower() or re.search(
                    r"(?:aug|flip|mosaic|brightness|exposure)",
                    input_path.name,
                    re.IGNORECASE,
                ):
                    processed_name_signals.append(input_path.name)
                with Image.open(input_path) as opened:
                    raw_size = opened.size
                    exif = opened.getexif()
                    exif_make = str(exif.get(271, ""))
                    exif_model = str(exif.get(272, ""))
                    exif_orientation = int(exif.get(274, 1) or 1)
                    canonical = ImageOps.exif_transpose(opened).convert("RGB")
                raw_dimensions[raw_size] += 1
                canonical_dimensions[canonical.size] += 1
                image_output = images_target / f"{stem}.jpg"
                label_output = labels_target / f"{stem}.txt"
                # Normalização determinística, não augmentation: materializa a orientação
                # EXIF usada na anotação e evita comportamento diferente entre loaders.
                canonical.save(
                    image_output,
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                    optimize=False,
                )
                label_output.write_text(label_text, encoding="utf-8")
                manifest_records.append(
                    {
                        "id": stem,
                        "original_name": input_path.name,
                        "image": f"images/{image_output.name}",
                        "label": f"labels/{label_output.name}",
                        "device": device,
                        "device_source": device_source,
                        "boxes": boxes,
                        "raw_size": list(raw_size),
                        "canonical_size": list(canonical.size),
                        "exif": {
                            "make": exif_make,
                            "model": exif_model,
                            "orientation": exif_orientation,
                        },
                        "raw_image_sha256": raw_hash,
                        "image_sha256": sha256_file(image_output),
                        "label_sha256": sha256_file(label_output),
                    }
                )
        if expected_boxes is not None and total_boxes != expected_boxes:
            raise ValueError(
                f"base real contém {total_boxes} caixas; esperado {expected_boxes}"
            )
        if not allow_processed_input:
            too_small = [
                item["original_name"]
                for item in manifest_records
                if min(item["raw_size"]) < 3000 or max(item["raw_size"]) < 4000
            ]
            if processed_name_signals or too_small:
                raise ValueError(
                    "a fonte parece redimensionada/aumentada; use somente os originais "
                    f"(nomes suspeitos={len(processed_name_signals)}, "
                    f"resolução abaixo da câmera={len(too_small)})"
                )
        manifest_records.sort(key=lambda item: item["id"])
        summary = {
            "format": detected_format,
            "images": len(manifest_records),
            "boxes": total_boxes,
            "devices": dict(Counter(item["device"] for item in manifest_records)),
            "raw_dimensions": {
                f"{width}x{height}": count
                for (width, height), count in sorted(raw_dimensions.items())
            },
            "canonical_dimensions": {
                f"{width}x{height}": count
                for (width, height), count in sorted(canonical_dimensions.items())
            },
            "augmentation_signals": {
                "processed_filenames": len(processed_name_signals),
                "exact_duplicates": 0,
                "images_below_original_camera_resolution": sum(
                    min(item["raw_size"]) < 3000 or max(item["raw_size"]) < 4000
                    for item in manifest_records
                ),
                "passed": not processed_name_signals
                and all(
                    min(item["raw_size"]) >= 3000 and max(item["raw_size"]) >= 4000
                    for item in manifest_records
                ),
            },
            "records_hash": stable_hash(manifest_records, 24),
        }
        summary["source"] = {
            "path": str(source),
            "bytes": source.stat().st_size if source.is_file() else None,
            "sha256": sha256_file(source) if source.is_file() else None,
        }
        atomic_write_json(
            temporary / "manifest.json",
            {"summary": summary, "records": manifest_records},
        )
        with (temporary / "metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "device", "source"])
            for item in manifest_records:
                writer.writerow(
                    [Path(item["image"]).name, item["device"], item["device_source"]]
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def proportional_allocation(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if target < 0 or target > total:
        raise ValueError(f"alocação inválida: {target}/{total}")
    exact = {name: target * size / total for name, size in group_sizes.items()}
    result = {
        name: min(size, math.floor(exact[name])) for name, size in group_sizes.items()
    }
    remaining = target - sum(result.values())
    order = sorted(
        group_sizes,
        key=lambda name: (
            exact[name] - math.floor(exact[name]),
            group_sizes[name],
            name,
        ),
        reverse=True,
    )
    for name in order:
        if remaining == 0:
            break
        if result[name] < group_sizes[name]:
            result[name] += 1
            remaining -= 1
    if remaining:
        raise ValueError("não foi possível completar a alocação estratificada")
    return result


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def split_real_dataset(config: dict, force: bool = False) -> dict:
    source = project_path(config["paths"]["real_source"])
    output = project_path(config["paths"]["real_yolo"])
    artifact = project_path(config["paths"]["artifacts"]) / "real_split.json"
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"base real importada ausente: {manifest_path}; execute import_real_dataset.py"
        )
    imported = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = imported["records"]
    real_config = config["real_dataset"]
    seed = int(config["project"]["seed"])
    fingerprint = imported["summary"]["records_hash"]
    if artifact.exists() and output.exists() and not force:
        frozen = json.loads(artifact.read_text(encoding="utf-8"))
        if frozen.get("source_records_hash") != fingerprint:
            raise RuntimeError("o split congelado pertence a outra versão da base real")
        return frozen["summary"]

    by_device: dict[str, list[dict]] = defaultdict(list)
    for item in records:
        by_device[item["device"]].append(item)
    group_sizes = {name: len(items) for name, items in by_device.items()}
    val_alloc = proportional_allocation(group_sizes, int(real_config["val_images"]))
    test_alloc = proportional_allocation(group_sizes, int(real_config["test_images"]))
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for device, items in sorted(by_device.items()):
        shuffled = sorted(items, key=lambda item: item["id"])
        random.Random(seed + int(stable_hash(device, 8), 16)).shuffle(shuffled)
        val_count = val_alloc[device]
        test_count = test_alloc[device]
        splits["val"].extend(shuffled[:val_count])
        splits["test"].extend(shuffled[val_count : val_count + test_count])
        splits["train"].extend(shuffled[val_count + test_count :])
    expected = {
        "train": int(real_config["train_images"]),
        "val": int(real_config["val_images"]),
        "test": int(real_config["test_images"]),
    }
    counts = {name: len(items) for name, items in splits.items()}
    if counts != expected:
        raise ValueError(f"contagens do split {counts} diferem do protocolo {expected}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.split-", dir=output.parent)
    )
    try:
        for split_name, items in splits.items():
            for item in sorted(items, key=lambda value: value["id"]):
                image_source = source / item["image"]
                label_source = source / item["label"]
                _link_or_copy(
                    image_source, temporary / "images" / split_name / image_source.name
                )
                _link_or_copy(
                    label_source, temporary / "labels" / split_name / label_source.name
                )
        data_yaml = {
            "path": str(temporary.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: config["project"]["class_names"][0]},
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
        # O caminho absoluto precisa apontar para o destino final, não o temporário.
        data_yaml["path"] = str(output.resolve())
        (output / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    split_ids = {
        name: sorted(item["id"] for item in items) for name, items in splits.items()
    }
    summary = {
        "counts": counts,
        "devices": {
            name: dict(Counter(item["device"] for item in items))
            for name, items in splits.items()
        },
        "boxes": {
            name: sum(item["boxes"] for item in items) for name, items in splits.items()
        },
    }
    frozen = {
        "protocol_version": 1,
        "seed": seed,
        "source_records_hash": fingerprint,
        "summary": summary,
        "splits": split_ids,
    }
    atomic_write_json(artifact, frozen)
    return summary
