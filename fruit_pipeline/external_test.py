from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

import yaml
from PIL import Image

from .common import atomic_write_json, sha256_file, stable_hash
from .real_data import detect_records, open_source, validate_yolo_text


def _collapse_to_class_zero(label_text: str) -> str:
    lines = []
    for line in label_text.splitlines():
        if not line.strip():
            continue
        _, *values = line.split()
        lines.append("0 " + " ".join(values))
    return "\n".join(lines) + ("\n" if lines else "")


def import_external_test(
    name: str,
    source: Path,
    target_root: Path,
    class_names: list[str],
    *,
    annotation_format: str = "auto",
    nested_archive: str | None = None,
    collapse_to_single_class: bool = True,
    expected_images: int | None = None,
    expected_boxes: int | None = None,
    source_metadata: dict | None = None,
    force: bool = False,
) -> dict:
    """Importa um conjunto de teste externo (YOLO/COCO/CVAT, detectado
    automaticamente) para `target_root/<name>/`, achatado num único split.

    Não participa de treino/validação, apenas de `evaluate_test.py` contra
    um checkpoint já selecionado. Pensado para ser repetível com outras
    fontes: basta chamar de novo com outro `name`/`source`."""
    target = target_root / name
    manifest_path = target / "manifest.json"
    if target.exists() and not force:
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))["summary"]
        raise FileExistsError(f"destino já existe sem manifesto: {target}; use --force")

    target_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.external-", dir=target_root))
    try:
        images_target = temporary / "images" / "test"
        labels_target = temporary / "labels" / "test"
        images_target.mkdir(parents=True)
        labels_target.mkdir(parents=True)
        with ExitStack() as stack:
            root = stack.enter_context(open_source(source))
            if nested_archive:
                matches = sorted(root.rglob(nested_archive))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"esperado exatamente um {nested_archive!r} em {source}; "
                        f"encontrados {len(matches)}"
                    )
                if not zipfile.is_zipfile(matches[0]):
                    raise ValueError(f"arquivo interno não é ZIP válido: {matches[0]}")
                root = stack.enter_context(open_source(matches[0]))
            detected_format, records = detect_records(root, annotation_format)
            if expected_images is not None and len(records) != expected_images:
                raise ValueError(
                    f"{name} contém {len(records)} imagens; esperado {expected_images}"
                )
            manifest_records = []
            total_boxes = 0
            stems: set[str] = set()
            for input_path, label_text in records:
                stem = input_path.stem.lower()
                if stem in stems:
                    raise ValueError(f"stem duplicado no conjunto externo: {stem}")
                stems.add(stem)
                if collapse_to_single_class:
                    label_text = _collapse_to_class_zero(label_text)
                boxes = validate_yolo_text(label_text, str(input_path))
                total_boxes += boxes
                # Preserva os bytes do teste. Não há resize, recompressão ou
                # transformação que possa alterar a geometria das caixas.
                with Image.open(input_path) as opened:
                    opened.verify()
                suffix = input_path.suffix.lower()
                image_output = images_target / f"{stem}{suffix}"
                shutil.copy2(input_path, image_output)
                label_output = labels_target / f"{stem}.txt"
                label_output.write_text(label_text, encoding="utf-8")
                manifest_records.append(
                    {
                        "id": stem,
                        "original_name": input_path.name,
                        "image": f"images/test/{image_output.name}",
                        "label": f"labels/test/{label_output.name}",
                        "boxes": boxes,
                        "image_sha256": sha256_file(image_output),
                    }
                )
        if not manifest_records:
            raise FileNotFoundError(f"nenhuma imagem encontrada em {source}")
        if expected_boxes is not None and total_boxes != expected_boxes:
            raise ValueError(
                f"{name} contém {total_boxes} caixas; esperado {expected_boxes}"
            )
        manifest_records.sort(key=lambda item: item["id"])
        summary = {
            "name": name,
            "source_format": detected_format,
            "nested_archive": nested_archive,
            "collapsed_to_single_class": collapse_to_single_class,
            "images": len(manifest_records),
            "boxes": total_boxes,
            "records_hash": stable_hash(manifest_records, 24),
            "source": {
                "path": str(source),
                "bytes": source.stat().st_size if source.is_file() else None,
                "sha256": sha256_file(source) if source.is_file() else None,
                **(source_metadata or {}),
            },
        }
        data_yaml = {
            "path": str(temporary.resolve()),
            "test": "images/test",
            "names": {
                index: class_name for index, class_name in enumerate(class_names)
            },
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        atomic_write_json(
            temporary / "manifest.json",
            {"summary": summary, "records": manifest_records},
        )
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
        data_yaml["path"] = str(target.resolve())
        (target / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary
