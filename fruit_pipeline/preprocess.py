from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .common import (
    atomic_write_json,
    image_files,
    project_path,
    relative_or_absolute,
    sha256_file,
)

FRUIT_BBOX_MANIFEST = "fruit_bboxes.json"


def _save_image_atomic(image: Image.Image, path: Path, **save_options: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    image.save(temporary, **save_options)
    temporary.replace(path)


def normalize_backgrounds(
    source: Path, target: Path, force: bool = False
) -> list[dict]:
    records = []
    for input_path in image_files(source):
        output_path = target / f"{input_path.stem}.jpg"
        if output_path.exists() and not force:
            records.append(
                {"input": str(input_path), "output": str(output_path), "reused": True}
            )
            continue
        with Image.open(input_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            _save_image_atomic(
                image, output_path, format="JPEG", quality=95, subsampling=0
            )
        records.append(
            {
                "input": relative_or_absolute(input_path),
                "output": relative_or_absolute(output_path),
                "sha256": sha256_file(output_path),
                "size": list(image.size),
            }
        )
    if not records:
        raise FileNotFoundError(f"nenhum fundo encontrado em {source}")
    return records


def segment_fruits(
    source: Path,
    target: Path,
    segmentation_config: dict,
    force: bool = False,
) -> list[dict]:
    try:
        from rembg import new_session, remove
    except ImportError as error:
        raise RuntimeError(
            "pré-processamento requer requirements-preprocess.txt (módulo rembg)"
        ) from error

    model_name = str(segmentation_config["model"])
    threshold = int(segmentation_config["alpha_threshold"])
    output_size = int(segmentation_config["output_size"])
    bbox_path = target.parent / FRUIT_BBOX_MANIFEST
    bboxes: dict[str, dict] = (
        json.loads(bbox_path.read_text(encoding="utf-8")) if bbox_path.exists() else {}
    )
    session = None
    records = []
    for input_path in image_files(source):
        stem = input_path.stem.lower()
        output_path = target / f"{stem}-trimmed.png"
        if output_path.exists() and not force and stem in bboxes:
            records.append(
                {"input": str(input_path), "output": str(output_path), "reused": True}
            )
            continue
        session = session or new_session(model_name)
        with Image.open(input_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            result = remove(image, session=session).convert("RGBA")
        alpha = np.asarray(result.getchannel("A"))
        ys, xs = np.nonzero(alpha > threshold)
        if len(xs) == 0:
            raise RuntimeError(f"segmentação vazia: {input_path}")
        # Bbox no referencial da foto bruta (após exif_transpose), preservada
        # para materializar a condição de treino "controlled" a partir das
        # fotos originais em vez do recorte já cortado abaixo.
        bboxes[stem] = {
            "bbox": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ],
            "image_size": list(image.size),
        }
        result = result.crop(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        )
        scale = min(output_size / result.width, output_size / result.height, 1.0)
        if scale < 1.0:
            result = result.resize(
                (
                    max(1, round(result.width * scale)),
                    max(1, round(result.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        _save_image_atomic(result, output_path, format="PNG", compress_level=6)
        records.append(
            {
                "input": relative_or_absolute(input_path),
                "output": relative_or_absolute(output_path),
                "sha256": sha256_file(output_path),
                "size": list(result.size),
            }
        )
    if not records:
        raise FileNotFoundError(f"nenhuma foto de fruta encontrada em {source}")
    atomic_write_json(bbox_path, bboxes)
    return records


class DepthEstimator:
    def __init__(self, depth_config: dict, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                "mapas de profundidade requerem requirements-preprocess.txt"
            ) from error
        self.torch = torch
        if device is None or device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device
        self.config = depth_config
        model_name = str(depth_config["model"])
        revision = str(depth_config["revision"])
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name, revision=revision
        ).to(device)
        self.model.eval()

    def infer(self, image: Image.Image) -> Image.Image:
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        # post_process_depth_estimation aplica a semântica oficial de cada
        # modelo (remoção de padding, calibração de escala a partir do campo
        # de visão previsto pelo DepthPro, etc.); interpolar diretamente o
        # tensor bruto ignoraria essa calibração e pode gerar profundidade
        # incorreta para modelos que não são apenas "redimensionar".
        processed = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )
        depth = processed[0]["predicted_depth"].detach().float().cpu().numpy()
        finite = np.isfinite(depth)
        if not finite.any():
            raise RuntimeError("o modelo produziu um mapa de profundidade não finito")
        low = float(np.percentile(depth[finite], self.config["lower_percentile"]))
        high = float(np.percentile(depth[finite], self.config["upper_percentile"]))
        if not high > low:
            raise RuntimeError(
                f"mapa de profundidade degenerado: low={low}, high={high}"
            )
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        proximity = (
            1.0 - normalized
            if bool(self.config.get("prediction_is_distance", True))
            else normalized
        )
        if not bool(self.config.get("closest_is_white", True)):
            proximity = 1.0 - proximity
        gray = np.where(finite, np.rint(proximity * 255), 0).astype(np.uint8)
        return Image.fromarray(gray)


def generate_depth_maps(
    backgrounds: Path,
    target: Path,
    depth_config: dict,
    device: str | None = None,
    force: bool = False,
) -> list[dict]:
    pending = [
        path
        for path in image_files(backgrounds)
        if force or not (target / f"{path.stem}_depth.png").exists()
    ]
    estimator = DepthEstimator(depth_config, device=device) if pending else None
    records = []
    for index, input_path in enumerate(image_files(backgrounds), 1):
        output_path = target / f"{input_path.stem}_depth.png"
        if output_path.exists() and not force:
            records.append(
                {"input": str(input_path), "output": str(output_path), "reused": True}
            )
            continue
        with Image.open(input_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            depth = estimator.infer(image)  # type: ignore[union-attr]
        _save_image_atomic(depth, output_path, format="PNG", compress_level=6)
        records.append(
            {
                "input": relative_or_absolute(input_path),
                "output": relative_or_absolute(output_path),
                "sha256": sha256_file(output_path),
                "range": [int(np.asarray(depth).min()), int(np.asarray(depth).max())],
            }
        )
        print(
            f"profundidade {index}/{len(image_files(backgrounds))}: {input_path.name}"
        )
    if not records:
        raise FileNotFoundError(f"nenhum fundo normalizado em {backgrounds}")
    return records


def preprocess_assets(
    config: dict,
    stage: str = "all",
    device: str | None = None,
    force: bool = False,
) -> dict:
    raw = project_path(config["paths"]["raw"])
    target = project_path(config["paths"]["regenerated_assets"])
    target.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "stage": stage,
        "target": relative_or_absolute(target),
        "depth_model": config["depth"],
        "segmentation_model": config["segmentation"],
    }
    if stage in {"all", "normalize"}:
        report["backgrounds"] = normalize_backgrounds(
            raw / "backgrounds", target / "backgrounds", force=force
        )
    if stage in {"all", "segment"}:
        report["cutouts"] = segment_fruits(
            raw / "fruits",
            target / "pictures_trimmed",
            config["segmentation"],
            force=force,
        )
    if stage in {"all", "depth"}:
        report["depth_maps"] = generate_depth_maps(
            target / "backgrounds",
            target / "backgrounds_map",
            config["depth"],
            device=device,
            force=force,
        )
    summary = {
        "complete": all(
            (target / name).is_dir()
            for name in ("backgrounds", "backgrounds_map", "pictures_trimmed")
        ),
        "counts": {
            name: len(image_files(target / name))
            for name in ("backgrounds", "backgrounds_map", "pictures_trimmed")
        },
        "configuration": {
            "depth": config["depth"],
            "segmentation": config["segmentation"],
        },
    }
    atomic_write_json(
        target / "preprocess_manifest.json", {"summary": summary, "files": report}
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
