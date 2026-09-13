#!/usr/bin/env python3
"""Mede caixas YOLO, negativos e aparência; exporta dados e mistura por imagem.

Caixas medem extensão anotada, não diâmetro físico nem máscara de fruto.
A mistura dá peso total igual a cada domínio e peso igual a cada imagem.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image

from fruit_pipeline.common import image_files, sha256_file
from fruit_pipeline.real_data import validate_yolo_text

QUANTILES = [5, 10, 25, 50, 75, 90, 95, 99]
BOX_METRICS = [
    "width_norm",
    "height_norm",
    "max_side_norm",
    "long_side_short_image",
    "area_norm",
    "aspect_ratio",
    "width_px",
    "height_px",
    "sqrt_area_960",
    "center_x",
    "center_y",
]
IMAGE_METRICS = [
    "boxes",
    "median_max_side",
    "within_p90_p10",
    "sum_box_area",
    "brightness",
    "contrast",
    "saturation",
]


def describe(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        **{f"p{q}": float(np.percentile(a, q)) for q in QUANTILES},
    }


def weighted_description(values, weights):
    a, w = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(a) & (w > 0)
    a, w = a[valid], w[valid]
    if not len(a):
        return {"n": 0}
    order = np.argsort(a)
    a, w = a[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    return {
        "n": len(a),
        "mean": float(np.average(a, weights=w)),
        **{
            f"p{q}": float(a[min(np.searchsorted(cdf, q / 100), len(a) - 1)])
            for q in QUANTILES
        },
    }


def scan(root, splits, name):
    rows, boxes, errors, hashes, duplicates = [], [], [], {}, []
    digest = hashlib.sha256()
    for split in splits:
        images = image_files(root / "images" / split)
        labels = {p.stem: p for p in (root / "labels" / split).glob("*.txt")}
        stems = [p.stem for p in images]
        if len(set(stems)) != len(stems):
            errors.append(f"{split}: stems duplicados")
        if not images:
            errors.append(f"{split}: nenhuma imagem")
        for orphan in sorted(labels.keys() - set(stems)):
            errors.append(f"{split}/{orphan}: rótulo sem imagem")
        for path in images:
            key = f"{split}/{path.stem}"
            label = labels.get(path.stem)
            if label is None:
                errors.append(f"{key}: rótulo ausente (não é negativo)")
                continue
            content = label.read_text()
            try:
                validate_yolo_text(content, str(label))
                with Image.open(path) as im:
                    im.load()
                    width, height = im.size
                    rgb = im.convert("RGB")
                    rgb.thumbnail((256, 256))
                    gray = np.asarray(rgb.convert("L"), float)
                    sat = np.asarray(rgb.convert("HSV"), float)[..., 1]
            except (ValueError, OSError) as e:
                errors.append(str(e))
                continue
            image_hash = sha256_file(path)
            label_hash = sha256_file(label)
            digest.update(f"{key}:{image_hash}:{label_hash}\n".encode())
            if image_hash in hashes:
                duplicates.append([hashes[image_hash], key])
            hashes[image_hash] = key
            local = []
            annotations = [
                line.split() for line in content.splitlines() if line.strip()
            ]
            duplicate_boxes = len(annotations) - len({tuple(a) for a in annotations})
            edge = 0
            for i, fields in enumerate(annotations):
                x, y, w, h = map(float, fields[1:])
                edge += int(
                    min(x - w / 2, y - h / 2, 1 - x - w / 2, 1 - y - h / 2) <= 1e-5
                )
                b = dict(
                    dataset=name,
                    image=key,
                    index=i,
                    width_norm=w,
                    height_norm=h,
                    max_side_norm=max(w, h),
                    long_side_short_image=max(w * width, h * height)
                    / min(width, height),
                    area_norm=w * h,
                    aspect_ratio=w * width / (h * height),
                    width_px=w * width,
                    height_px=h * height,
                    sqrt_area_960=np.sqrt(w * width * h * height)
                    * 960
                    / max(width, height),
                    center_x=x,
                    center_y=y,
                )
                local.append(b)
            boxes.extend(local)
            sides = [b["max_side_norm"] for b in local]
            q10, q90 = (
                np.percentile(sides, [10, 90]) if len(sides) >= 2 else (None, None)
            )
            condition = (
                re.match(r"[a-z]+", path.stem).group()
                if name == "oranges_field"
                else name
            )
            source = (
                "_".join(path.stem.split("_")[:2])
                if name == "oranges_field"
                else path.stem
            )
            rows.append(
                dict(
                    dataset=name,
                    image=key,
                    split=split,
                    condition=condition,
                    source_photo=source,
                    width=width,
                    height=height,
                    boxes=len(local),
                    median_max_side=float(np.median(sides)) if sides else None,
                    within_p90_p10=float(q90 / q10) if q10 else None,
                    sum_box_area=sum(b["area_norm"] for b in local),
                    edge_boxes=edge,
                    duplicate_boxes=duplicate_boxes,
                    brightness=float(gray.mean()),
                    contrast=float(gray.std()),
                    saturation=float(sat.mean()),
                    image_sha256=image_hash,
                    label_sha256=label_hash,
                )
            )

    def summarize(selected):
        ids = {r["image"] for r in selected}
        bb = [b for b in boxes if b["image"] in ids]
        empty = sum(r["boxes"] == 0 for r in selected)
        return {
            "images": len(selected),
            "boxes": len(bb),
            "empty_images": empty,
            "empty_percent": 100 * empty / len(selected) if selected else None,
            "source_photos": len({r["source_photo"] for r in selected}),
            "image_metrics": {
                k: describe([r[k] for r in selected]) for k in IMAGE_METRICS
            },
            "box_metrics": {k: describe([b[k] for b in bb]) for k in BOX_METRICS},
            "count_bins": {
                label: sum(lo <= r["boxes"] <= hi for r in selected)
                for label, lo, hi in [
                    ("0", 0, 0),
                    ("1-5", 1, 5),
                    ("6-15", 6, 15),
                    ("16-30", 16, 30),
                    ("31-60", 31, 60),
                    ("61+", 61, float("inf")),
                ]
            },
            "edge_boxes_percent": 100 * sum(r["edge_boxes"] for r in selected) / len(bb)
            if bb
            else None,
            "duplicate_boxes": sum(r["duplicate_boxes"] for r in selected),
            "size_at_960_percent": {
                label: 100 * sum(lo <= b["sqrt_area_960"] < hi for b in bb) / len(bb)
                if bb
                else None
                for label, lo, hi in [
                    ("under_16", 0, 16),
                    ("16_to_32", 16, 32),
                    ("32_to_96", 32, 96),
                    ("96_plus", 96, float("inf")),
                ]
            },
        }

    summary = summarize(rows)
    positive = [r for r in rows if r["boxes"]]
    if (
        len(positive) > 1
        and np.std([r["boxes"] for r in positive]) > 0
        and np.std([r["median_max_side"] for r in positive]) > 0
    ):
        summary["count_size_correlation"] = float(
            np.corrcoef(
                [r["boxes"] for r in positive], [r["median_max_side"] for r in positive]
            )[0, 1]
        )
    else:
        summary["count_size_correlation"] = None
    summary["resolutions"] = dict(Counter(f"{r['width']}x{r['height']}" for r in rows))
    summary["duplicate_box_images"] = [r["image"] for r in rows if r["duplicate_boxes"]]
    summary["between_scene_p90_p10"] = (
        summary["image_metrics"]["median_max_side"]["p90"]
        / summary["image_metrics"]["median_max_side"]["p10"]
        if positive
        else None
    )
    summary.update(
        root=str(root),
        content_sha256=digest.hexdigest(),
        errors=errors,
        exact_duplicate_images=duplicates,
        splits={s: summarize([r for r in rows if r["split"] == s]) for s in splits},
        conditions={
            s: summarize([r for r in rows if r["condition"] == s])
            for s in sorted({r["condition"] for r in rows})
        },
    )
    return summary, rows, boxes


def raw_archive(path):
    groups = defaultdict(
        lambda: Counter(images=0, boxes=0, empty_images=0, missing_labels=0)
    )
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for n in sorted(names):
            p = Path(n)
            if p.parent.name != "images" or p.suffix.lower() not in {
                ".jpg",
                ".png",
                ".jpeg",
            }:
                continue
            label = str(p.parent.parent / "labels" / f"{p.stem}.txt")
            c = groups[p.parent.parent.name]
            c["images"] += 1
            if label not in names:
                c["missing_labels"] += 1
                continue
            count = len(
                [line for line in z.read(label).decode().splitlines() if line.strip()]
            )
            c["boxes"] += count
            c["empty_images"] += count == 0
    total = Counter(
        {
            key: sum(c[key] for c in groups.values())
            for key in ("images", "boxes", "empty_images", "missing_labels")
        }
    )
    for c in [total, *groups.values()]:
        c["empty_percent"] = (
            100 * c["empty_images"] / c["images"] if c["images"] else None
        )
    return {
        "sha256": sha256_file(path),
        "total": dict(total),
        "conditions": dict(groups),
    }


def mixture(profiles, manual_weight):
    # Igual peso por imagem; caixas repartem o peso da sua imagem positiva.
    output = {
        "manual_weight": manual_weight,
        "external_weight": 1 - manual_weight,
        "image_metrics": {},
        "box_metrics_image_balanced": {},
    }
    domains = [("manual-full", manual_weight), ("oranges_field", 1 - manual_weight)]
    for metric in IMAGE_METRICS:
        values, weights = [], []
        for name, weight in domains:
            selected = [r for r in profiles[name][1] if r[metric] is not None]
            values.extend(r[metric] for r in selected)
            weights.extend([weight / len(selected)] * len(selected))
        output["image_metrics"][metric] = weighted_description(values, weights)
    for metric in BOX_METRICS:
        values, weights = [], []
        for name, weight in domains:
            positive = {r["image"]: r["boxes"] for r in profiles[name][1] if r["boxes"]}
            for b in profiles[name][2]:
                values.append(b[metric])
                weights.append(weight / len(positive) / positive[b["image"]])
        output["box_metrics_image_balanced"][metric] = weighted_description(
            values, weights
        )
    output["empty_percent"] = sum(
        w * profiles[n][0]["empty_percent"] for n, w in domains
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manual", type=Path, default=Path("data/real_yolo_confirmatory"))
    p.add_argument(
        "--external", type=Path, default=Path("data/external_tests/oranges_field")
    )
    p.add_argument(
        "--synthetic", type=Path, default=Path("data/generated/confirmatory_pool")
    )
    p.add_argument(
        "--archive", type=Path, default=Path("data/archives/oranges-in-the-field.zip")
    )
    p.add_argument("--output", type=Path, default=Path("artifacts/dataset_profiles"))
    p.add_argument("--manual-weight", type=float, default=0.5)
    a = p.parse_args()
    if not 0 < a.manual_weight < 1:
        p.error("--manual-weight deve estar entre 0 e 1, exclusive")
    profiles = {}
    for name, root, splits in [
        ("manual-full", a.manual, ("train", "val")),
        ("oranges_field", a.external, ("test",)),
        ("synthetic", a.synthetic, ("train", "val")),
    ]:
        profiles[name] = scan(root, splits, name)
        print(name, profiles[name][0]["images"], profiles[name][0]["boxes"], flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    for file, index in [("images.csv", 1), ("boxes.csv", 2)]:
        rows = [r for v in profiles.values() for r in v[index]]
        with (a.output / file).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "schema_version": 1,
        "profiles": {k: v[0] for k, v in profiles.items()},
        "mixture": mixture(profiles, a.manual_weight),
        "external_archive": raw_archive(a.archive),
        "method": {
            "box_size": "max(w/W,h/H); also report max(w,h)/min(W,H)",
            "appearance": "RGB thumbnail <=256px; L mean/std and HSV saturation, 0..255",
            "mixture": "equal image weight within each domain; per-box weights split across each positive image; inverse empirical CDF",
            "within_scene": "p90/p10 of max normalized box side for images with >=2 boxes",
            "coverage": "sum of bbox areas, overlaps counted repeatedly; not fruit mask coverage",
            "size_960": "sqrt box area after aspect-preserving longest-side resize to 960; excludes padding",
        },
    }
    (a.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if any(v[0]["errors"] for v in profiles.values()):
        raise SystemExit("Há erros de integridade; confira summary.json")


if __name__ == "__main__":
    main()
