"""Refresh selected decks from explicit YOLO datasets; record each source image."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures/diagram-assets"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="manual, controlled, synthetic or test; repeat to refresh several decks",
    )
    args = parser.parse_args()
    manifest_path = OUT / "cell_sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for item in args.dataset:
        name, directory = item.split("=", 1)
        if name not in {"manual", "controlled", "synthetic", "test"}:
            parser.error(f"Unknown deck: {name}")
        source = Path(directory).resolve()
        for split in ["test"] if name == "test" else ["train", "val"]:
            slug = "test" if name == "test" else f"{name}_{split}"
            files = sorted(
                p
                for p in (source / "images" / split).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not files:
                parser.error(f"No images in {source / 'images' / split}")
            if name == "controlled":
                # Interleave isolated fruits and foliage using the actual labels.
                groups = {True: [], False: []}
                for path in files:
                    label = source / "labels" / split / f"{path.stem}.txt"
                    groups[bool(label.read_text().strip())].append(path)
                if not all(groups.values()):
                    parser.error(f"controlled/{split} needs positives and negatives")
                files = [p for pair in zip(groups[True], groups[False]) for p in pair]
            limit = min(len(files), 64 if split == "train" else 30)
            # Keep the alternating order for controlled; evenly sample other decks.
            chosen = (
                files[:limit]
                if name == "controlled"
                else [
                    files[round(i * (len(files) - 1) / max(1, limit - 1))]
                    for i in range(limit)
                ]
            )
            records = []
            for index, path in enumerate(chosen):
                with Image.open(path) as image:
                    frame = ImageOps.fit(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        (120, 120),
                        Image.Resampling.LANCZOS,
                    )
                    frame.save(
                        OUT / f"cell_{slug}_{index}.jpg", quality=82, optimize=True
                    )
                records.append(
                    {
                        "image": path.relative_to(source).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            for old in OUT.glob(f"cell_{slug}_*.jpg"):
                if int(old.stem.rsplit("_", 1)[1]) >= len(chosen):
                    old.unlink()
            manifest[slug] = {"dataset": source.name, "images": records}
            print(f"{slug}: {len(chosen)} distinct thumbnails")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
