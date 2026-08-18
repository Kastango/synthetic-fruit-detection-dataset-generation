from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fruit_pipeline.real_data import validate_yolo_text
from fruit_pipeline.synthesis import (
    create_asset_split,
    generate_dataset,
    validate_synthesis_config,
)


def build_assets(root: Path) -> None:
    for name in ("backgrounds", "backgrounds_map", "pictures_trimmed"):
        (root / name).mkdir(parents=True)
    for index in range(4):
        Image.new("RGB", (64, 64), (30 + index, 90, 40)).save(
            root / "backgrounds" / f"background-{index}.jpg"
        )
        Image.new("L", (64, 64), 100).save(
            root / "backgrounds_map" / f"background-{index}_depth.png"
        )
        cutout = Image.new("RGBA", (16, 16), (230, 120 + index, 20, 0))
        for x in range(3, 13):
            for y in range(3, 13):
                cutout.putpixel((x, y), (230, 120 + index, 20, 255))
        cutout.save(root / "pictures_trimmed" / f"fruit-{index}.png")


def tiny_config() -> dict:
    return {
        "name": "tiny",
        "seed": 42,
        "images": {"train": 2, "val": 1},
        "canvas": [64, 64],
        "objects": {
            "min": 1,
            "max": 1,
            "min_scale": 0.25,
            "max_scale": 0.25,
            "scale_mode": "canvas",
            "rotation_degrees": 0,
        },
        "placement": {
            "min_depth": 0,
            "min_visibility": 0.25,
            "max_attempts_per_object": 5,
            "z_method": "quantile",
            "z_quantile": 0.5,
        },
        "appearance": {"hardlight_power": 0.0},
        "occlusion": {"edge_blur": 0.0},
        "annotation": {"mode": "visible", "min_box_pixels": 2},
        "output": {"jpeg_quality": 90},
    }


def test_exclude_bottom_fraction_validates_bounds() -> None:
    config = tiny_config()
    config["placement"]["exclude_bottom_fraction"] = 1.5
    with pytest.raises(ValueError, match="exclude_bottom_fraction"):
        validate_synthesis_config(config)


def test_exclude_bottom_fraction_keeps_instances_out_of_bottom_band(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["images"] = {"train": 20, "val": 0}
    config["placement"]["exclude_bottom_fraction"] = 0.3
    generate_dataset(assets, output, config, train_ratio=0.5, split_seed=42, workers=1)
    canvas_height = config["canvas"][1]
    for label in (output / "labels" / "train").glob("*.txt"):
        for line in label.read_text().splitlines():
            _, _, center_y, _, _height = line.split()
            assert float(center_y) * canvas_height < canvas_height * 0.85


def test_light_texture_options_are_rejected() -> None:
    config = tiny_config()
    config["appearance"]["light_probability"] = 0.5
    with pytest.raises(ValueError, match="texturas de iluminação"):
        validate_synthesis_config(config)


def test_legacy_asset_split_is_regenerated_without_light_assets(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    build_assets(assets)
    split = create_asset_split(assets, train_ratio=0.5, seed=42)
    split["version"] = 1
    for partition in split["splits"].values():
        partition["lights"] = ["lights/legacy.png"]
    (assets / "asset_split.json").write_text(json.dumps(split))

    regenerated = create_asset_split(assets, train_ratio=0.5, seed=42)

    assert regenerated["version"] == 2
    assert all(
        "lights" not in partition for partition in regenerated["splits"].values()
    )


def test_generation_is_deterministic_and_labels_are_valid(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    first = generate_dataset(
        assets,
        output,
        tiny_config(),
        train_ratio=0.5,
        split_seed=42,
        workers=1,
    )
    manifest_before = (output / "manifest.jsonl").read_bytes()
    second = generate_dataset(
        assets,
        output,
        tiny_config(),
        train_ratio=0.5,
        split_seed=42,
        workers=1,
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == manifest_before
    records = [json.loads(line) for line in manifest_before.splitlines()]
    assert len(records) == 3
    assert all(record["annotations"] == 1 for record in records)
    split = json.loads((assets / "asset_split.json").read_text())
    assert set(split["splits"]["train"]) == {"backgrounds", "cutouts"}
    assert split["version"] == 2
    for label in (output / "labels").rglob("*.txt"):
        assert validate_yolo_text(label.read_text(), str(label)) == 1


def test_hsv_cast_config_validates_power_bounds() -> None:
    config = tiny_config()
    config["appearance"]["hsv_cast"] = {
        "enabled": True,
        "hue_power": 1.5,
        "saturation_power": 0.1,
        "value_power": 0.1,
    }
    with pytest.raises(ValueError, match="hsv_cast"):
        validate_synthesis_config(config)


def test_hsv_cast_generation_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["appearance"]["hsv_cast"] = {
        "enabled": True,
        "hue_power": 0.05,
        "saturation_power": 0.1,
        "value_power": 0.25,
    }
    first = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    manifest_before = (output / "manifest.jsonl").read_bytes()
    second = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == manifest_before
    for label in (output / "labels").rglob("*.txt"):
        assert validate_yolo_text(label.read_text(), str(label)) == 1


def test_hsv_cast_hardlight_target_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["appearance"]["hsv_cast"] = {
        "enabled": True,
        "use_hardlight_target": True,
        "hue_power": 0.08,
        "saturation_power": 0.12,
        "value_power": 0.35,
    }
    first = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    manifest_before = (output / "manifest.jsonl").read_bytes()
    second = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == manifest_before
    for label in (output / "labels").rglob("*.txt"):
        assert validate_yolo_text(label.read_text(), str(label)) == 1


def test_depth_smooth_radius_validates_non_negative() -> None:
    config = tiny_config()
    config["occlusion"]["depth_smooth_radius"] = -1.0
    with pytest.raises(ValueError, match="depth_smooth_radius"):
        validate_synthesis_config(config)


def test_depth_smooth_radius_generation_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["occlusion"]["depth_smooth_radius"] = 2.0
    first = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    manifest_before = (output / "manifest.jsonl").read_bytes()
    second = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == manifest_before
    for label in (output / "labels").rglob("*.txt"):
        assert validate_yolo_text(label.read_text(), str(label)) == 1


def test_scene_grading_config_validates_positive_factors() -> None:
    config = tiny_config()
    config["output"]["scene_grading"] = {"enabled": True, "contrast": 0.0}
    with pytest.raises(ValueError, match="scene_grading"):
        validate_synthesis_config(config)


def test_scene_grading_generation_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["output"]["scene_grading"] = {
        "enabled": True,
        "contrast": 1.25,
        "saturation": 1.35,
        "brightness": 1.05,
        "sharpen_percent": 60,
    }
    first = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    manifest_before = (output / "manifest.jsonl").read_bytes()
    second = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == manifest_before
    for label in (output / "labels").rglob("*.txt"):
        assert validate_yolo_text(label.read_text(), str(label)) == 1


def test_depth_scale_config_validates_scale_bounds() -> None:
    config = tiny_config()
    config["objects"]["depth_scale"] = {
        "enabled": True,
        "near_scale": 0.5,
        "far_scale": 1.5,
    }
    with pytest.raises(ValueError, match="depth_scale"):
        validate_synthesis_config(config)


def test_depth_scale_generation_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["objects"]["depth_scale"] = {
        "enabled": True,
        "near_scale": 1.4,
        "far_scale": 0.6,
    }
    first = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    manifest_before = (output / "manifest.jsonl").read_bytes()
    second = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == manifest_before
    for label in (output / "labels").rglob("*.txt"):
        assert validate_yolo_text(label.read_text(), str(label)) == 1


def test_parallel_generation_matches_single_worker(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    single = tmp_path / "single"
    parallel = tmp_path / "parallel"
    build_assets(assets)
    generate_dataset(
        assets,
        single,
        tiny_config(),
        train_ratio=0.5,
        split_seed=42,
        workers=1,
    )
    generate_dataset(
        assets,
        parallel,
        tiny_config(),
        train_ratio=0.5,
        split_seed=42,
        workers=2,
    )
    assert (single / "manifest.jsonl").read_bytes() == (
        parallel / "manifest.jsonl"
    ).read_bytes()
    for relative in sorted(
        path.relative_to(single)
        for path in single.rglob("*")
        if path.is_file() and path.parts[-2] in {"train", "val"}
    ):
        assert (single / relative).read_bytes() == (parallel / relative).read_bytes()
