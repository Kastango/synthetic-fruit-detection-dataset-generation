from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fruit_pipeline.real_data import validate_yolo_text
from fruit_pipeline.synthesis import (
    _apply_appearance_hsv_cast,
    _finish_placement,
    create_asset_catalog,
    create_scene_split,
    generate_dataset,
    materialize_nested_subsets,
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
        "images": {"total": 3},
        "canvas": [64, 64],
        "objects": {
            "min": 1,
            "max": 1,
            "min_scale": 0.25,
            "max_scale": 0.25,
            "rotation_degrees": 0,
        },
        "placement": {
            "min_depth": 0,
            "min_visibility": 0.25,
            "max_attempts_per_object": 5,
        },
        # hsv_cast neutro: nenhum canal adota o alvo ambiental, então a fruta
        # sai idêntica ao recorte e os testes de geometria ficam isolados da
        # etapa de aparência.
        "appearance": {
            "hsv_cast": {
                "hue_power": 0.0,
                "saturation_power": 0.0,
                "value_power": 0.0,
            }
        },
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
    config["images"] = {"total": 20}
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


def test_asset_catalog_contains_every_pair_and_cutout(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    build_assets(assets)
    catalog = create_asset_catalog(assets)

    assert catalog["version"] == 1
    assert len(catalog["assets"]["backgrounds"]) == 4
    assert len(catalog["assets"]["cutouts"]) == 4
    assert "splits" not in catalog


def test_scene_split_is_deterministic_complete_and_disjoint() -> None:
    first = create_scene_split(10, 0.8, 42)
    second = create_scene_split(10, 0.8, 42)

    assert first == second
    assert len(first["splits"]["train"]) == 8
    assert len(first["splits"]["val"]) == 2
    assert not set(first["splits"]["train"]) & set(first["splits"]["val"])
    assert set(first["splits"]["train"]) | set(first["splits"]["val"]) == set(range(10))


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
    catalog = json.loads((assets / "asset_catalog.json").read_text())
    assert set(catalog["assets"]) == {"backgrounds", "cutouts"}
    assert catalog["version"] == 1
    scene_split = json.loads((output / "scene_split.json").read_text())
    assert len(scene_split["splits"]["train"]) == 2
    assert len(scene_split["splits"]["val"]) == 1
    for label in (output / "labels").rglob("*.txt"):
        assert validate_yolo_text(label.read_text(), str(label)) == 1


def test_train_and_val_are_generated_from_the_same_complete_asset_catalog(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    for name in ("backgrounds", "backgrounds_map", "pictures_trimmed"):
        (assets / name).mkdir(parents=True)
    Image.new("RGB", (64, 64), (30, 90, 40)).save(
        assets / "backgrounds" / "only-background.jpg"
    )
    Image.new("L", (64, 64), 100).save(
        assets / "backgrounds_map" / "only-background_depth.png"
    )
    cutout = Image.new("RGBA", (16, 16), (230, 120, 20, 255))
    cutout.save(assets / "pictures_trimmed" / "only-fruit.png")
    config = tiny_config()
    config["images"] = {"total": 2}

    generate_dataset(assets, output, config, train_ratio=0.5, split_seed=42, workers=1)

    records = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert {record["split"] for record in records} == {"train", "val"}
    assert {record["background"] for record in records} == {"only-background.jpg"}
    assert {tuple(record["cutouts"]) for record in records} == {("only-fruit.png",)}


def test_scene_composition_is_independent_of_the_final_split_ratio(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    build_assets(assets)
    config = tiny_config()
    config["images"] = {"total": 4}

    generate_dataset(
        assets, first_output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    generate_dataset(
        assets, second_output, config, train_ratio=0.75, split_seed=42, workers=1
    )

    def records_by_generation_index(root: Path) -> dict[int, dict]:
        return {
            int(record["generation_index"]): record
            for record in (
                json.loads(line)
                for line in (root / "manifest.jsonl").read_text().splitlines()
            )
        }

    first = records_by_generation_index(first_output)
    second = records_by_generation_index(second_output)
    assert set(first) == set(second) == set(range(4))
    for generation_index in first:
        first_record = first[generation_index]
        second_record = second[generation_index]
        assert first_record["seed"] == second_record["seed"]
        assert first_record["background"] == second_record["background"]
        assert first_record["cutouts"] == second_record["cutouts"]
        assert (first_output / first_record["image"]).read_bytes() == (
            second_output / second_record["image"]
        ).read_bytes()
        assert (first_output / first_record["label"]).read_bytes() == (
            second_output / second_record["label"]
        ).read_bytes()


def test_materialize_nested_subsets_are_complete_and_nested(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    pool = tmp_path / "pool"
    generated = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["images"] = {"total": 10}
    generate_dataset(assets, pool, config, train_ratio=0.8, split_seed=42, workers=1)

    summary = materialize_nested_subsets(
        pool,
        generated,
        [1, 2],
        base_size=4,
        base_val_size=1,
    )

    assert summary["synthetic-1x"]["train_images"] == 4
    assert summary["synthetic-1x"]["val_images"] == 1
    assert summary["synthetic-2x"]["train_images"] == 8
    assert summary["synthetic-2x"]["val_images"] == 2
    one_train = {
        path.name
        for path in (generated / "synthetic-1x" / "images" / "train").glob("*")
    }
    two_train = {
        path.name
        for path in (generated / "synthetic-2x" / "images" / "train").glob("*")
    }
    one_val = {
        path.name for path in (generated / "synthetic-1x" / "images" / "val").glob("*")
    }
    two_val = {
        path.name for path in (generated / "synthetic-2x" / "images" / "val").glob("*")
    }
    assert one_train < two_train
    assert one_val < two_val
    for multiplier, train_count, val_count in ((1, 4, 1), (2, 8, 2)):
        root = generated / f"synthetic-{multiplier}x"
        assert len(list((root / "images" / "train").glob("*"))) == train_count
        assert len(list((root / "images" / "val").glob("*"))) == val_count
        assert (root / "manifest.jsonl").exists()
        assert (root / "data.yaml").exists()


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


def test_value_power_jitter_validates_non_negative() -> None:
    config = tiny_config()
    config["appearance"]["hsv_cast"] = {
        "enabled": True,
        "hue_power": 0.1,
        "saturation_power": 0.1,
        "value_power": 0.4,
        "value_power_jitter": -0.1,
    }
    with pytest.raises(ValueError, match="value_power_jitter"):
        validate_synthesis_config(config)


def test_value_power_jitter_generation_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["appearance"]["hsv_cast"] = {
        "enabled": True,
        "use_hardlight_target": True,
        "hue_power": 0.1,
        "saturation_power": 0.1,
        "value_power": 0.45,
        "value_power_jitter": 0.3,
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


def test_bright_flatten_strength_validates_bounds() -> None:
    config = tiny_config()
    config["appearance"]["hsv_cast"] = {
        "enabled": True,
        "hue_power": 0.1,
        "saturation_power": 0.1,
        "value_power": 0.4,
        "bright_flatten_strength": 1.5,
    }
    with pytest.raises(ValueError, match="bright_flatten_strength"):
        validate_synthesis_config(config)


def test_bright_flatten_strength_pulls_more_in_bright_regions() -> None:
    fruit = Image.new("RGBA", (8, 8), (200, 120, 40, 255))
    bright_region = Image.new("RGB", (8, 8), (250, 250, 250))
    unboosted = _apply_appearance_hsv_cast(
        fruit,
        bright_region,
        {
            "use_hardlight_target": True,
            "hue_power": 0.1,
            "saturation_power": 0.1,
            "value_power": 0.3,
            "bright_flatten_strength": 0.0,
        },
    )
    boosted = _apply_appearance_hsv_cast(
        fruit,
        bright_region,
        {
            "use_hardlight_target": True,
            "hue_power": 0.1,
            "saturation_power": 0.1,
            "value_power": 0.3,
            "bright_flatten_strength": 1.0,
        },
    )
    unboosted_v = np.asarray(unboosted.convert("HSV"))[..., 2].astype(np.float32)
    boosted_v = np.asarray(boosted.convert("HSV"))[..., 2].astype(np.float32)
    assert boosted_v.mean() > unboosted_v.mean()


def test_cast_shadow_config_validates_coverage_order() -> None:
    config = tiny_config()
    config["occlusion"]["cast_shadow"] = {
        "enabled": True,
        "min_coverage": 0.7,
        "max_coverage": 0.5,
    }
    with pytest.raises(ValueError, match="cast_shadow"):
        validate_synthesis_config(config)


def test_cast_shadow_generation_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["occlusion"]["cast_shadow"] = {
        "enabled": True,
        "probability": 1.0,
        "strength": 0.3,
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


def test_edge_feather_radius_validates_non_negative() -> None:
    config = tiny_config()
    config["occlusion"]["edge_feather_radius"] = -1.0
    with pytest.raises(ValueError, match="edge_feather_radius"):
        validate_synthesis_config(config)


def test_edge_feather_radius_generation_is_deterministic_and_labels_are_valid(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["occlusion"]["edge_feather_radius"] = 0.8
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


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("z_patch_fraction", 0.0, "z_patch_fraction"),
        ("mask_threshold", 1.0, "mask_threshold"),
    ],
)
def test_center_patch_occlusion_config_validates_bounds(
    key: str, value: float, message: str
) -> None:
    config = tiny_config()
    if key == "z_patch_fraction":
        config["placement"][key] = value
    else:
        config["occlusion"][key] = value
    with pytest.raises(ValueError, match=message):
        validate_synthesis_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [("strength", 1.1), ("radius_fraction", 0.0)],
)
def test_contact_shadow_config_validates_bounds(key: str, value: float) -> None:
    config = tiny_config()
    config["occlusion"]["contact_shadow"] = {
        "enabled": True,
        "strength": 0.45,
        "radius_fraction": 0.04,
        key: value,
    }
    with pytest.raises(ValueError, match=key):
        validate_synthesis_config(config)


def test_center_patch_anchors_z_and_binarizes_blurred_mask() -> None:
    fruit = Image.new("RGBA", (9, 9), (230, 130, 20, 255))
    canvas = Image.new("RGB", (9, 9), (30, 90, 40))
    depth = np.full((9, 9), 30, dtype=np.uint8)
    depth[:, 5:] = 180
    depth[4, 4] = 100
    alpha = np.asarray(fruit.getchannel("A"), dtype=np.uint8)
    config = tiny_config()
    config["placement"].update(
        {
            "z_patch_fraction": 0.1,
            "z_offset": 0.0,
            "min_visibility": 0.0,
        }
    )
    config["occlusion"].update({"edge_blur": 1.5, "mask_threshold": 0.5})

    result = _finish_placement(
        fruit,
        0,
        0,
        alpha,
        alpha.astype(np.float32),
        alpha > 8,
        int((alpha > 8).sum()),
        canvas,
        depth,
        config,
        anchor=(4, 4),
    )

    assert result is not None
    assert result["z"] == 100.0
    assert set(np.unique(result["visible_mask"])) <= {0, 255}


def test_contact_shadow_darkens_only_next_to_occluded_region() -> None:
    fruit = Image.new("RGBA", (15, 15), (200, 120, 40, 255))
    canvas = Image.new("RGB", fruit.size, (30, 90, 40))
    depth = np.full(fruit.size[::-1], 30, dtype=np.uint8)
    depth[:, 8:] = 180
    alpha = np.asarray(fruit.getchannel("A"), dtype=np.uint8)
    config = tiny_config()
    config["placement"].update(
        {
            "z_patch_fraction": 0.1,
            "min_visibility": 0.0,
        }
    )
    config["occlusion"].update(
        {
            "mask_threshold": 0.5,
            "contact_shadow": {
                "enabled": True,
                "strength": 0.6,
                "radius_fraction": 0.2,
            },
        }
    )

    result = _finish_placement(
        fruit,
        0,
        0,
        alpha,
        alpha.astype(np.float32),
        alpha > 8,
        int((alpha > 8).sum()),
        canvas,
        depth,
        config,
        anchor=(3, 7),
    )

    assert result is not None
    rendered = np.asarray(result["image"].convert("RGB"))
    assert rendered[7, 7].mean() < rendered[7, 1].mean()
    assert result["visible_mask"][7, 7] == 255
    assert result["visible_mask"][7, 10] == 0


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


def test_z_offset_jitter_validates_non_negative() -> None:
    config = tiny_config()
    config["placement"]["z_offset_jitter"] = -1.0
    with pytest.raises(ValueError, match="z_offset_jitter"):
        validate_synthesis_config(config)


def test_z_offset_jitter_generation_is_deterministic(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    output = tmp_path / "generated"
    build_assets(assets)
    config = tiny_config()
    config["placement"]["z_offset_jitter"] = 40.0
    first = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    manifest_before = (output / "manifest.jsonl").read_bytes()
    second = generate_dataset(
        assets, output, config, train_ratio=0.5, split_seed=42, workers=1
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert (output / "manifest.jsonl").read_bytes() == manifest_before


def test_min_value_ratio_validates_bounds() -> None:
    config = tiny_config()
    config["appearance"]["hsv_cast"] = {
        "enabled": True,
        "hue_power": 0.1,
        "saturation_power": 0.1,
        "value_power": 0.5,
        "min_value_ratio": 1.5,
    }
    with pytest.raises(ValueError, match="min_value_ratio"):
        validate_synthesis_config(config)


def test_min_value_ratio_floors_darkening_toward_dark_target() -> None:
    fruit = Image.new("RGBA", (8, 8), (200, 120, 40, 255))
    dark_region = Image.new("RGB", (8, 8), (5, 5, 5))
    unfloored = _apply_appearance_hsv_cast(
        fruit,
        dark_region,
        {
            "use_hardlight_target": True,
            "hue_power": 0.1,
            "saturation_power": 0.1,
            "value_power": 0.9,
            "min_value_ratio": 0.0,
        },
    )
    floored = _apply_appearance_hsv_cast(
        fruit,
        dark_region,
        {
            "use_hardlight_target": True,
            "hue_power": 0.1,
            "saturation_power": 0.1,
            "value_power": 0.9,
            "min_value_ratio": 0.6,
        },
    )
    unfloored_v = np.asarray(unfloored.convert("HSV"))[..., 2].astype(np.float32)
    floored_v = np.asarray(floored.convert("HSV"))[..., 2].astype(np.float32)
    original_v = np.asarray(fruit.convert("RGB").convert("HSV"))[..., 2].astype(
        np.float32
    )
    assert floored_v.mean() > unfloored_v.mean()
    assert floored_v.mean() >= original_v.mean() * 0.6 - 1.0


def test_debug_panel_is_generated_without_changing_main_output(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    without_debug = tmp_path / "without_debug"
    with_debug = tmp_path / "with_debug"
    build_assets(assets)
    config = tiny_config()
    generate_dataset(
        assets, without_debug, config, train_ratio=0.5, split_seed=42, workers=1
    )
    generate_dataset(
        assets,
        with_debug,
        config,
        train_ratio=0.5,
        split_seed=42,
        workers=1,
        debug=True,
    )
    for relative in sorted(
        path.relative_to(without_debug)
        for path in without_debug.rglob("*")
        if path.is_file() and path.parts[-2] in {"train", "val"}
    ):
        assert (without_debug / relative).read_bytes() == (
            with_debug / relative
        ).read_bytes()
    debug_images = list((with_debug / "images_debug").rglob("*.jpg"))
    assert len(debug_images) == 3
    with Image.open(debug_images[0]) as panel:
        assert panel.width == config["canvas"][0] * 2 + 4


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
