from copy import deepcopy
import json
from pathlib import Path
import pytest
from PIL import Image
from fruit_pipeline.common import ROOT, load_yaml, sha256_file
from fruit_pipeline.studio import resolve_recipe, DEFAULTS, SIMPLE, Studio
from fruit_pipeline.synthesis import generate_dataset, scene_seed, create_asset_catalog
from fruit_pipeline.similarity import image_features, compare_features
from test_synthesis import build_assets, tiny_config


def test_paired_appearance_preserves_geometry_and_changes_pixels(tmp_path):
    assets = tmp_path / "assets"
    build_assets(assets)
    config = tiny_config()
    config["sampling"] = {"mode": "paired-v1"}
    config["objects"].update(min=3, max=3)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(assets, first, config, train_ratio=0.67, split_seed=42)
    changed = deepcopy(config)
    changed["appearance"]["exposure_jitter"] = {
        "enabled": True,
        "probability": 1,
        "range": [0.4, 0.4],
        "saturation_pull": 0,
    }
    generate_dataset(
        assets, second, changed, train_ratio=0.67, split_seed=42, workers=2
    )
    for label in (first / "labels").rglob("*.txt"):
        assert label.read_bytes() == (second / label.relative_to(first)).read_bytes()
    a = [json.loads(s) for s in (first / "manifest.jsonl").read_text().splitlines()]
    b = [json.loads(s) for s in (second / "manifest.jsonl").read_text().splitlines()]
    for x, y in zip(a, b):
        for key in ["seed", "background", "cutouts", "annotations"]:
            assert x[key] == y[key]
    assert any(
        (first / r["image"]).read_bytes() != (second / r["image"]).read_bytes()
        for r in a
    )
    assert (first / "manifest.jsonl").read_bytes() != (
        second / "manifest.jsonl"
    ).read_bytes()


def test_seed_reproduces_across_workers_and_name_and_total(tmp_path):
    assets = tmp_path / "assets"
    build_assets(assets)
    c = tiny_config()
    c["sampling"] = {"mode": "paired-v1"}
    c["augmentation"] = {"horizontal_flip": True}
    c["objects"].update(min=1, max=4)
    c["images"]["total"] = 20
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset(assets, a, c, train_ratio=0.67, split_seed=42)
    generate_dataset(assets, b, c, train_ratio=0.67, split_seed=42, workers=2)
    for file in (a / "images").rglob("*.jpg"):
        assert file.read_bytes() == (b / file.relative_to(a)).read_bytes()
    assert (a / "manifest.jsonl").read_bytes() == (b / "manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in (a / "manifest.jsonl").read_text().splitlines()]
    assert len({row["background"] for row in rows}) == 4
    assert {row["background_mirrored"] for row in rows} == {True, False}
    unmirrored = deepcopy(c)
    unmirrored["augmentation"]["horizontal_flip"] = False
    generate_dataset(assets, tmp_path / "unmirrored", unmirrored, train_ratio=0.67, split_seed=42)
    before = [json.loads(line) for line in (tmp_path / "unmirrored/manifest.jsonl").read_text().splitlines()]
    assert [(r["background"], r["requested_objects"]) for r in rows] == [
        (r["background"], r["requested_objects"]) for r in before
    ]
    other = deepcopy(c)
    other["name"] = "renamed"
    other["images"]["total"] = 40
    assert scene_seed(c, "assets", 0) == scene_seed(other, "assets", 0)
    other["seed"] += 1
    assert scene_seed(c, "assets", 0) != scene_seed(other, "assets", 0)


def test_recipe_limits_and_removes_effects():
    base = load_yaml(ROOT / "configs/synthesis/confirmatory_pool.yaml")
    c = resolve_recipe(base, SIMPLE, "essential", 42)
    assert "cast_shadow" not in c["occlusion"]
    assert "exposure_jitter" not in c["appearance"]
    assert "dense" not in c["objects"]
    assert (c["objects"]["min"], c["objects"]["max"]) == (10, 100)
    assert c["augmentation"]["horizontal_flip"] is True
    assert "dense" in base["objects"]
    for invalid in [
        {"max_scale": float("nan")},
        {"min_scale": 7, "max_scale": 6},
        {"unknown": 1},
        {"fruit_max": 3.5},
        {"fruit_min": 20, "fruit_max": 10},
        {"fruit_min": -1},
    ]:
        with pytest.raises(ValueError):
            resolve_recipe(base, invalid, "reference", 42)
    with pytest.raises(ValueError):
        resolve_recipe(base, {}, "reference", -1)


def test_empty_boxes_and_identical_distribution():
    image = Image.new("RGB", (20, 20), "green")
    features = image_features(image, [])
    rows = compare_features(features, features)
    assert rows[0]["distance"] == 0
    assert all(row["distance"] is None for row in rows[1:])
    features = image_features(image, [[0.5, 0.5, 0.3, 0.3]])
    assert all(row["distance"] == 0 for row in compare_features(features, features))


def test_stale_generator_marker_rejected(tmp_path):
    assets = tmp_path / "assets"
    build_assets(assets)
    out = tmp_path / "out"
    c = tiny_config()
    generate_dataset(assets, out, c, train_ratio=0.67, split_seed=42)
    marker = json.loads((out / "generation_config.json").read_text())
    assert marker["generator_sha256"] == sha256_file(
        ROOT / "fruit_pipeline/synthesis.py"
    )
    marker["generator_sha256"] = "old-code"
    (out / "generation_config.json").write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="generator_sha256"):
        generate_dataset(assets, out, c, train_ratio=0.67, split_seed=42)


def test_content_hash_catches_same_size_asset_change(tmp_path):
    assets = tmp_path / "assets"
    build_assets(assets)
    before = create_asset_catalog(assets)
    path = assets / "pictures_trimmed/fruit-0.png"
    original = path.read_bytes()
    # Modification de conteúdo mesmo tamanho; catálogo não precisa decodificar.
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    after = create_asset_catalog(assets)
    assert before["source_fingerprint"] != after["source_fingerprint"]
    assert len(before["sha256"]) == 12


def test_studio_needs_only_synthetic_assets_and_exports_zip(tmp_path):
    import time, zipfile, yaml

    assets = tmp_path / "assets"
    build_assets(assets)
    studio = Studio(assets, tmp_path / "studio/preview")
    try:
        preview = studio.render(
            {"seed": 12, "controls": {"fruit_min": 1, "fruit_max": 2}}
        )
        assert len(preview["synthetic"]) == 8
        assert all(
            v["background"].startswith("data:image/jpeg") for v in preview["synthetic"]
        )
        assert "real" not in preview
        job = studio.start_job(
            {
                "seed": 12,
                "controls": {"fruit_min": 1, "fruit_max": 2},
                "total": 3,
            }
        )
        deadline = time.monotonic() + 30
        while (
            studio.job_status(job["id"])["status"] == "running"
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        final = studio.job_status(job["id"])
        assert final["status"] == "complete", final
        with zipfile.ZipFile(
            (studio.jobs_root / job["id"]).with_suffix(".zip")
        ) as archive:
            names = archive.namelist()
            assert len([n for n in names if n.endswith(".jpg")]) == 3
            assert len([n for n in names if n.startswith("labels/")]) == 3
            assert "path" not in yaml.safe_load(archive.read("data.yaml"))
            recipe = yaml.safe_load(archive.read("recipe.yaml"))
            assert recipe["seed"] == 12
            assert "dense" not in recipe["objects"]
            assert recipe["augmentation"]["horizontal_flip"] is True
        assert (
            studio.start_job(
                {
                    "seed": 12,
                    "controls": {"fruit_min": 1, "fruit_max": 2},
                    "total": 3,
                }
            )["id"]
            == job["id"]
        )
        with pytest.raises(ValueError):
            studio.start_job({"total": 5001})
        with pytest.raises(ValueError):
            studio.job_status("../bad")
    finally:
        studio.executor.shutdown(wait=True)


def test_preview_matches_generation_by_scene_index(tmp_path):
    import json
    from fruit_pipeline.studio import picture

    assets = tmp_path / "assets"
    build_assets(assets)
    studio = Studio(assets, tmp_path / "preview")
    try:
        preview = studio.render(
            {"seed": 42, "controls": {"fruit_min": 1, "fruit_max": 2}}
        )
        config = deepcopy(preview["config"])
        config["images"]["total"] = 4
        output = tmp_path / "generated"
        generate_dataset(
            assets, output, config, train_ratio=0.5, split_seed=42, workers=2
        )
        rows = [
            json.loads(s) for s in (output / "manifest.jsonl").read_text().splitlines()
        ]
        for row in rows:
            with Image.open(output / row["image"]) as im:
                assert (
                    picture(im)
                    == preview["synthetic"][row["generation_index"]]["image"]
                )
    finally:
        studio.executor.shutdown(wait=True)
