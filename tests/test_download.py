from fruit_pipeline.download import (
    SOURCES,
    _remove_ignored_prepared_assets,
    selected_sources,
)


def test_frozen_public_source_sizes() -> None:
    assert SOURCES["prepared"].expected_bytes == 1_544_768_252
    assert SOURCES["fruits"].expected_bytes == 236_417_183
    assert SOURCES["backgrounds"].expected_bytes == 1_431_673_341


def test_raw_mode_has_the_two_rebuild_inputs() -> None:
    assert [source.key for source in selected_sources("raw")] == [
        "fruits",
        "backgrounds",
    ]


def test_prepared_cleanup_removes_legacy_light_directory(tmp_path) -> None:
    ignored = tmp_path / "lights"
    ignored.mkdir()
    (ignored / "texture.png").write_bytes(b"legacy")

    _remove_ignored_prepared_assets(SOURCES["prepared"], tmp_path)

    assert not ignored.exists()
