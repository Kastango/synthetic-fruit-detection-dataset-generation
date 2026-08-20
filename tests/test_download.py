import hashlib
import zipfile

import fruit_pipeline.download as download_module
from fruit_pipeline.common import ROOT, load_yaml
from fruit_pipeline.download import (
    SOURCES,
    _remove_ignored_prepared_assets,
    download_http,
    download_real_source,
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


def test_canonical_dataset_download_urls_are_frozen() -> None:
    config = load_yaml(ROOT / "configs" / "pipeline.yaml")
    manual = config["real_dataset"]["source"]
    citdet = config["external_datasets"]["citdet"]

    assert manual["download_url"] == (
        "https://drive.usercontent.google.com/download?"
        "id=1wonaftsf0E_KzMJ3O1-a-79ik-PcsICy&export=download&authuser=0"
    )
    assert citdet["download_url"] == (
        "https://mavmatrix.uta.edu/context/cse_datasets/article/1000/"
        "type/native/viewcontent"
    )


def test_manual_download_uses_configured_google_drive_url(
    tmp_path, monkeypatch
) -> None:
    fixture = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture, "w") as archive:
        archive.writestr("images/sample.txt", "fixture")
    payload = fixture.read_bytes()
    seen_drive_ids = []

    def fake_download(source, destination) -> None:
        seen_drive_ids.append(source.drive_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(download_module, "download_google_drive", fake_download)
    config = {
        "paths": {"archives": str(tmp_path / "archives")},
        "real_dataset": {
            "source": {
                "download_url": (
                    "https://drive.usercontent.google.com/download?"
                    "id=manual-drive-id&export=download"
                ),
                "archive_name": "datanotation.zip",
                "expected_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }

    result = download_real_source(config)

    assert result.read_bytes() == payload
    assert seen_drive_ids == ["manual-drive-id"]


def test_http_download_reuses_valid_archive(tmp_path, monkeypatch, capsys) -> None:
    destination = tmp_path / "dataset.zip"
    payload = b"already downloaded"
    destination.write_bytes(payload)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("a rede não deveria ser acessada")

    monkeypatch.setattr(download_module.urllib.request, "urlopen", fail_if_called)

    result = download_http(
        "https://example.test/dataset.zip",
        destination,
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert result == destination
    assert "reutilizando arquivo validado" in capsys.readouterr().out
