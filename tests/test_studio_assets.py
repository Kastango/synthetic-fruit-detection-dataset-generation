import io
import json
import zipfile
from hashlib import sha256

import pytest
from fruit_pipeline import studio_assets


def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_assets, "ROOT", tmp_path)
    resources = tmp_path / "resources"
    resources.mkdir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("backgrounds/example.txt", "demo")
    payload = buffer.getvalue()
    (resources / "studio-demo.json").write_text(
        json.dumps(
            {
                "url": "https://example.test/assets.zip",
                "bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
        )
    )
    return resources, payload


def test_download_validates_and_installs_without_overwriting(tmp_path, monkeypatch):
    resources, payload = bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        studio_assets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    target = tmp_path / "installed"
    studio_assets.install_demo_assets(target)
    assert (target / "backgrounds/example.txt").read_text() == "demo"
    with pytest.raises(FileExistsError):
        studio_assets.install_demo_assets(target)
    assert (target / "backgrounds/example.txt").read_text() == "demo"


def test_corrupt_bundle_is_rejected_before_installation(tmp_path, monkeypatch):
    resources, payload = bundle(tmp_path, monkeypatch)
    (resources / "studio-demo.zip").write_bytes(b"x" * len(payload))
    target = tmp_path / "installed"
    with pytest.raises(ValueError, match="inválido"):
        studio_assets.install_demo_assets(target)
    assert not target.exists()
