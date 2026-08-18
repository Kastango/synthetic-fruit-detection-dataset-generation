from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from fruit_pipeline.common import extract_zip_atomic


def test_safe_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="insegura"):
        extract_zip_atomic(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_zip_can_exclude_top_level_directory(tmp_path: Path) -> None:
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("backgrounds/tree.jpg", b"background")
        handle.writestr("lights/texture.png", b"ignored")

    output = tmp_path / "output"
    extract_zip_atomic(archive, output, excluded_roots=frozenset({"lights"}))

    assert (output / "backgrounds" / "tree.jpg").read_bytes() == b"background"
    assert not (output / "lights").exists()
