"""Install the small Studio asset bundle without preprocessing models."""

from pathlib import Path
import json
import tempfile
import urllib.request

from .common import ROOT, extract_zip_atomic, sha256_file


def install_demo_assets(target: Path) -> None:
    spec = json.loads((ROOT / "resources/studio-demo.json").read_text())
    bundled = ROOT / "resources/studio-demo.zip"
    with tempfile.TemporaryDirectory(prefix="studio-download-") as tmp:
        archive = bundled
        if not bundled.exists():
            archive = Path(tmp) / "assets.zip"
            with urllib.request.urlopen(spec["url"], timeout=60) as response:
                with archive.open("wb") as output:
                    remaining = spec["bytes"] + 1
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        output.write(chunk)
                        remaining -= len(chunk)
        if (
            archive.stat().st_size != spec["bytes"]
            or sha256_file(archive) != spec["sha256"]
        ):
            raise ValueError("Pacote de dados inválido. Baixe novamente o repositório.")
        extract_zip_atomic(archive, target)
