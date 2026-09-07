"""Check portable SVGs, embedded images and accessible labels without a browser."""

import base64
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NS = "{http://www.w3.org/2000/svg}"


def check(path):
    source = path.read_text()
    root = ET.fromstring(source)
    assert root.tag == NS + "svg" and root.get("viewBox"), path
    assert root.get("role") == "img", path
    assert root[0].tag == NS + "title" and root[0].text, path
    ids = [node.get("id") for node in root.iter() if node.get("id")]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {path}"
    labels = root.get("aria-labelledby", "").split()
    assert len(labels) == 2 and all(label in ids for label in labels), path
    assert root.find(NS + "desc").text, path
    for url in re.findall(r"url\(([^)]+)\)", source):
        assert url.strip("'\"").startswith(("data:", "#")), f"External URL in {path}"
    for node in root.iter():
        assert node.tag not in {NS + "script", NS + "foreignObject"}, path
        href = node.get("href", "")
        assert not href or href.startswith(("#", "data:")), path
        if href.startswith("#"):
            assert href[1:] in ids, f"Missing reference {href}: {path}"
        if node.tag == NS + "image":
            media, data = href.split(",", 1)
            with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
                assert media == f"data:image/{image.format.lower()};base64", path
                image.verify()
    print(f"OK {path.relative_to(ROOT)}")


if __name__ == "__main__":
    check(ROOT / "docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg")
    for path in sorted((ROOT / "docs/figures/condicoes").glob("*.svg")):
        check(path)
