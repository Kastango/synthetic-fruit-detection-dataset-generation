"""Embed the versioned fonts. Diagram builds never fetch fonts from the network."""

import base64
import io
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

FONTS = Path(__file__).resolve().parents[2] / "docs/figures/fonts"
FILES = {
    "Geist": "Geist[wght].ttf",
    "Geist Mono": "GeistMono[wght].ttf",
    "Instrument Serif": "InstrumentSerif-Regular.ttf",
}


def face(family, spec, glyphs, slug, *, weight, style="normal"):
    font = TTFont(FONTS / FILES[family], recalcTimestamp=False)
    missing = {ord(c) for c in glyphs} - set(font.getBestCmap())
    if missing:
        raise ValueError(f"{family}: missing glyphs {sorted(missing)}")
    options = subset.Options()
    options.drop_tables += ["meta"]
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text="".join(sorted(glyphs)))
    subsetter.subset(font)
    stream = io.BytesIO()
    font.save(stream)
    data = base64.b64encode(stream.getvalue()).decode()
    return (
        f"@font-face{{font-family:'{family}';font-style:{style};"
        f"font-weight:{weight};font-display:block;"
        f"src:url(data:font/ttf;base64,{data}) format('truetype');}}"
    )
