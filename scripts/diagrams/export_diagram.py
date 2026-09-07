"""Exporta o primeiro SVG de um diagrama HTML sem adicionar dependências de rede.

O HTML é um preview gerado pelos scripts Python. A exportação SVG apenas extrai o elemento vetorial;
a exportação PNG segue o procedimento da skill `diagram-design` e requer Playwright.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def first_svg(page: str, source: Path) -> str:
    match = re.search(r"<svg\b.*?</svg>", page, re.DOTALL)
    if not match:
        raise SystemExit(f"nenhum <svg> encontrado em {source}")
    svg = match.group(0)
    opening = svg.split(">", 1)[0]
    if "viewBox=" not in opening:
        raise SystemExit(f"o primeiro <svg> de {source} não possui viewBox")
    if "xmlns=" not in opening:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg


def export_svg(source: Path, svg: str) -> Path:
    output = source.with_suffix(".svg")
    output.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n",
        encoding="utf-8",
    )
    return output


def export_png(source: Path, scale: float) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright isn't installed. To enable PNG export, run:\n"
            "```\n"
            "pip install playwright\n"
            "playwright install chromium\n"
            "```\n"
            "Then repeat the export command."
        ) from None

    output = source.with_suffix(".png")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(device_scale_factor=scale)
        page.goto(source.as_uri())
        page.wait_for_load_state("networkidle")
        page.evaluate("() => document.fonts.ready")
        page.locator("svg").first.screenshot(path=str(output), omit_background=True)
        browser.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--format", choices=("svg", "png", "both"), default="both")
    parser.add_argument("--scale", type=float, default=2)
    args = parser.parse_args()

    source = args.source.resolve()
    page = source.read_text(encoding="utf-8")
    svg = first_svg(page, source)
    outputs: list[Path] = []
    if args.format in {"svg", "both"}:
        outputs.append(export_svg(source, svg))
    if args.format in {"png", "both"}:
        outputs.append(export_png(source, args.scale))
    print(" · ".join(path.name for path in outputs))


if __name__ == "__main__":
    main()
