#!/usr/bin/env python3
"""Monta folhas de contato das figuras de resultados.

Cada folha reúne numa imagem só o que seria uma dezena de arquivos soltos. A
página de resultados faz poucas requisições grandes em vez de muitas pequenas,
o que a torna menos sensível a falhas intermitentes da CDN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from fruit_pipeline.common import ROOT

FONT = ROOT / "docs/figures/fonts/Geist[wght].ttf"
CONDITION_ORDER = [
    "ground-truth",
    "manual-full",
    "controlled",
    "synthetic-1x",
    "synthetic-2x",
    "synthetic-3x",
    "synthetic-5x",
    "synthetic-10x",
]
INK = "#233A33"
PAPER = "#FFFFFF"
RULE = "#D6DFDA"


def load_font(size):
    try:
        return ImageFont.truetype(str(FONT), size)
    except OSError:
        return ImageFont.load_default()


def sheet(tiles, columns, cell_width, label_height=34, pad=10):
    """Grade rotulada a partir de pares (legenda, imagem)."""
    font = load_font(21)
    scaled = []
    for caption, image in tiles:
        w, h = image.size
        target = (cell_width, max(1, round(h * cell_width / w)))
        scaled.append((caption, image.resize(target, Image.LANCZOS)))
    rows = (len(scaled) + columns - 1) // columns
    row_heights = [
        max(img.height for _, img in scaled[r * columns : (r + 1) * columns])
        for r in range(rows)
    ]
    width = columns * cell_width + (columns + 1) * pad
    height = sum(h + label_height for h in row_heights) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    y = pad
    for r in range(rows):
        x = pad
        for caption, img in scaled[r * columns : (r + 1) * columns]:
            draw.text((x, y + 6), caption, font=font, fill=INK)
            canvas.paste(img, (x, y + label_height))
            draw.rectangle(
                [x, y + label_height, x + img.width - 1, y + label_height + img.height - 1],
                outline=RULE,
            )
            x += cell_width + pad
        y += row_heights[r] + label_height + pad
    return canvas


def detection_sheet(folder: Path, columns: int, cell_width: int):
    tiles = []
    for name in CONDITION_ORDER:
        path = folder / f"{name}.jpg"
        if not path.exists():
            continue
        caption = "gabarito" if name == "ground-truth" else name
        tiles.append((caption, Image.open(path).convert("RGB")))
    if not tiles:
        raise SystemExit(f"nenhuma imagem em {folder}")
    return sheet(tiles, columns, cell_width)


def heatmap_sheet(folder: Path, columns: int, cell_width: int):
    order = [
        ("manual-full", "manual-full"),
        ("controlled", "controlled"),
        ("synthetic-1x", "synthetic-1x"),
        ("synthetic-2x", "synthetic-2x"),
        ("synthetic-3x", "synthetic-3x"),
        ("synthetic-5x", "synthetic-5x"),
        ("synthetic-10x", "synthetic-10x"),
        ("oranges_field", "coleta externa"),
        ("manual_full_val", "manual-full · val"),
    ]
    tiles = [
        (caption, Image.open(folder / f"{stem}.png").convert("RGB"))
        for stem, caption in order
        if (folder / f"{stem}.png").exists()
    ]
    return sheet(tiles, columns, cell_width)


def _com_caixas(imagem: Path, rotulo: Path):
    img = Image.open(imagem).convert("RGB")
    desenho = ImageDraw.Draw(img)
    largura = max(2, round(min(img.size) / 260))
    for linha in rotulo.read_text().splitlines():
        if not linha.strip():
            continue
        cx, cy, w, h = (float(v) for v in linha.split()[1:5])
        desenho.rectangle(
            [
                (cx - w / 2) * img.width, (cy - h / 2) * img.height,
                (cx + w / 2) * img.width, (cy + h / 2) * img.height,
            ],
            outline="#00E0FF", width=largura,
        )
    return img


def _medianas(imagens: Path, rotulos: Path, quantas: int):
    """As `quantas` cenas com contagem mais próxima da mediana da coleção.

    Escolher pela mediana, e não pelas primeiras do alfabeto, evita montar a
    comparação com as cenas mais fáceis ou mais cheias de qualquer um dos dois
    lados.
    """
    contagem = {}
    for r in sorted(rotulos.glob("*.txt")):
        alvo = next((p for p in imagens.glob(f"{r.stem}.*")), None)
        if alvo is not None:
            contagem[r] = sum(1 for l in r.read_text().splitlines() if l.strip())
    if not contagem:
        return []
    mediana = sorted(contagem.values())[len(contagem) // 2]
    melhores = sorted(contagem, key=lambda r: (abs(contagem[r] - mediana), r.stem))
    return [
        (next(imagens.glob(f"{r.stem}.*")), r, contagem[r]) for r in melhores[:quantas]
    ]


def comparison_sheet(cell_width: int, por_lado: int = 4):
    """Pomar real e cena composta lado a lado, ambos com o gabarito desenhado."""
    colecoes = [
        ("pomar real", ROOT / "data/real_yolo_confirmatory/images/train",
         ROOT / "data/real_yolo_confirmatory/labels/train"),
        ("cena composta", ROOT / "data/generated/confirmatory_pool/images/train",
         ROOT / "data/generated/confirmatory_pool/labels/train"),
    ]
    tiles = []
    for legenda, imagens, rotulos in colecoes:
        for caminho, rotulo, n in _medianas(imagens, rotulos, por_lado):
            tiles.append((f"{legenda} · {n} frutas", _com_caixas(caminho, rotulo)))
    if not tiles:
        raise SystemExit("coleções ausentes para a comparação real x sintético")
    return sheet(tiles, por_lado, cell_width)


def save(image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=82, optimize=True, progressive=True, subsampling=1)
    print(f"{path.relative_to(ROOT)} · {image.size[0]}x{image.size[1]} · {path.stat().st_size // 1024} KB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell-width", type=int, default=430)
    args = p.parse_args()
    base = ROOT / "docs/figures/results"

    save(
        detection_sheet(base / "examples", 4, args.cell_width),
        base / "sheets/externo-cena-densa.jpg",
    )
    save(
        detection_sheet(base / "examples/cena-mediana", 4, args.cell_width),
        base / "sheets/externo-cena-mediana.jpg",
    )
    for model in ("yolov8s", "yolo26s", "rtdetr-l"):
        folder = base / "examples/manual-full-val" / model
        if folder.exists():
            save(
                detection_sheet(folder, 4, args.cell_width),
                base / f"sheets/manual-full-val-{model}.jpg",
            )
    save(heatmap_sheet(base / "heatmaps", 3, 300), base / "sheets/mapas-de-anotacoes.jpg")
    save(comparison_sheet(args.cell_width), base / "sheets/real-x-sintetico.jpg")

    scenes = base / "synthetic-examples"
    tiles = []
    for i in (1, 2, 3, 4):
        for suffix, caption in (("", "cena"), ("-boxes", "gabarito")):
            path = scenes / f"scene-{i}{suffix}.jpg"
            if path.exists():
                tiles.append((f"{caption} {i}", Image.open(path).convert("RGB")))
    if tiles:
        save(sheet(tiles, 4, 380), base / "sheets/cenas-sinteticas.jpg")


if __name__ == "__main__":
    main()
