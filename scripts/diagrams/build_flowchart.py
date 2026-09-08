"""Fluxograma contínuo da geração da base sintética.

Fontes de verdade do conteúdo: `configs/synthesis/confirmatory_pool.yaml`,
`configs/pipeline.yaml` e `fruit_pipeline/synthesis.py`.

Sistema de design: skill `diagram-design`, tipo flowchart, detalhe `faithful` com
zonas, skin `tcc-roxo-laranja`, grade de 4 px e conectores ortogonais com cotovelos
arredondados. Fontes e imagens ficam embutidas: os arquivos não dependem de rede.
"""

from __future__ import annotations

import argparse
import base64
import html
from itertools import pairwise
from pathlib import Path

import fontkit
import yaml

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/figures/diagram-assets"
CONFIG = yaml.safe_load((ROOT / "configs/synthesis/confirmatory_pool.yaml").read_text())
PIPELINE = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text())


def number(value):
    return f"{value:g}"


# --- tokens (skin tcc-roxo-laranja) -------------------------------------------------
PAPER = "#f9f6fc"
PAPER2 = "#efe7fa"
INK = "#2b1b46"
MUTED = "#6d5a92"
SOFT = "#77688f"
RULE = "rgba(43,27,70,0.14)"
ACCENT = "#ffa629"  # laranja do TCC — só preenchimento, nunca traço ou texto
ACCENT_TINT = "rgba(255,166,41,0.30)"
DECISION = "#f2cc60"
WHITE = "#ffffff"

SANS = "'Geist', sans-serif"
MONO = "'Geist Mono', monospace"
SERIF = "'Instrument Serif', serif"

FS_NAME, FS_SUB, FS_LABEL = 20, 16, 16
LINE_NAME, LINE_SUB = 26, 22

W = 1152
ZX, ZW = 32, 1088
SECTION_GAP = 24


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def ortho(points: list[tuple[float, float]], radius: float = 8) -> str:
    """Caminho ortogonal em que cada dobra vira um quarto de arco."""
    for (ax, ay), (bx, by) in pairwise(points):
        if ax != bx and ay != by:
            raise ValueError(
                f"segmento diagonal em ({ax}, {ay}) → ({bx}, {by}): "
                "conectores precisam ser ortogonais"
            )
    parts = [f"M {points[0][0]:g} {points[0][1]:g}"]
    for i in range(1, len(points) - 1):
        (px, py), (cx, cy), (nx, ny) = points[i - 1], points[i], points[i + 1]
        dx1, dy1 = sign(cx - px), sign(cy - py)
        dx2, dy2 = sign(nx - cx), sign(ny - cy)
        span = min(
            radius, (abs(cx - px) + abs(cy - py)) / 2, (abs(nx - cx) + abs(ny - cy)) / 2
        )
        sweep = 1 if dx1 * dy2 - dy1 * dx2 > 0 else 0
        parts.append(f"L {cx - dx1 * span:g} {cy - dy1 * span:g}")
        parts.append(
            f"A {span:g} {span:g} 0 0 {sweep} {cx + dx2 * span:g} {cy + dy2 * span:g}"
        )
    parts.append(f"L {points[-1][0]:g} {points[-1][1]:g}")
    return " ".join(parts)


class Canvas:
    """Acumula um trecho do SVG e o conjunto de glifos usado por cada família."""

    def __init__(self, slug: str, title: str, desc: str, height: int):
        self.slug, self.title, self.desc, self.height = slug, title, desc, height
        self.parts: list[str] = []
        self.glyphs: dict[str, set[str]] = {SANS: set(), MONO: set(), SERIF: set()}

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def text(
        self,
        x,
        y,
        content,
        *,
        family,
        size,
        fill,
        weight=None,
        anchor="middle",
        tracking=None,
    ):
        self.glyphs[family].update(content)
        extra = f' font-weight="{weight}"' if weight else ""
        extra += f' letter-spacing="{tracking}"' if tracking else ""
        self.add(
            f'<text x="{x:g}" y="{y:g}" fill="{fill}" font-size="{size}" '
            f'font-family="{family}"{extra} text-anchor="{anchor}">{esc(content)}</text>'
        )


# ====================================================================================
# Primitivas
# ====================================================================================
def arrow(c: Canvas, points, *, dashed=False) -> None:
    attrs = f'fill="none" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#arrow)"'
    if dashed:
        attrs += ' stroke-dasharray="6,5"'
    c.add(f'<path d="{ortho(points)}" {attrs}/>')


def arrow_label(c: Canvas, x, y, content, *, anchor="middle", bg=PAPER2) -> None:
    """Rótulo de seta com máscara opaca; o chamador garante a folga de 6–10 px."""
    # Largura em múltiplos de 8: máscara, centro e texto permanecem na grade de 4 px.
    width = ((round(len(content) * 10 + 20) + 7) // 8) * 8
    left = {"middle": x - width / 2, "start": x - 8, "end": x - width + 8}[anchor]
    c.add(
        f'<rect x="{left:g}" y="{y - 16:g}" width="{width:g}" height="20" rx="4" fill="{bg}"/>'
    )
    c.text(
        x,
        y,
        content,
        family=MONO,
        size=FS_LABEL,
        fill=SOFT,
        anchor=anchor,
        tracking="0.06em",
    )


def zone(c: Canvas, x, y, w, h, label, *, loop=False) -> None:
    c.add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PAPER2}" '
        f'stroke="{RULE}" stroke-width="1"/>'
    )
    if loop:
        sync_icon(c, x + 24, y + 16)
    c.text(
        x + (64 if loop else 28),
        y + 34,
        label,
        family=MONO,
        size=FS_LABEL,
        fill=SOFT,
        anchor="start",
        tracking="0.18em",
    )


def frame(c: Canvas, x, y, w, h, label, *, loop=False) -> None:
    """Contêiner interno (laço por fruta): fundo branco sobre a zona lilás."""
    c.add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
        f'stroke="{RULE}" stroke-width="1"/>'
    )
    if loop:
        sync_icon(c, x + 24, y + 16)
    c.text(
        x + (64 if loop else 28),
        y + 34,
        label,
        family=MONO,
        size=FS_LABEL,
        fill=SOFT,
        anchor="start",
        tracking="0.18em",
    )


def block(c: Canvas, cx, cy, names, subs=()) -> None:
    total = len(names) * LINE_NAME + len(subs) * LINE_SUB
    y = cy - total / 2 + 19
    for line in names:
        c.text(cx, y, line, family=SANS, size=FS_NAME, fill=INK, weight="600")
        y += LINE_NAME
    y += 2
    for line in subs:
        c.text(cx, y, line, family=MONO, size=FS_SUB, fill=MUTED)
        y += LINE_SUB


def node(
    c: Canvas, x, y, w, h, names, subs=(), *, accent=False, radius=6, bg=PAPER2
) -> None:
    c.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{bg}"/>')
    fill = ACCENT_TINT if accent else WHITE
    c.add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    block(c, x + w / 2, y + h / 2, names, subs)


def oval(c: Canvas, cx, cy, rx, ry, label, *, bg=PAPER2) -> None:
    c.add(
        f'<rect x="{cx - rx}" y="{cy - ry}" width="{2 * rx}" height="{2 * ry}" rx="{ry}" fill="{bg}"/>'
    )
    c.add(
        f'<rect x="{cx - rx}" y="{cy - ry}" width="{2 * rx}" height="{2 * ry}" rx="{ry}" '
        f'fill="{WHITE}" stroke="{INK}" stroke-width="1.2"/>'
    )
    c.text(cx, cy + 7, label, family=SANS, size=FS_NAME, fill=INK, weight="600")


def diamond(c: Canvas, cx, cy, hw, hh, lines, *, bg=WHITE) -> None:
    pts = f"{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}"
    c.add(f'<polygon points="{pts}" fill="{bg}"/>')
    c.add(
        f'<polygon points="{pts}" fill="{DECISION}" stroke="{INK}" stroke-width="1.2"/>'
    )
    y = cy - (len(lines) - 1) * 13 + 7
    for line in lines:
        c.text(cx, y, line, family=SANS, size=FS_NAME, fill=INK, weight="600")
        y += LINE_NAME


def document(c: Canvas, x, y, w, h, name, sub, *, accent=False, bg=PAPER2) -> None:
    """Retângulo com o canto superior direito dobrado — arquivo gravado em disco."""
    fold = 20
    path = (
        f"M {x + 6} {y} L {x + w - fold} {y} L {x + w} {y + fold} L {x + w} {y + h - 6} "
        f"A 6 6 0 0 1 {x + w - 6} {y + h} L {x + 6} {y + h} A 6 6 0 0 1 {x} {y + h - 6} "
        f"L {x} {y + 6} A 6 6 0 0 1 {x + 6} {y} Z"
    )
    c.add(f'<path d="{path}" fill="{bg}"/>')
    fill = ACCENT_TINT if accent else WHITE
    c.add(f'<path d="{path}" fill="{fill}" stroke="{INK}" stroke-width="1.2"/>')
    c.add(
        f'<path d="M {x + w - fold} {y} L {x + w - fold} {y + fold} L {x + w} {y + fold}" '
        f'fill="none" stroke="{INK}" stroke-width="1.2"/>'
    )
    block(c, x + w / 2, y + h / 2, [name], [sub])


def photo(
    c: Canvas, name, x, y, w, h, caption: str | tuple[str, ...], *, above=False
) -> None:
    data = base64.b64encode((ASSETS / f"{name}.png").read_bytes()).decode()
    c.add(
        f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{data}"/>'
    )
    lines = (caption,) if isinstance(caption, str) else caption
    if above:
        first_y = y - 16 - (len(lines) - 1) * 18
    else:
        first_y = y + h + (24 if len(lines) == 1 else 14)
    for index, line in enumerate(lines):
        c.text(
            x + w / 2,
            first_y + index * 18,
            line,
            family=MONO,
            size=FS_SUB,
            fill=SOFT,
        )


def dot(c: Canvas, cx, cy) -> None:
    c.add(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{INK}"/>')


def sync_icon(c: Canvas, x, y, size=24, color=SOFT) -> None:
    """Tabler `refresh` (MIT), do catálogo de ícones da skill: marca um contêiner cíclico."""
    c.add(
        f'<g transform="translate({x} {y}) scale({size / 24:g})" fill="none" stroke="{color}" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M20 11a8.1 8.1 0 0 0 -15.5 -2m-.5 -4v4h4"/>'
        '<path d="M4 13a8.1 8.1 0 0 0 15.5 2m.5 4v-4h-4"/></g>'
    )


def illustrated(c: Canvas, x, y, w, h, band, names, subs=(), *, band_h=64) -> None:
    """Nó com uma faixa ilustrativa no topo: a etapa mostra o que ela produz."""
    clip = f"{c.slug}-{band}"
    source = ASSETS / f"{band}.jpg"
    mime = "jpeg" if source.exists() else "png"
    if mime == "png":
        source = ASSETS / f"{band}.png"
    data = base64.b64encode(source.read_bytes()).decode()
    c.add(
        f'<clipPath id="{clip}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/></clipPath>'
    )
    c.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{WHITE}"/>')
    c.add(
        f'<g clip-path="url(#{clip})"><image x="{x}" y="{y}" width="{w}" height="{band_h}" '
        f'preserveAspectRatio="xMidYMid slice" href="data:image/{mime};base64,{data}"/></g>'
    )
    if band == "loop_xy":
        # Vector overlay remains legible when the README scales the figure.
        cx, cy = x + w / 2, y + band_h / 2
        path = f"M {cx - 12} {cy} H {cx + 12} M {cx} {cy - 12} V {cy + 12}"
        for color, width in ((INK, 6), (WHITE, 4), (ACCENT, 2)):
            c.add(
                f'<path d="{path}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" stroke-linecap="round"/>'
            )
    c.add(
        f'<line x1="{x}" y1="{y + band_h}" x2="{x + w}" y2="{y + band_h}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    c.add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    block(c, x + w / 2, y + band_h + (h - band_h) / 2, names, subs)


def legend(c: Canvas, y, items) -> None:
    c.add(
        f'<line x1="{ZX}" y1="{y}" x2="{W - ZX}" y2="{y}" stroke="{RULE}" stroke-width="0.8"/>'
    )
    c.text(
        ZX,
        y + 30,
        "Legend",
        family=MONO,
        size=FS_LABEL,
        fill=MUTED,
        anchor="start",
        tracking="0.14em",
    )
    x = ZX + 148
    for glyph, label in items:
        c.add(glyph(x, y + 24))
        c.text(
            x + 40, y + 30, label, family=MONO, size=FS_SUB, fill=MUTED, anchor="start"
        )
        x += 40 + len(label) * 9.9 + 40


def g_step(x, y):
    return (
        f'<rect x="{x}" y="{y - 11}" width="28" height="20" rx="4" fill="{WHITE}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )


def g_decision(x, y):
    return (
        f'<polygon points="{x + 14},{y - 13} {x + 28},{y - 1} {x + 14},{y + 11} {x},{y - 1}" '
        f'fill="{DECISION}" stroke="{INK}" stroke-width="1.2"/>'
    )


def g_file(x, y):
    return (
        f'<path d="M {x} {y - 11} L {x + 20} {y - 11} L {x + 28} {y - 3} L {x + 28} {y + 9} '
        f'L {x} {y + 9} Z" fill="{WHITE}" stroke="{INK}" stroke-width="1.2"/>'
    )


def g_terminal(x, y):
    return (
        f'<rect x="{x}" y="{y - 11}" width="28" height="20" rx="10" fill="{WHITE}" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )


def g_dashed(x, y):
    return (
        f'<line x1="{x}" y1="{y - 1}" x2="{x + 28}" y2="{y - 1}" stroke="{MUTED}" '
        f'stroke-width="1.6" stroke-dasharray="6,5"/>'
    )


# ====================================================================================
# Aquisição e pré-processamento
# ====================================================================================
def panel_acquisition() -> Canvas:
    c = Canvas(
        "aquisicao",
        "Prepare fruit cutouts and tree proximity maps",
        "Fotos de frutas em fundo uniforme e de árvores fora do período produtivo são "
        "segmentadas e convertidas em mapas de profundidade DepthPro; todos os ativos "
        "regenerados permanecem disponíveis durante a composição do pool sintético.",
        700,
    )
    RAIL, RAIL_L, RAIL_R = 140, 56, 956
    COL_L, COL_R, COL_W = 228, 596, 328
    CX_L, CX_R, CENTER = COL_L + COL_W // 2, COL_R + COL_W // 2, W // 2
    A_Y, A_H = 204, 136
    B_Y, B_H = 476, 158

    # O terminal pertence ao fluxo completo, não ao agrupamento de aquisição.
    # A zona começa abaixo dele e recebe apenas as duas etapas de captura.
    zone(c, ZX, 140, ZW, 240, "Image collection")
    zone(c, ZX, 140 + 240 + SECTION_GAP, ZW, 296, "Asset preparation")

    # setas — ramos espelhados em torno do eixo central
    arrow(c, [(CENTER - 24, 98), (CENTER - 24, 120), (CX_L, 120), (CX_L, A_Y)])
    arrow(c, [(CENTER + 24, 98), (CENTER + 24, 120), (CX_R, 120), (CX_R, A_Y)])
    arrow(c, [(COL_L, A_Y + A_H / 2), (RAIL_L + RAIL, A_Y + A_H / 2)], dashed=True)
    arrow(c, [(COL_R + COL_W, A_Y + A_H / 2), (RAIL_R, A_Y + A_H / 2)], dashed=True)
    arrow(c, [(CX_L, A_Y + A_H), (CX_L, B_Y)])
    arrow(c, [(CX_R, A_Y + A_H), (CX_R, B_Y)])
    arrow(c, [(COL_L, B_Y + B_H / 2), (RAIL_L + RAIL, B_Y + B_H / 2)], dashed=True)
    arrow(c, [(COL_R + COL_W, B_Y + B_H / 2), (RAIL_R, B_Y + B_H / 2)], dashed=True)
    # As pilhas não são só ilustração: seus rótulos identificam os artefatos que
    # alimentam a etapa seguinte. Cada guia começa sob o texto e entra perto do
    # centro da borda superior da célula receptora.
    arrow(
        c,
        [(RAIL_L + RAIL / 2, 376), (RAIL_L + RAIL / 2, 392), (376, 392), (376, B_Y)],
        dashed=True,
    )
    arrow(
        c,
        [(RAIL_R + RAIL / 2, 376), (RAIL_R + RAIL / 2, 392), (776, 392), (776, B_Y)],
        dashed=True,
    )
    # Os dois ramos convergem diretamente na geração; um bloco intermediário de
    # "reunir ativos" apenas repetiria uma relação já expressa pelas setas.
    c.add(
        f'<path d="{ortho([(CX_L, B_Y + B_H), (CX_L, 668), (CENTER, 668)])}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.6"/>'
    )
    c.add(
        f'<path d="{ortho([(CX_R, B_Y + B_H), (CX_R, 668), (CENTER, 668)])}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.6"/>'
    )

    oval(c, CENTER, 68, 64, 30, "Start")
    node(
        c,
        COL_L,
        A_Y,
        COL_W,
        A_H,
        ["Photograph fruit"],
        ["127 plain-background photos"],
    )
    node(
        c,
        COL_R,
        A_Y,
        COL_W,
        A_H,
        ["Photograph trees"],
        ["228 photos without fruit"],
    )
    node(
        c,
        COL_L,
        B_Y,
        COL_W,
        B_H,
        ["Remove fruit backgrounds"],
        ["rembg / ISNet, RGBA cutouts", "Alpha defines visible fruit"],
    )
    node(
        c,
        COL_R,
        B_Y,
        COL_W,
        B_H,
        ["Estimate depth"],
        ["DepthPro, 8-bit proximity maps", "Higher values mean nearer"],
    )
    dot(c, CENTER, 668)

    photo(c, "stack_frutas", RAIL_L, 200, RAIL, RAIL, "fruit photos")
    photo(c, "stack_arvores", RAIL_R, 200, RAIL, RAIL, "tree photos")
    photo(c, "stack_recortes", RAIL_L, 484, RAIL, RAIL, "RGBA cutouts")
    photo(c, "stack_mapas", RAIL_R, 484, RAIL, RAIL, "proximity maps")

    return c


# ====================================================================================
# Geração das cenas e dos rótulos
# ====================================================================================
def panel_generation() -> Canvas:
    c = Canvas(
        "geracao",
        "Depth-guided composition and nested datasets",
        "Para cada cena, um fundo, seu mapa e os recortes são amostrados do catálogo completo; cada "
        "fruta é inserida por tentativa e erro guiada pela profundidade; encerrada a inserção, "
        "as caixas são extraídas, as 1.300 cenas são divididas em train/val e delas saem os "
        "subconjuntos aninhados synthetic-1x, 2x, 3x, 5x e 10x.",
        1740,
    )
    CENTER = W // 2
    N_Y, N_H = 96, 120
    # O contêiner é centrado na zona (32..1120) e as colunas deixam 32 px de folga
    # em cada lado, com os canais de retorno em 80 e 1072 — o laço fica simétrico.
    IN_X, IN_Y, IN_W, IN_H = 48, 264, 1056, 480
    C1, C2, C3, CW = 112, 432, 752, 288
    DOT_X, RETURN_X = 80, 1072
    RA_Y, RB_Y, RH = 392, 568, 144
    RA_CY, RB_CY = RA_Y + RH // 2, RB_Y + RH // 2
    DIA_CX, DIA_HW, DIA_HH = C3 + CW // 2, CW // 2, RH // 2
    SAVE_X, SAVE_Y, SAVE_W, SAVE_H = 176, 812, 456, 112
    CARD_Y, CARD_W, CARD_H = 800, 84, 108
    STACK_X = (688, 808, 928)
    SPLIT_X, SPLIT_Y, SPLIT_W, SPLIT_H = 296, 1056, 560, 80
    FINAL_Y, FINAL_W, FINAL_H = 1216, 184, 96
    FINAL_X = (48, 266, 484, 702, 920)

    zone(
        c,
        ZX,
        32,
        ZW,
        1272,
        f"Generate {CONFIG['images']['total']:,} scenes".replace(",", ","),
        loop=True,
    )
    zone(c, ZX, 32 + 1272 + SECTION_GAP, ZW, 328, "Scene split and nested subsets")
    c.add('<g transform="translate(0 320)">')
    frame(c, IN_X, IN_Y, IN_W, IN_H, "For each sampled fruit", loop=True)

    arrow(
        c, [(CENTER, N_Y + N_H), (CENTER, 336), (C1 + CW / 2, 336), (C1 + CW / 2, RA_Y)]
    )

    arrow(c, [(C1 + CW, RA_CY), (C2, RA_CY)])
    arrow(c, [(C2 + CW, RA_CY), (C3, RA_CY)])
    arrow(c, [(DIA_CX, RA_Y + RH), (DIA_CX, RB_CY - DIA_HH)])

    # NÃO: nova coordenada para a mesma fruta; depois de 100 tentativas ela é descartada
    arrow(
        c,
        [
            (DIA_CX + DIA_HW, RB_CY),
            (RETURN_X, RB_CY),
            (RETURN_X, 360),
            (C2 + CW // 2, 360),
            (C2 + CW // 2, RA_Y),
        ],
    )
    arrow_label(c, 876, 340, "No: retry X, Y", bg=WHITE)
    c.text(
        1072,
        298,
        f"Skip fruit after {CONFIG['placement']['max_attempts_per_object']} failed attempts",
        family=MONO,
        size=16,
        fill=MUTED,
        anchor="end",
    )

    # SIM: a fruta é ajustada e composta
    arrow(c, [(DIA_CX - DIA_HW, RB_CY), (C2 + CW, RB_CY)])
    arrow_label(c, C3, RB_CY - 20, "Yes", bg=WHITE)
    arrow(c, [(C2, RB_CY), (C1 + CW, RB_CY)])
    arrow(c, [(C1, RB_CY), (DOT_X, RB_CY), (DOT_X, RA_CY), (C1, RA_CY)])

    save_center = SAVE_X + SAVE_W / 2
    arrow(
        c,
        [
            (CENTER, IN_Y + IN_H),
            (CENTER, 792),
            (save_center, 792),
            (save_center, SAVE_Y),
        ],
    )
    arrow(
        c,
        [
            (save_center, SAVE_Y + SAVE_H),
            (save_center, 996),
            (544, 996),
            (544, SPLIT_Y),
        ],
    )
    # Um único barramento tracejado agrupa os três artefatos sem criar outro
    # contêiner visual dentro da seção.
    stack_centers = tuple(x + CARD_W / 2 for x in STACK_X)
    c.add(
        f'<path d="{ortho([(SAVE_X + SAVE_W, SAVE_Y + SAVE_H / 2), (656, SAVE_Y + SAVE_H / 2), (656, 784), (stack_centers[-1], 784)])}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="6,5"/>'
    )
    for center in stack_centers:
        arrow(c, [(center, 784), (center, CARD_Y)], dashed=True)

    # O split consome os três artefatos efetivamente gravados, não o bloco de
    # salvamento de forma abstrata. As guias descem dos rótulos até um único
    # barramento no vão entre as seções, que entra no centro da célula.
    artifact_bus_y = 996
    for center in stack_centers:
        c.add(
            f'<path d="M {center:g} 952 L {center:g} {artifact_bus_y}" '
            f'fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="6,5"/>'
        )
    c.add(
        f'<path d="M {CENTER} {artifact_bus_y} L {stack_centers[-1]} {artifact_bus_y}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="6,5"/>'
    )
    arrow(c, [(CENTER, artifact_bus_y), (CENTER, SPLIT_Y)], dashed=True)

    final_centers = tuple(x + FINAL_W // 2 for x in FINAL_X)
    final_bus_y = 1192
    c.add(
        f'<path d="M {CENTER} {SPLIT_Y + SPLIT_H} L {CENTER} {final_bus_y}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.6"/>'
    )
    c.add(
        f'<path d="M {final_centers[0]} {final_bus_y} L {final_centers[-1]} {final_bus_y}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.6"/>'
    )
    for center in final_centers:
        arrow(c, [(center, final_bus_y), (center, FINAL_Y)])

    illustrated(
        c,
        C1,
        RA_Y,
        CW,
        RH,
        "loop_escala",
        ["Sample size and rotation"],
        [
            f"{number(CONFIG['objects']['min_scale'])}–{number(CONFIG['objects']['max_scale'])} / rotation ±{CONFIG['objects']['rotation_degrees']}°"
        ],
    )
    illustrated(
        c,
        C2,
        RA_Y,
        CW,
        RH,
        "loop_xy",
        ["Sample X, Y"],
        ["Exclude the bottom 15%"],
    )
    illustrated(
        c,
        C3,
        RA_Y,
        CW,
        RH,
        "loop_profundidade",
        ["Apply local depth"],
        ["Depth scaling and Z jitter"],
    )
    diamond(
        c,
        DIA_CX,
        RB_CY,
        DIA_HW,
        DIA_HH,
        [
            f"Proximity ≥ {CONFIG['placement']['min_depth']}?",
            f"Visibility ≥ {CONFIG['placement']['min_visibility']:.0%}?",
        ],
    )
    illustrated(
        c,
        C2,
        RB_Y,
        CW,
        RH,
        "loop_aparencia",
        ["Adjust fruit appearance"],
        ["Hue, light and shadows"],
    )
    illustrated(
        c,
        C1,
        RB_Y,
        CW,
        RH + 16,
        "loop_compor",
        ["Composite the fruit"],
        ["Update visibility masks", "of previously placed fruit"],
    )

    node(
        c,
        SAVE_X,
        SAVE_Y,
        SAVE_W,
        SAVE_H,
        ["Extract final boxes and save"],
        ["After all insertions, omit boxes below 2 px"],
    )
    photo(
        c,
        "stack_cenas",
        STACK_X[0],
        CARD_Y,
        CARD_W,
        CARD_H,
        ("scene", "JPG"),
    )
    photo(
        c,
        "stack_caixas",
        STACK_X[1],
        CARD_Y,
        CARD_W,
        CARD_H,
        ("YOLO boxes", "TXT"),
    )
    photo(
        c,
        "stack_metadados",
        STACK_X[2],
        CARD_Y,
        CARD_W,
        CARD_H,
        ("provenance", "JSON"),
    )
    node(
        c,
        SPLIT_X,
        SPLIT_Y,
        SPLIT_W,
        SPLIT_H,
        [f"Split {CONFIG['images']['total']:,} scenes".replace(",", ",")],
        ["80% train, 20% validation, shared source assets"],
        radius=20,
        bg=PAPER,
    )

    for x, center, multiplier, train_count, val_count in zip(
        FINAL_X,
        final_centers,
        PIPELINE["synthetic_subsets"]["multipliers"],
        [
            m * PIPELINE["synthetic_subsets"]["base_train_images"]
            for m in PIPELINE["synthetic_subsets"]["multipliers"]
        ],
        [
            m * PIPELINE["synthetic_subsets"]["base_val_images"]
            for m in PIPELINE["synthetic_subsets"]["multipliers"]
        ],
        strict=True,
    ):
        node(
            c,
            x,
            FINAL_Y,
            FINAL_W,
            FINAL_H,
            [f"synthetic-{multiplier}x"],
            [
                f"{train_count:,}".replace(",", ",") + " train",
                f"{val_count} validation",
            ],
            radius=20,
            bg=PAPER,
        )

    legend(
        c,
        1360,
        [
            (g_step, "process"),
            (g_decision, "decision"),
            (g_terminal, "start / dataset"),
            (g_dashed, "artifact flow"),
        ],
    )

    c.add("</g>")
    node(
        c,
        296,
        96,
        560,
        120,
        ["Prepare the scene"],
        [
            "Sample background and map; grade and smooth",
            "RNG: seed, recipe hash, assets, scene index",
        ],
    )
    arrow(c, [(CENTER, 216), (CENTER, 248)])
    diamond(c, CENTER, 304, 128, 56, ["Dense scene?"], bg=DECISION)
    arrow(c, [(448, 304), (280, 304), (280, 384)])
    arrow(c, [(704, 304), (872, 304), (872, 384)])
    probability = CONFIG["objects"]["dense"]["probability"]
    arrow_label(c, 338, 284, f"No ({1 - probability:.0%})")
    arrow_label(c, 814, 284, f"Yes ({probability:.0%})")
    node(
        c,
        128,
        384,
        304,
        80,
        ["Sample fruit count"],
        [f"{CONFIG['objects']['min']} to {CONFIG['objects']['max']} fruit"],
    )
    node(
        c,
        720,
        384,
        304,
        80,
        ["Sample fruit count"],
        [
            f"{CONFIG['objects']['dense']['min']} to {CONFIG['objects']['dense']['max']} fruit"
        ],
    )
    for x in (280, 872):
        c.add(
            f'<path d="{ortho([(x, 464), (x, 520), (CENTER, 520)])}" '
            f'fill="none" stroke="{MUTED}" stroke-width="1.6"/>'
        )
    dot(c, CENTER, 520)
    c.add(
        f'<path d="M {CENTER} 520 L {CENTER} 536" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.6"/>'
    )

    return c


# ====================================================================================
# Emissão
# ====================================================================================
PAGE_TITLE = "Synthetic orchard dataset generation"


def font_css(panels: list[Canvas]) -> str:
    sans: set[str] = set()
    mono: set[str] = set()
    for panel in panels:
        sans |= panel.glyphs[SANS]
        mono |= panel.glyphs[MONO]
    return (
        fontkit.face("Geist", "Geist:wght@600", sans, "geist-600", weight="600")
        + fontkit.face(
            "Geist Mono", "Geist Mono:wght@400", mono, "geist-mono-400", weight="400"
        )
        + fontkit.face(
            "Instrument Serif",
            "Instrument Serif:ital@0",
            set(PAGE_TITLE),
            "instrument-serif-400",
            weight="400",
        )
    )


def combined_svg(panels: list[Canvas], css: str) -> str:
    """Um único SVG alto, composto como um fluxo visual ininterrupto."""
    panel_1_y = 104
    # As três transições entre seções usam o mesmo vão. Como a geração começa 32 px
    # dentro do painel 2, seu deslocamento compensa essa margem sem sobrepor fundos.
    panel_2_y = panel_1_y + panels[0].height + SECTION_GAP - 32
    height = panel_2_y + panels[1].height + 32

    groups = []
    for panel, offset in zip(panels, (panel_1_y, panel_2_y), strict=True):
        groups.append(
            f'<g role="group" aria-label="{esc(panel.title)}" transform="translate(0 {offset})">\n'
            + "\n".join(panel.parts)
            + "\n</g>"
        )

    # O catálogo é a entrada direta da composição, no mesmo eixo vertical.
    bridge = ortho(
        [(W / 2, panel_1_y + panels[0].height - 32), (W / 2, panel_2_y + 96)]
    )

    # Recortes e mapas são o insumo direto da amostragem. As guias cruzam a fronteira
    # entre os painéis, então são desenhadas aqui, depois dos dois grupos, e não dentro
    # do painel 1 — onde a zona seguinte passaria por cima delas.
    # Os rótulos identificam os artefatos que alimentam a seleção. As guias
    # começam sob o texto, descem antes de desviar e chegam agrupadas ao redor
    # do eixo da célula.
    feed_start_y = panel_1_y + 660
    feed_bus_y = panel_1_y + panels[0].height + SECTION_GAP / 2
    select_top = panel_2_y + 96
    feeds = [
        ortho(
            [
                (126, feed_start_y),
                (126, feed_bus_y),
                (560, feed_bus_y),
                (560, select_top),
            ]
        ),
        ortho(
            [
                (1026, feed_start_y),
                (1026, feed_bus_y),
                (592, feed_bus_y),
                (592, select_top),
            ]
        ),
    ]

    return (
        f'<svg viewBox="0 0 {W} {height}" role="img" '
        'aria-labelledby="fluxograma-sintese-v3-title fluxograma-sintese-v3-desc" '
        'xmlns="http://www.w3.org/2000/svg">\n'
        f'<title id="fluxograma-sintese-v3-title">{esc(PAGE_TITLE)}</title>\n'
        '<desc id="fluxograma-sintese-v3-desc">Prepare source assets, compose scenes using depth, '
        "split generated scenes and materialize "
        "nested synthetic-1x to synthetic-10x datasets.</desc>\n"
        "<defs>\n<style>" + css + "</style>\n"
        f'<marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 9 3.5, 0 7" fill="{MUTED}"/></marker>\n'
        "</defs>\n"
        f'<rect width="{W}" height="{height}" fill="{PAPER}"/>\n'
        f'<text x="{ZX}" y="58" fill="{INK}" font-size="40" font-family="{SERIF}" '
        f'font-weight="400">{esc(PAGE_TITLE)}</text>\n'
        f'<line x1="{ZX}" y1="88" x2="{W - ZX}" y2="88" stroke="{RULE}" stroke-width="0.8"/>\n'
        + groups[0]
        + "\n"
        + groups[1]
        + "\n"
        + f'<path d="{bridge}" fill="none" stroke="{MUTED}" stroke-width="1.6" '
        'marker-end="url(#arrow)"/>\n'
        + "\n".join(
            f'<path d="{feed}" fill="none" stroke="{MUTED}" stroke-width="1.6" '
            'stroke-dasharray="6,5" marker-end="url(#arrow)"/>'
            for feed in feeds
        )
        + "\n</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "artifacts/diagrams")
    args = parser.parse_args()
    panels = [panel_acquisition(), panel_generation()]
    css = font_css(panels)
    markup = combined_svg(panels, css)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(PAGE_TITLE)}</title>
<style>
{css}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: {PAPER}; }}
  body {{ padding: 2rem; }}
  main {{ max-width: {W}px; margin: 0 auto; }}
  svg {{ display: block; width: 100%; height: auto; }}
</style>
</head>
<body>
<main>{markup}</main>
</body>
</html>
"""
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    out = args.preview_dir / "fluxograma-geracao-conjuntos-sinteticos.html"
    svg_out = ROOT / "docs/figures/fluxograma-geracao-conjuntos-sinteticos.svg"
    svg_out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + markup + "\n", encoding="utf-8"
    )
    out.write_text(page, encoding="utf-8")
    box = markup.split('viewBox="0 0 ', 1)[1].split('"', 1)[0]
    print(f"{out.name} · {box.replace(' ', '×')} · {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
