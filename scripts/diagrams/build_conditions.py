"""Um micro-fluxograma por condição de treinamento, para caber numa célula da tabela.

Cada arquivo mostra, da esquerda para a direita, os três conjuntos que a condição
usa — treino, validação e o teste externo — com quadros reais de cada um. A
profundidade do baralho de treino cresce com o volume da condição, então a coluna
inteira se lê como uma escala.

Fontes de verdade: `configs/confirmatory.yaml` (diretórios de cada condição),
`configs/pipeline.yaml` (104/26 reais, 80/20 do controlled) e
`fruit_pipeline/synthesis.py::materialize_nested_subsets` (base 104/26 dos sintéticos).
"""

from __future__ import annotations

import base64
from pathlib import Path

import fontkit
from build_flowchart import MONO, MUTED, PAPER, PIPELINE, esc

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/figures/diagram-assets"
OUT = ROOT / "docs/figures/condicoes"

# Cartão e moldura branca em volta da foto. O deslocamento é só horizontal: o
# baralho cresce para a direita, sem cascata na diagonal.
CARD_W, CARD_H, CARD_PAD = 64, 64, 2
# Vão largo o bastante para a seta ler como seta ao lado de um baralho de 176 px.
GAP = 88
FS = 12

# O baralho ocupa exatamente o que suas cartas pedem, com a folga fixa em MAX_OFF,
# até bater no teto da coluna; daí em diante a folga é que encolhe. A seta fica com
# a sobra, então um conjunto pequeno tem baralho curto e seta longa.
MAX_OFF = 16
# Tetos distintos: a mesma escala vale para as duas colunas, mas o treino chega a
# 120 cartas e a validação a 30. Espremer as 120 no teto da validação daria 1,9 px
# por carta — a borda comeria a foto e o baralho viraria uma mancha. Abaixo de
# ~3,5 px a carta deixa de ler como carta, e é isso que fixa o teto do treino.
TRAIN_MAX_DECK, VAL_MAX_DECK = 520, 288
# A seta tem comprimento fixo e cada linha flui da esquerda para a direita a partir
# das larguras reais dos baralhos. Colunas fixas obrigariam a seta do manual-full a
# esticar 304 px para alcançar a validação; assim a própria extensão da linha fica
# proporcional ao volume da condição.
ARROW = 40
GUTTER = 14  # folga entre baralho e seta
TEST_CARDS_WIDTH = 140  # o teste não escala: são sempre as mesmas 119 imagens

DECK_Y, LABEL_Y = 6, 88
HEIGHT = 100
TRAIN_X = 4


def row_extent(train: int, val: int) -> tuple[float, float, float, float]:
    """Posições x de cada baralho na linha, e onde ela termina."""
    train_w = CARD_W + (deck_size(train) - 1) * card_offset(
        deck_size(train), deck_width(deck_size(train), TRAIN_MAX_DECK)
    )
    val_w = CARD_W + (deck_size(val) - 1) * card_offset(
        deck_size(val), deck_width(deck_size(val), VAL_MAX_DECK)
    )
    val_x = TRAIN_X + train_w + GUTTER + ARROW + GUTTER
    test_x = val_x + val_w + GUTTER + ARROW + GUTTER
    return train_w, val_x, test_x, test_x + TEST_CARDS_WIDTH + 4


def deck_width(cards: int, ceiling: int) -> int:
    """A largura que as cartas pedem com folga MAX_OFF, limitada ao teto da coluna."""
    return min(ceiling, CARD_W + (cards - 1) * MAX_OFF)


def card_offset(cards: int, width: int) -> float:
    """Folga fracionária: presa a inteiros, um baralho de 30 cartas só conseguiria
    5 px (209 de largura) ou 6 px (238), e o 6 não deixaria seta nenhuma."""
    if cards < 2:
        return 0.0
    return max(3.5, (width - CARD_W) / (cards - 1))


def pool_size(slug: str) -> int:
    """Quantos quadros distintos existem para esse conjunto."""
    return len(list(ASSETS.glob(f"cell_{slug}_*.jpg")))


# nome, treino, validação, conjunto de quadros
CONDITIONS = [
    (
        "manual-full",
        PIPELINE["real_dataset"]["train_images"],
        PIPELINE["real_dataset"]["val_images"],
        "manual",
    ),
    (
        "controlled",
        PIPELINE["controlled_dataset"]["train_images"],
        PIPELINE["controlled_dataset"]["val_images"],
        "controlled",
    ),
    *[
        (
            f"synthetic-{m}x",
            m * PIPELINE["synthetic_subsets"]["base_train_images"],
            m * PIPELINE["synthetic_subsets"]["base_val_images"],
            "synthetic",
        )
        for m in PIPELINE["synthetic_subsets"]["multipliers"]
    ],
]

TEST_IMAGES = PIPELINE["external_datasets"]["citdet"]["expected_images"]

# O teste não escala com nada: é o mesmo conjunto nas sete condições.
TEST_CARDS = 3


# Uma unidade só para as duas colunas: 26 imagens (a validação do `1x`, o menor
# conjunto do experimento) valem 3 cartas. Medir treino e validação contra bases
# diferentes fazia os dois baralhos empatarem, escondendo que a validação é um
# quarto do treino.
UNIT_IMAGES, UNIT_CARDS = 26, 3


def deck_size(count: int) -> int:
    """Cartas proporcionais ao volume, na mesma escala para treino e validação."""
    return max(1, round(UNIT_CARDS * count / UNIT_IMAGES))


WIDTH = max(row_extent(train, val)[3] for _, train, val, _ in CONDITIONS)


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


class Micro:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.defs: dict[str, str] = {}
        self.glyphs: set[str] = set()

    def cell(self, slug: str, index: int) -> str:
        ident = f"c-{slug}-{index}".replace("_", "-")
        if ident not in self.defs:
            data = base64.b64encode(
                (ASSETS / f"cell_{slug}_{index}.jpg").read_bytes()
            ).decode()
            self.defs[ident] = (
                f'<image id="{ident}" width="{CARD_W - 2 * CARD_PAD}" '
                f'height="{CARD_H - 2 * CARD_PAD}" href="data:image/jpeg;base64,{data}"/>'
            )
        return ident

    def deck(self, slug: str, available: int, cards: int, x: int, offset: int) -> None:
        """Desenha de trás para a frente: o cartão da frente fica à esquerda."""
        for position in reversed(range(cards)):
            left = x + position * offset
            self.parts.append(
                f'<rect x="{left:g}" y="{DECK_Y}" width="{CARD_W}" height="{CARD_H}" '
                f'rx="3" fill="#ffffff" stroke="rgba(43,27,70,0.35)" stroke-width="1"/>'
            )
            ident = self.cell(slug, position % available)
            self.parts.append(
                f'<use href="#{ident}" x="{left + CARD_PAD:g}" y="{DECK_Y + CARD_PAD}"/>'
            )

    def label(self, text: str, center: float) -> None:
        self.glyphs.update(text)
        self.parts.append(
            f'<text x="{center:g}" y="{LABEL_Y}" fill="{MUTED}" font-size="{FS}" '
            f'font-family="{MONO}" text-anchor="middle">{esc(text)}</text>'
        )

    def arrow(
        self, start: float, end: float, *, dy: float = 0, dashed: bool = False
    ) -> None:
        y = DECK_Y + CARD_H / 2 + dy
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        self.parts.append(
            f'<path d="M {start:g} {y:g} L {end:g} {y:g}" fill="none" '
            f'stroke="{MUTED}" stroke-width="1.6"{dash} marker-end="url(#a)"/>'
        )


def build(name: str, train: int, val: int, slug: str) -> tuple[str, set[str]]:
    m = Micro()
    train_cards = deck_size(train)
    val_cards = deck_size(val)
    train_off = card_offset(train_cards, deck_width(train_cards, TRAIN_MAX_DECK))
    val_off = card_offset(val_cards, deck_width(val_cards, VAL_MAX_DECK))
    test_off = card_offset(TEST_CARDS, TEST_CARDS_WIDTH)
    # Largura real, já com a folga truncada — é dela que a seta parte.
    train_w = CARD_W + (train_cards - 1) * train_off
    val_w = CARD_W + (val_cards - 1) * val_off
    test_w = CARD_W + (TEST_CARDS - 1) * test_off
    _, val_x, test_x, _ = row_extent(train, val)

    m.deck(f"{slug}_train", pool_size(f"{slug}_train"), train_cards, TRAIN_X, train_off)
    m.deck(f"{slug}_val", pool_size(f"{slug}_val"), val_cards, val_x, val_off)
    m.deck("test", pool_size("test"), TEST_CARDS, test_x, test_off)

    # Treino e validação se alternam a cada época, daí o par de idas e voltas; o
    # teste externo roda uma vez só, depois do checkpoint escolhido — daí o tracejado.
    m.arrow(TRAIN_X + train_w + GUTTER, val_x - GUTTER, dy=-10)
    m.arrow(val_x - GUTTER, TRAIN_X + train_w + GUTTER, dy=10)
    m.arrow(val_x + val_w + GUTTER, test_x - GUTTER, dashed=True)

    m.label(f"{thousands(train)} treino", TRAIN_X + train_w / 2)
    m.label(f"{val} validação", val_x + val_w / 2)
    m.label(f"{TEST_IMAGES} teste", test_x + test_w / 2)

    ident = name.replace(".", "-")
    title = f"Conjuntos da condição {name}"
    desc = (
        f"Da esquerda para a direita: {thousands(train)} imagens de treino, {val} de "
        f"validação e as {TEST_IMAGES} imagens do teste externo CitDet, o mesmo para "
        "todas as condições. Fotos ilustram os tipos de dados, não identificam os splits. "
        "Treino e validação usam 3 cartas por 26 imagens; as 3 cartas de teste são fixas."
    )
    svg = (
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" role="img" '
        f'aria-labelledby="{ident}-t {ident}-d" xmlns="http://www.w3.org/2000/svg">\n'
        f'<title id="{ident}-t">{esc(title)}</title>\n'
        f'<desc id="{ident}-d">{esc(desc)}</desc>\n'
        "<defs>\n{css}\n"
        f'<marker id="a" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>\n'
        + "\n".join(m.defs.values())
        + "\n</defs>\n"
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAPER}"/>\n'
        + "\n".join(m.parts)
        + "\n</svg>\n"
    )
    return svg, m.glyphs


def main() -> None:
    OUT.mkdir(exist_ok=True)
    built: dict[str, str] = {}
    glyphs: set[str] = set()
    for name, train, val, slug in CONDITIONS:
        svg, used = build(name, train, val, slug)
        built[name] = svg
        glyphs |= used

    css = (
        "<style>"
        + fontkit.face(
            "Geist Mono", "Geist Mono:wght@400", glyphs, "geist-mono-400", weight="400"
        )
        + "</style>"
    )
    for name, svg in built.items():
        path = OUT / f"condicao-{name}.svg"
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n' + svg.replace("{css}", css),
            encoding="utf-8",
        )
        print(f"{path.name} · {WIDTH}×{HEIGHT} · {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
