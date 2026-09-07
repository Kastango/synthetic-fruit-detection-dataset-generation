"""Recorta e empilha os ativos da pipeline para uso como miniaturas do fluxograma.

Fonte de verdade: o diretório `data/assets/regenerated/` da pipeline atual.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

PIPE = Path(__file__).resolve().parents[2]
ROOT = PIPE / "docs/figures"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--preview",
    type=Path,
    required=True,
    help="Dataset ilustrativo com images, labels e metadata",
)
args = parser.parse_args()
ASSETS = PIPE / "data/assets/regenerated"
OUT = ROOT / "diagram-assets"
OUT.mkdir(parents=True, exist_ok=True)

SS = 4  # supersampling durante a composição
EXPORT = 3  # o PNG guardado fica 3x maior que o tamanho de exibição no SVG
CARD = 112 * SS  # lado do cartão da frente, em px de render
OFFSET = 14 * SS  # deslocamento entre cartões empilhados
FRAME = 3 * SS  # moldura branca tipo foto revelada
INK = (43, 27, 70)  # roxo-preto do skin
TCC_ORANGE = (255, 166, 41)
TCC_MUTED = (109, 90, 146)


def crop_square(
    image: Image.Image, cx: float = 0.5, cy: float = 0.5, frac: float = 1.0
) -> Image.Image:
    """Recorte quadrado centrado em (cx, cy) relativos, cobrindo `frac` do menor lado."""
    w, h = image.size
    side = int(min(w, h) * frac)
    left = max(0, min(int(cx * w - side / 2), w - side))
    top = max(0, min(int(cy * h - side / 2), h - side))
    return image.crop((left, top, left + side, top + side))


def checkerboard(size: int, square: int = 8 * SS) -> Image.Image:
    board = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    for row in range(0, size, square):
        for col in range(0, size, square):
            if (row // square + col // square) % 2:
                draw.rectangle(
                    [col, row, col + square, row + square], fill=(214, 214, 214)
                )
    return board


def as_card(image: Image.Image, fade: float = 0.0) -> Image.Image:
    """Cartão quadrado com moldura branca e fio de contorno; `fade` clareia os cartões de trás."""
    inner = CARD - 2 * FRAME
    photo = image.convert("RGB").resize((inner, inner), Image.Resampling.LANCZOS)
    if fade:
        photo = Image.blend(photo, Image.new("RGB", photo.size, (255, 255, 255)), fade)
    card = Image.new("RGBA", (CARD, CARD), (255, 255, 255, 255))
    card.paste(photo, (FRAME, FRAME))
    ImageDraw.Draw(card).rectangle(
        [0, 0, CARD - 1, CARD - 1], outline=(*INK, 90), width=max(1, SS // 2)
    )
    return card


def stack(images: list[Image.Image], name: str, mirror: bool = False) -> None:
    """Empilha do fundo para a frente. `mirror` traz o cartão da frente para a direita,
    de modo que as pilhas do trilho esquerdo apresentem a foto nítida ao diagrama."""
    depth = len(images) - 1
    span = CARD + depth * OFFSET
    canvas = Image.new("RGBA", (span, span), (0, 0, 0, 0))
    for index, image in enumerate(reversed(images)):  # o último da lista fica na frente
        back = depth - index
        card = as_card(image, fade=0.30 * back / max(1, depth))
        if back:
            card.putalpha(
                card.getchannel("A").point(
                    lambda alpha, back=back: int(alpha * (1 - 0.15 * back))
                )
            )
        left = span - CARD - back * OFFSET if mirror else back * OFFSET
        canvas.alpha_composite(card, (left, back * OFFSET))
    side = span * EXPORT // SS
    canvas = canvas.resize((side, side), Image.Resampling.LANCZOS)
    canvas.save(OUT / f"{name}.png", optimize=True)
    print(f"{name}.png {canvas.size} (exibido a {span // SS}px)")


def single(image: Image.Image, name: str, width: int, height: int) -> None:
    """Cartão retangular isolado, para as saídas da pipeline."""
    inner = (width * SS - 2 * FRAME, height * SS - 2 * FRAME)
    card = Image.new("RGBA", (width * SS, height * SS), (255, 255, 255, 255))
    card.paste(
        ImageOps.fit(image.convert("RGB"), inner, Image.Resampling.LANCZOS),
        (FRAME, FRAME),
    )
    ImageDraw.Draw(card).rectangle(
        [0, 0, width * SS - 1, height * SS - 1],
        outline=(*INK, 90),
        width=max(1, SS // 2),
    )
    out = card.resize((width * EXPORT, height * EXPORT), Image.Resampling.LANCZOS)
    out.save(OUT / f"{name}.png", optimize=True)
    print(f"{name}.png {out.size} (exibido a {width}×{height})")


def portrait_stack(images: list[Image.Image], name: str) -> None:
    """Empilha cenas verticais preservando a proporção 720 × 960 do canvas."""
    width, height, offset = 88 * SS, 118 * SS, 10 * SS
    depth = len(images) - 1
    canvas = Image.new(
        "RGBA",
        (width + depth * offset, height + depth * offset),
        (0, 0, 0, 0),
    )
    for index, image in enumerate(reversed(images)):
        back = depth - index
        inner = (width - 2 * FRAME, height - 2 * FRAME)
        photo = ImageOps.fit(image.convert("RGB"), inner, Image.Resampling.LANCZOS)
        if back:
            photo = Image.blend(
                photo,
                Image.new("RGB", photo.size, (255, 255, 255)),
                0.15 * back,
            )
        card = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        card.paste(photo, (FRAME, FRAME))
        ImageDraw.Draw(card).rectangle(
            [0, 0, width - 1, height - 1],
            outline=(*INK, 90),
            width=max(1, SS // 2),
        )
        if back:
            card.putalpha(
                card.getchannel("A").point(
                    lambda alpha, back=back: int(alpha * (1 - 0.12 * back))
                )
            )
        canvas.alpha_composite(card, (back * offset, back * offset))
    out = canvas.resize(
        (canvas.width * EXPORT // SS, canvas.height * EXPORT // SS),
        Image.Resampling.LANCZOS,
    )
    out.save(OUT / f"{name}.png", optimize=True)
    print(
        f"{name}.png {out.size} "
        f"(exibido a {canvas.width // SS}×{canvas.height // SS}px)"
    )


def load(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def metadata_preview(path: Path) -> Image.Image:
    """Representação legível de um JSON real quando reduzido à miniatura."""
    data = json.loads(path.read_text(encoding="utf-8"))
    page = Image.new("RGB", (720, 960), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=42)
    brace_font = ImageFont.load_default(size=62)
    draw.text((52, 44), "{", font=brace_font, fill=INK)
    y = 132
    for key in list(data)[:8]:
        draw.text((96, y), f'"{key}"', font=font, fill=TCC_MUTED)
        draw.text((500, y), ": …,", font=font, fill=TCC_ORANGE)
        y += 82
    draw.text((52, y + 12), "}", font=brace_font, fill=INK)
    return page


# --- 1. fundos de árvores fora do período produtivo e seus mapas DepthPro -------------
TREES = ["IMG_3466", "IMG_3518", "IMG_3541"]
CROP = {"cx": 0.5, "cy": 0.42, "frac": 0.92}

stack(
    [crop_square(load(ASSETS / "backgrounds" / f"{s}.jpg"), **CROP) for s in TREES],
    "stack_arvores",
)
stack(
    [
        crop_square(Image.open(ASSETS / "backgrounds_map" / f"{s}_depth.png"), **CROP)
        for s in TREES
    ],
    "stack_mapas",
)

# --- 2. fotos de frutas em fundo uniforme e os recortes correspondentes ----------------
FRUITS = ["IMG_2145", "IMG_2190", "IMG_2217"]
stack(
    [
        crop_square(load(PIPE / "data/raw/fruits" / f"{s}.JPG"), cy=0.52, frac=0.72)
        for s in FRUITS
    ],
    "stack_frutas",
    mirror=True,
)

cards = []
for stem in FRUITS:
    cut = Image.open(
        ASSETS / "pictures_trimmed" / f"{stem.lower()}-trimmed.png"
    ).convert("RGBA")
    side = int(max(cut.size) * 1.12)
    board = checkerboard(side).convert("RGBA")
    scaled = cut.copy()
    scaled.thumbnail((int(side * 0.86), int(side * 0.86)), Image.Resampling.LANCZOS)
    board.alpha_composite(
        scaled, ((side - scaled.width) // 2, (side - scaled.height) // 2)
    )
    cards.append(board)
stack(cards, "stack_recortes", mirror=True)

# --- 3. saídas do loop: cenas sintéticas e arquivos contendo apenas as caixas ----------
PREVIEW = args.preview.resolve()
SCENES = [p.stem for p in sorted((PREVIEW / "images/train").glob("*.jpg"))[:3]]
if len(SCENES) < 3:
    raise SystemExit("O preview deve conter pelo menos três cenas de treino.")
scene_images = [
    Image.open(PREVIEW / "images/train" / f"{stem}.jpg").convert("RGB")
    for stem in SCENES
]
box_images = []
for stem, scene in zip(SCENES, scene_images, strict=True):
    boxes = Image.new("RGB", scene.size, (255, 255, 255))
    draw = ImageDraw.Draw(boxes)
    width, height = boxes.size
    for line in (PREVIEW / "labels/train" / f"{stem}.txt").read_text().split("\n"):
        if not line.strip():
            continue
        _, xc, yc, bw, bh = (float(value) for value in line.split())
        x0, y0 = (xc - bw / 2) * width, (yc - bh / 2) * height
        box = [x0, y0, x0 + bw * width, y0 + bh * height]
        draw.rectangle(box, outline=TCC_ORANGE, width=8)
    box_images.append(boxes)

portrait_stack(scene_images, "stack_cenas")
portrait_stack(box_images, "stack_caixas")
portrait_stack(
    [metadata_preview(PREVIEW / "metadata/train" / f"{stem}.json") for stem in SCENES],
    "stack_metadados",
)


# --- 4. ilustrações do laço: uma única tentativa de inserção, do começo ao fim ---------
# As cinco faixas mostram a MESMA fruta, no MESMO fundo e no MESMO ponto. A inserção é
# produzida pelas próprias funções de `fruit_pipeline.synthesis`, então cada faixa é o
# resultado real da etapa que ela ilustra — não uma recriação.
import random
import sys

sys.path.insert(0, str(PIPE))

import numpy as np
import yaml

from fruit_pipeline import synthesis

BAND_W, BAND_H = 288, 64  # tamanho de exibição da faixa dentro do nó
BAND = (BAND_W * SS, BAND_H * SS)
CFG = yaml.safe_load(
    (PIPE / "configs/synthesis/confirmatory_pool.yaml").read_text(encoding="utf-8")
)
ROTATION = 24  # o giro sorteado que a faixa "escala e giro" mostra
CUTOUT = "img_2190-trimmed.png"


def save_band(image: Image.Image, name: str) -> None:
    # As faixas são opacas: JPEG corta ~80 % do peso que elas somam ao SVG embutido.
    out = image.convert("RGB").resize(
        (BAND_W * EXPORT, BAND_H * EXPORT), Image.Resampling.LANCZOS
    )
    out.save(OUT / f"{name}.jpg", quality=92, optimize=True, subsampling=0)
    print(f"{name}.jpg {out.size} (exibido a {BAND_W}×{BAND_H})")


def crosshair(
    draw: ImageDraw.ImageDraw, cx: float, cy: float, size=11, width=3
) -> None:
    for dx, dy in ((1, 0), (0, 1)):
        line = [cx - dx * size, cy - dy * size, cx + dx * size, cy + dy * size]
        draw.line(line, fill=INK, width=width + 4)
        draw.line(line, fill=TCC_ORANGE, width=width)


# A cena, o mapa e o recorte que aparecem em todas as faixas.
canvas_size = tuple(int(v) for v in CFG["canvas"])
scene, depth_image = synthesis._open_background_pair(
    ASSETS / "backgrounds" / f"{TREES[0]}.jpg",
    ASSETS / "backgrounds_map" / f"{TREES[0]}_depth.png",
    canvas_size,
)
scene = synthesis._apply_scene_grading(scene, CFG["output"]["scene_grading"])
depth_image = depth_image.filter(
    ImageFilter.GaussianBlur(float(CFG["occlusion"]["depth_smooth_radius"]))
)
depth_array = np.asarray(depth_image, dtype=np.uint8)

cutout = Image.open(ASSETS / "pictures_trimmed" / CUTOUT).convert("RGBA")
cutout = synthesis._scale_cutout(cutout, CFG["objects"], random.Random(42), canvas_size)
cutout = cutout.rotate(ROTATION, resample=Image.Resampling.BICUBIC, expand=True)
cutout = synthesis._trim_alpha(cutout)

# Procura uma tentativa em que a folhagem cubra parte da fruta sem descaracterizá-la.
placed = None
for seed in range(600):
    candidate = synthesis._placement_with_depth_scale(
        scene,
        depth_array,
        cutout,
        CFG,
        random.Random(seed),
        CFG["objects"]["depth_scale"],
    )
    if candidate is None or not 0.45 <= candidate["visibility_at_insert"] <= 0.72:
        continue
    # A janela precisa caber inteira em volta da fruta: perto da borda do canvas
    # o enquadramento sairia cortado e a moldura da faixa ficaria assimétrica.
    fx = candidate["x"] + candidate["image"].width / 2
    fy = candidate["y"] + candidate["image"].height / 2
    if 0.28 < fx / canvas_size[0] < 0.72 and 0.22 < fy / canvas_size[1] < 0.62:
        placed = candidate
        break
if placed is None:
    raise SystemExit("nenhuma inserção com oclusão parcial encontrada")

px, py = placed["x"], placed["y"]
pw, ph = placed["image"].size
center = (px + pw / 2, py + ph / 2)
print(
    f"  inserção: {pw}×{ph} px em ({px}, {py}) · {placed['visibility_at_insert']:.0%} visível"
)

# Janela comum às faixas 2–5: mesma moldura, mesmo enquadramento.
# A janela nunca pode passar da cena: fora dela o recorte viria preenchido de preto.
win_w = min(int(pw * 6.8), canvas_size[0])
win_h = min(int(win_w * BAND_H / BAND_W), canvas_size[1])
win_w = int(win_h * BAND_W / BAND_H)
left = max(0, min(int(center[0] - win_w / 2), canvas_size[0] - win_w))
top = max(0, min(int(center[1] - win_h / 2), canvas_size[1] - win_h))
WINDOW = (left, top, left + win_w, top + win_h)
LOCAL = (center[0] - WINDOW[0], center[1] - WINDOW[1])


def window(image: Image.Image) -> Image.Image:
    return image.convert("RGB").crop(WINDOW).resize(BAND, Image.Resampling.LANCZOS)


# 4.1 escala e giro: o recorte exatamente como entra na tentativa
board = checkerboard(BAND[0]).crop((0, 0, *BAND)).convert("RGBA")
piece = cutout.copy()
piece = ImageOps.contain(
    piece, (int(BAND[1] * 0.88), int(BAND[1] * 0.88)), Image.Resampling.LANCZOS
)
board.alpha_composite(
    piece, ((BAND[0] - piece.width) // 2, (BAND[1] - piece.height) // 2)
)
save_band(board, "loop_escala")

# 4.2 a coordenada sorteada, sobre o fundo
xy = window(scene)
crosshair(
    ImageDraw.Draw(xy),
    *[v / (WINDOW[2] - WINDOW[0]) * BAND[0] for v in LOCAL][:1],
    LOCAL[1] / (WINDOW[3] - WINDOW[1]) * BAND[1],
)
save_band(xy, "loop_xy")

# 4.3 a mesma coordenada no mapa, com a área que a fruta ocupa após o fator de profundidade
depth_band = window(depth_image)
draw = ImageDraw.Draw(depth_band)
scale_x = BAND[0] / (WINDOW[2] - WINDOW[0])
scale_y = BAND[1] / (WINDOW[3] - WINDOW[1])
draw.rectangle(
    [
        (px - WINDOW[0]) * scale_x,
        (py - WINDOW[1]) * scale_y,
        (px + pw - WINDOW[0]) * scale_x,
        (py + ph - WINDOW[1]) * scale_y,
    ],
    outline=TCC_ORANGE,
    width=3,
)
crosshair(draw, LOCAL[0] * scale_x, LOCAL[1] * scale_y)
save_band(depth_band, "loop_profundidade")

# 4.4 aparência e sombras: o mesmo pixel de fruta já corrigido, ainda sem a oclusão
appearance = scene.copy()
opaque = placed["image"].copy()
opaque.putalpha(Image.fromarray(placed["amodal_mask"]))
appearance.paste(opaque, (px, py), opaque)
save_band(window(appearance), "loop_aparencia")

# 4.5 composição: a mesma instância com a máscara de visibilidade aplicada
composed = scene.copy()
composed.paste(placed["image"], (px, py), placed["image"])
save_band(window(composed), "loop_compor")

# Registrar a origem dos exemplos sem publicar caminhos absolutos da máquina.
from fruit_pipeline.common import stable_hash

provenance = {
    "purpose": "Ilustração, não amostra do pool usado nos resultados publicados",
    "loop_config_hash": stable_hash(CFG, 24),
    "preview_summary": json.loads((PREVIEW / "summary.json").read_text()),
    "scale_band": "Insertion cutout enlarged for legibility",
    "backgrounds": TREES,
    "fruit_photos": FRUITS,
    "preview_scenes": SCENES,
    "loop": {
        "cutout": CUTOUT,
        "background": TREES[0],
        "rotation": ROTATION,
        "placement_seed": seed,
        "x": px,
        "y": py,
        "visibility_at_insert": placed["visibility_at_insert"],
    },
}
(OUT / "provenance.json").write_text(
    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
)
