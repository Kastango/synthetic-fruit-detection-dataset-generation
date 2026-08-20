from __future__ import annotations

import concurrent.futures
import json
import math
import random
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    ImageStat,
)

from .common import (
    IMAGE_SUFFIXES,
    atomic_write_json,
    atomic_write_text,
    image_files,
    relative_or_absolute,
    sha256_file,
    stable_hash,
)

GENERATOR_SCHEMA_VERSION = 5
ASSET_SPLIT_SCHEMA_VERSION = 2


def find_depth_map(background: Path, depth_directory: Path) -> Path | None:
    stems = (f"{background.stem}_depth", f"{background.stem}_map", background.stem)
    for stem in stems:
        for suffix in sorted(IMAGE_SUFFIXES):
            candidate = depth_directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _split_paths(
    paths: list[Path], ratio: float, seed: int, namespace: str
) -> dict[str, list[Path]]:
    shuffled = sorted(paths)
    random.Random(seed + int(stable_hash(namespace, 8), 16)).shuffle(shuffled)
    train_count = round(len(shuffled) * ratio)
    if len(shuffled) > 1:
        train_count = min(max(train_count, 1), len(shuffled) - 1)
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count:]),
    }


def create_asset_split(
    asset_root: Path,
    *,
    train_ratio: float,
    seed: int,
    force: bool = False,
) -> dict:
    target = asset_root / "asset_split.json"
    backgrounds_dir = asset_root / "backgrounds"
    depth_dir = asset_root / "backgrounds_map"
    cutouts_dir = asset_root / "pictures_trimmed"
    pairs = []
    for background in image_files(backgrounds_dir):
        depth = find_depth_map(background, depth_dir)
        if depth is not None:
            pairs.append((background, depth))
    if not pairs:
        raise FileNotFoundError(f"nenhum par fundo/profundidade em {asset_root}")
    cutouts = image_files(cutouts_dir)
    if not cutouts:
        raise FileNotFoundError(f"nenhum recorte em {cutouts_dir}")

    pair_lookup = {background: depth for background, depth in pairs}

    def rel(path: Path) -> str:
        return path.relative_to(asset_root).as_posix()

    source_fingerprint = stable_hash(
        [
            (
                rel(path),
                path.stat().st_size,
                rel(pair_lookup[path]),
                pair_lookup[path].stat().st_size,
            )
            for path in sorted(pair_lookup)
        ]
        + [(rel(path), path.stat().st_size) for path in sorted(cutouts)],
        24,
    )
    if target.exists() and not force:
        previous = json.loads(target.read_text(encoding="utf-8"))
        if (
            previous.get("version") == ASSET_SPLIT_SCHEMA_VERSION
            and previous.get("seed") == seed
            and previous.get("train_ratio") == train_ratio
            and previous.get("source_fingerprint") == source_fingerprint
        ):
            return previous

    background_split = _split_paths(list(pair_lookup), train_ratio, seed, "backgrounds")
    cutout_split = _split_paths(cutouts, train_ratio, seed, "cutouts")
    result = {
        "version": ASSET_SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "train_ratio": train_ratio,
        "asset_root": relative_or_absolute(asset_root),
        "source_fingerprint": source_fingerprint,
        "splits": {},
        "orphans": {
            "depth_maps": sorted(
                rel(path)
                for path in image_files(depth_dir)
                if path not in set(pair_lookup.values())
            )
        },
    }
    for split_name in ("train", "val"):
        result["splits"][split_name] = {
            "backgrounds": [
                {"image": rel(path), "depth": rel(pair_lookup[path])}
                for path in background_split[split_name]
            ],
            "cutouts": [rel(path) for path in cutout_split[split_name]],
        }
    atomic_write_json(target, result)
    return result


def validate_synthesis_config(config: dict) -> None:
    required = {
        "name",
        "seed",
        "images",
        "canvas",
        "objects",
        "placement",
        "appearance",
        "occlusion",
        "annotation",
        "output",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuração sintética sem chaves: {sorted(missing)}")
    width, height = map(int, config["canvas"])
    if width <= 0 or height <= 0:
        raise ValueError("canvas deve ser positivo")
    objects = config["objects"]
    if not 0 <= int(objects["min"]) <= int(objects["max"]):
        raise ValueError("intervalo de objetos inválido")
    if not 0 < float(objects["min_scale"]) <= float(objects["max_scale"]):
        raise ValueError("intervalo de escala inválido")
    if objects["scale_mode"] not in {"cutout", "canvas"}:
        raise ValueError("scale_mode deve ser cutout ou canvas")
    depth_scale = objects.get("depth_scale")
    if (
        depth_scale
        and depth_scale.get("enabled", False)
        and not (
            0 < float(depth_scale["far_scale"]) <= float(depth_scale["near_scale"])
        )
    ):
        raise ValueError(
            "depth_scale requer 0 < far_scale <= near_scale (objetos mais "
            "próximos não podem ficar menores que os mais distantes)"
        )
    if config["annotation"]["mode"] not in {"visible", "amodal", "rect"}:
        raise ValueError("annotation.mode deve ser visible, amodal ou rect")
    if not 0 <= float(config["placement"]["min_visibility"]) <= 1:
        raise ValueError("min_visibility deve estar entre 0 e 1")
    z_method = config["placement"]["z_method"]
    if z_method not in {"quantile", "mean_plus_std", "center_patch"}:
        raise ValueError(f"z_method desconhecido: {z_method}")
    if z_method == "center_patch":
        patch_fraction = float(config["placement"].get("z_patch_fraction", 0.2))
        if not 0 < patch_fraction <= 1:
            raise ValueError(
                "placement.z_patch_fraction deve estar entre 0 (exclusivo) e 1"
            )
        z_offset_jitter = float(config["placement"].get("z_offset_jitter", 0.0))
        if z_offset_jitter < 0:
            raise ValueError("placement.z_offset_jitter não pode ser negativo")
    exclude_bottom = config["placement"].get("exclude_bottom_fraction", 0.0)
    if not 0 <= float(exclude_bottom) < 1:
        raise ValueError("placement.exclude_bottom_fraction deve estar entre 0 e 1")
    appearance = config["appearance"]
    obsolete = {"light_probability", "light_power"} & set(appearance)
    if obsolete:
        options = ", ".join(f"appearance.{name}" for name in sorted(obsolete))
        raise ValueError(f"texturas de iluminação não são suportadas; remova {options}")
    if not 0 <= float(appearance["hardlight_power"]) <= 1:
        raise ValueError("appearance.hardlight_power deve estar entre 0 e 1")
    hsv_cast = appearance.get("hsv_cast")
    if hsv_cast and hsv_cast.get("enabled", False):
        for key in ("hue_power", "saturation_power", "value_power"):
            if not 0 <= float(hsv_cast[key]) <= 1:
                raise ValueError(f"appearance.hsv_cast.{key} deve estar entre 0 e 1")
        if not 0 <= float(hsv_cast.get("min_value_ratio", 0.0)) <= 1:
            raise ValueError(
                "appearance.hsv_cast.min_value_ratio deve estar entre 0 e 1"
            )
        if float(hsv_cast.get("value_power_jitter", 0.0)) < 0:
            raise ValueError(
                "appearance.hsv_cast.value_power_jitter não pode ser negativo"
            )
        if not 0 <= float(hsv_cast.get("bright_flatten_strength", 0.0)) <= 1:
            raise ValueError(
                "appearance.hsv_cast.bright_flatten_strength deve estar entre 0 e 1"
            )
    grading = config["output"].get("scene_grading")
    if grading and grading.get("enabled", False):
        for key in ("contrast", "saturation", "brightness"):
            if key in grading and float(grading[key]) <= 0:
                raise ValueError(f"output.scene_grading.{key} deve ser positivo")
    depth_smooth_radius = config["occlusion"].get("depth_smooth_radius", 0.0)
    if float(depth_smooth_radius) < 0:
        raise ValueError("occlusion.depth_smooth_radius não pode ser negativo")
    mask_threshold = config["occlusion"].get("mask_threshold")
    if mask_threshold is not None and not 0 < float(mask_threshold) < 1:
        raise ValueError("occlusion.mask_threshold deve estar entre 0 e 1 (exclusivos)")
    edge_feather_radius = config["occlusion"].get("edge_feather_radius", 0.0)
    if float(edge_feather_radius) < 0:
        raise ValueError("occlusion.edge_feather_radius não pode ser negativo")
    contact_shadow = config["occlusion"].get("contact_shadow")
    if contact_shadow and contact_shadow.get("enabled", False):
        strength = float(contact_shadow.get("strength", 0.0))
        radius_fraction = float(contact_shadow.get("radius_fraction", 0.04))
        if not 0 <= strength <= 1:
            raise ValueError("occlusion.contact_shadow.strength deve estar entre 0 e 1")
        if not 0 < radius_fraction <= 0.5:
            raise ValueError(
                "occlusion.contact_shadow.radius_fraction deve estar entre 0 "
                "(exclusivo) e 0.5"
            )
    cast_shadow = config["occlusion"].get("cast_shadow")
    if cast_shadow and cast_shadow.get("enabled", False):
        probability = float(cast_shadow.get("probability", 0.4))
        strength = float(cast_shadow.get("strength", 0.25))
        min_quantile = float(cast_shadow.get("min_quantile", 0.25))
        max_quantile = float(cast_shadow.get("max_quantile", 0.6))
        blur_radius = float(cast_shadow.get("blur_radius", 3.0))
        if not 0 <= probability <= 1:
            raise ValueError("occlusion.cast_shadow.probability deve estar entre 0 e 1")
        if not 0 <= strength <= 1:
            raise ValueError("occlusion.cast_shadow.strength deve estar entre 0 e 1")
        if not 0 <= min_quantile < max_quantile <= 1:
            raise ValueError(
                "occlusion.cast_shadow requer 0 <= min_quantile < max_quantile <= 1"
            )
        if blur_radius < 0:
            raise ValueError("occlusion.cast_shadow.blur_radius não pode ser negativo")
        if float(cast_shadow.get("distance_fraction", 0.35)) < 0:
            raise ValueError(
                "occlusion.cast_shadow.distance_fraction não pode ser negativo"
            )
        if float(cast_shadow.get("light_angle_jitter_degrees", 20.0)) < 0:
            raise ValueError(
                "occlusion.cast_shadow.light_angle_jitter_degrees não pode ser negativo"
            )


@lru_cache(maxsize=8)
def _open_background_pair_cached(
    background_path: str, depth_path: str, size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    with Image.open(background_path) as opened:
        background = ImageOps.exif_transpose(opened).convert("RGB")
    with Image.open(depth_path) as opened:
        oriented_depth = ImageOps.exif_transpose(opened)
        # O pacote preparado contém mapas ZoeDepth coloridos e usa o canal R.
        # Os mapas regenerados já são L; esta seleção mantém ambos compatíveis.
        depth = (
            oriented_depth.convert("RGB").getchannel("R")
            if len(oriented_depth.getbands()) > 1
            else oriented_depth.convert("L")
        )
    if background.size != depth.size and background.size == depth.size[::-1]:
        background = background.rotate(-90, expand=True)
    background = background.resize(size, Image.Resampling.LANCZOS)
    # BILINEAR só amostra uma vizinhança 2x2 e ignora a taxa de redução; numa
    # queda de ~4x (a fonte sai de ~4000px, o canvas é 720-960px) isso
    # equivale a subamostrar e perde exatamente o detalhe fino de folha/galho
    # que justificou trocar para o DepthPro. LANCZOS pondera a área
    # correspondente da imagem original e preserva bordas de profundidade
    # nitidamente melhor nessa mesma proporção.
    depth = depth.resize(size, Image.Resampling.LANCZOS)
    return background, depth


def _open_background_pair(
    background_path: Path, depth_path: Path, size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    background, depth = _open_background_pair_cached(
        str(background_path), str(depth_path), size
    )
    # O canvas é modificado pelas inserções; o mapa de profundidade é somente leitura.
    return background.copy(), depth


@lru_cache(maxsize=128)
def _open_cutout_cached(path: str) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGBA")


def _clear_image_caches() -> None:
    _open_background_pair_cached.cache_clear()
    _open_cutout_cached.cache_clear()


def _trim_alpha(image: Image.Image, threshold: int = 1) -> Image.Image:
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.nonzero(alpha >= threshold)
    if len(xs) == 0:
        return image.crop((0, 0, 1, 1))
    return image.crop(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    )


def _scale_cutout(
    image: Image.Image, config: dict, rng: random.Random, canvas: tuple[int, int]
) -> Image.Image:
    fraction = rng.uniform(float(config["min_scale"]), float(config["max_scale"]))
    if config["scale_mode"] == "cutout":
        scale = fraction
    else:
        scale = fraction * min(canvas) / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _apply_appearance_hardlight(
    fruit: Image.Image,
    background_region: Image.Image,
    appearance: dict,
) -> Image.Image:
    alpha = fruit.getchannel("A")
    rgb = fruit.convert("RGB")
    mean = tuple(
        round(value) for value in ImageStat.Stat(background_region.convert("RGB")).mean
    )
    color = Image.new("RGB", fruit.size, mean)
    hardlight = ImageChops.hard_light(rgb, color)
    power = min(max(float(appearance["hardlight_power"]), 0.0), 1.0)
    rgb = Image.blend(rgb, hardlight, power)
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def _apply_appearance_hsv_cast(
    fruit: Image.Image,
    background_region: Image.Image,
    hsv_cast: dict,
    rng: random.Random | None = None,
) -> Image.Image:
    # Os recortes são fotos de estúdio com luz difusa uniforme; o fundo real
    # tem luz solar direcional. Um hard-light plano contra a cor média do
    # fundo apaga o brilho/sombra natural da fruta. Aqui a luminância (V)
    # original é preservada quase inteira e só o matiz/saturação (H, S) são
    # puxados em direção a um alvo derivado da cor ambiente local, então a
    # fruta mantém sua forma tridimensional mas ganha a temperatura de cor
    # da cena.
    alpha = fruit.getchannel("A")
    rgb = fruit.convert("RGB")
    mean = tuple(
        round(value) for value in ImageStat.Stat(background_region.convert("RGB")).mean
    )
    if hsv_cast.get("use_hardlight_target", False):
        # O hard-light já responde de forma não linear ao valor de cada
        # pixel da fruta (clareia onde já é claro, escurece onde é escuro),
        # em vez de puxar tudo para um único valor plano. Usar seu
        # resultado como alvo por pixel do matiz/saturação recupera parte
        # da coesão visual do hard-light original sem achatar a luminância.
        target = ImageChops.hard_light(rgb, Image.new("RGB", fruit.size, mean))
        target_h, target_s, target_v = target.convert("HSV").split()
        bg_h = np.asarray(target_h, dtype=np.float32)
        bg_s = np.asarray(target_s, dtype=np.float32)
        bg_v = np.asarray(target_v, dtype=np.float32)
    else:
        bg_h, bg_s, bg_v = (
            Image.new("RGB", (1, 1), mean).convert("HSV").getpixel((0, 0))
        )
    fruit_h, fruit_s, fruit_v = rgb.convert("HSV").split()
    h_array = np.asarray(fruit_h, dtype=np.float32)
    s_array = np.asarray(fruit_s, dtype=np.float32)
    v_array = np.asarray(fruit_v, dtype=np.float32)
    hue_power = min(max(float(hsv_cast["hue_power"]), 0.0), 1.0)
    saturation_power = min(max(float(hsv_cast["saturation_power"]), 0.0), 1.0)
    value_power = float(hsv_cast["value_power"])
    value_power_jitter = float(hsv_cast.get("value_power_jitter", 0.0))
    if value_power_jitter > 0:
        # Um value_power fixo dá a mesma resposta de luz/sombra pra toda
        # fruta; a variação observada nas fotos reais é maior (algumas bem
        # mais claras ou mais escuras que a média). Sortear por instância
        # (mesmo rng da posição, determinístico pela seed) alarga o
        # espalhamento sem mudar o valor médio de value_power no conjunto.
        sampler = rng or random.Random()
        value_power = sampler.uniform(
            value_power - value_power_jitter, value_power + value_power_jitter
        )
    value_power = min(max(value_power, 0.0), 1.0)
    value_power_effective = value_power
    saturation_power_effective = saturation_power
    bright_flatten_strength = min(
        max(float(hsv_cast.get("bright_flatten_strength", 0.0)), 0.0), 1.0
    )
    if bright_flatten_strength > 0:
        # Perto do céu/luz estourada, a fruta real perde relevo e satura
        # menos (estoura de exposição) — o usuário pediu que ela fique mais
        # "chapada" nessas regiões, sem mudar o comportamento em fundos
        # médios/escuros, que já estão bons. brightness_factor cresce só
        # onde o alvo (bg_v) já é bem claro, e empurra value/saturation_power
        # em direção a 1.0 (adoção quase total do alvo) proporcionalmente.
        brightness_factor = np.clip(
            np.asarray(bg_v, dtype=np.float32) / 255.0, 0.0, 1.0
        )
        boost = bright_flatten_strength * brightness_factor
        value_power_effective = value_power + (1.0 - value_power) * boost
        saturation_power_effective = saturation_power + (1.0 - saturation_power) * boost
    hue_diff = ((bg_h - h_array + 128) % 256) - 128
    h_new = (h_array + hue_diff * hue_power) % 256
    s_new = s_array + (bg_s - s_array) * saturation_power_effective
    v_new = v_array + (bg_v - v_array) * value_power_effective
    min_value_ratio = min(max(float(hsv_cast.get("min_value_ratio", 0.0)), 0.0), 1.0)
    if min_value_ratio > 0:
        # Em regiões muito escuras, puxar o valor todo para o alvo derrete a
        # fruta num blob marrom indistinguível do fundo — a cor real de uma
        # fruta na sombra continua identificável, só menos brilhante. Um piso
        # relativo ao valor original evita esse colapso sem tocar em regiões
        # claras (onde bg_v > v_array e o piso não é atingido).
        v_new = np.maximum(v_new, v_array * min_value_ratio)
    blended = Image.merge(
        "HSV",
        [
            Image.fromarray(np.clip(h_new, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(s_new, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(v_new, 0, 255).astype(np.uint8)),
        ],
    ).convert("RGB")
    result = blended.convert("RGBA")
    result.putalpha(alpha)
    return result


def _apply_appearance(
    fruit: Image.Image,
    background_region: Image.Image,
    appearance: dict,
    rng: random.Random | None = None,
) -> Image.Image:
    hsv_cast = appearance.get("hsv_cast")
    if hsv_cast and hsv_cast.get("enabled", False):
        return _apply_appearance_hsv_cast(fruit, background_region, hsv_cast, rng)
    return _apply_appearance_hardlight(fruit, background_region, appearance)


def _apply_occlusion_contact_shadow(
    fruit: Image.Image,
    visibility: np.ndarray,
    opaque: np.ndarray,
    occlusion: dict,
) -> Image.Image:
    contact_shadow = occlusion.get("contact_shadow")
    if not contact_shadow or not contact_shadow.get("enabled", False):
        return fruit
    # Uma folha que passa à frente não produz apenas um recorte geométrico:
    # ela também bloqueia parte da luz na faixa imediatamente vizinha da
    # fruta ainda visível. Desfocar apenas a região realmente ocluída gera
    # essa penumbra curta sem escurecer o contorno externo do recorte.
    occluded = opaque.astype(np.float32) * (1.0 - visibility.astype(np.float32) / 255)
    if not np.any(occluded > 0):
        return fruit
    radius = max(
        0.5,
        min(fruit.size) * float(contact_shadow.get("radius_fraction", 0.04)),
    )
    shadow = (
        np.asarray(
            Image.fromarray(np.rint(occluded * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius)
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    gain = 1.0 - float(contact_shadow.get("strength", 0.0)) * shadow
    rgb = np.asarray(fruit.convert("RGB"), dtype=np.float32)
    shaded = Image.fromarray(
        np.clip(rgb * gain[..., None], 0, 255).astype(np.uint8)
    ).convert("RGBA")
    shaded.putalpha(fruit.getchannel("A"))
    return shaded


def _apply_cast_shadow(
    fruit: Image.Image,
    x: int,
    y: int,
    depth: np.ndarray,
    opaque: np.ndarray,
    occlusion: dict,
    rng: random.Random,
) -> Image.Image:
    cast_shadow = occlusion.get("cast_shadow")
    if not cast_shadow or not cast_shadow.get("enabled", False):
        return fruit
    probability = min(max(float(cast_shadow.get("probability", 0.4)), 0.0), 1.0)
    if rng.random() > probability:
        return fruit
    height, width = opaque.shape
    canvas_height, canvas_width = depth.shape
    # Uma sombra real vem de um objeto a uma certa distância, numa direção
    # ditada pela luz — não fica alinhada com a própria fruta. Desloca a
    # amostragem de profundidade nessa direção antes de derivar o formato,
    # em vez de reusar a região exatamente sob a fruta.
    base_angle = float(cast_shadow.get("light_angle_degrees", 315.0))
    angle_jitter = float(cast_shadow.get("light_angle_jitter_degrees", 20.0))
    angle = math.radians(base_angle + rng.uniform(-angle_jitter, angle_jitter))
    distance = max(width, height) * float(cast_shadow.get("distance_fraction", 0.35))
    offset_x = min(max(x + round(math.cos(angle) * distance), 0), canvas_width - width)
    offset_y = min(
        max(y + round(math.sin(angle) * distance), 0), canvas_height - height
    )
    region_depth = depth[offset_y : offset_y + height, offset_x : offset_x + width]
    local_values = region_depth[opaque]
    if not len(local_values):
        return fruit
    # Sorteia um limiar independente da oclusão real (que decide o que é
    # removido) dentro da faixa de profundidade observada na região
    # deslocada — luz de dossel real projeta manchas de sombra sobre frutas
    # que não estão geometricamente ocluídas por nada. Usar a estrutura de
    # profundidade local (em vez de uma mancha genérica) mantém a sombra
    # com formato plausível para aquela cena específica.
    quantile = rng.uniform(
        float(cast_shadow.get("min_quantile", 0.25)),
        float(cast_shadow.get("max_quantile", 0.6)),
    )
    shadow_z = float(np.quantile(local_values, quantile))
    shadow_mask = (region_depth > shadow_z).astype(np.uint8) * 255
    blur_radius = float(cast_shadow.get("blur_radius", 3.0))
    if blur_radius > 0:
        shadow_mask = np.asarray(
            Image.fromarray(shadow_mask).filter(ImageFilter.GaussianBlur(blur_radius))
        )
    shadow = shadow_mask.astype(np.float32) / 255.0
    strength = min(max(float(cast_shadow.get("strength", 0.25)), 0.0), 1.0)
    gain = 1.0 - strength * shadow
    rgb = np.asarray(fruit.convert("RGB"), dtype=np.float32)
    shaded = Image.fromarray(
        np.clip(rgb * gain[..., None], 0, 255).astype(np.uint8)
    ).convert("RGBA")
    shaded.putalpha(fruit.getchannel("A"))
    return shaded


def _bbox(mask: np.ndarray, threshold: int = 1) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask >= threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _finish_placement(
    fruit: Image.Image,
    x: int,
    y: int,
    alpha_original: np.ndarray,
    alpha_float: np.ndarray,
    opaque: np.ndarray,
    original_pixels: int,
    canvas: Image.Image,
    depth: np.ndarray,
    config: dict,
    anchor: tuple[int, int] | None = None,
    rng: random.Random | None = None,
) -> dict | None:
    placement = config["placement"]
    region_depth = depth[y : y + fruit.height, x : x + fruit.width]
    local_values = region_depth[opaque]
    if not len(local_values):
        return None
    if placement["z_method"] == "quantile":
        placement_values = local_values
        z_value = float(np.quantile(local_values, float(placement["z_quantile"])))
    elif placement["z_method"] == "mean_plus_std":
        placement_values = local_values
        z_value = float(local_values.mean()) + float(
            placement["z_std_multiplier"]
        ) * float(local_values.std())
    elif placement["z_method"] == "center_patch":
        # O Z da fruta deve vir do ponto onde ela foi ancorada, não de um
        # quantil calculado sobre toda a sua silhueta. O quantil local força
        # quase a mesma fração de oclusão em toda inserção (por exemplo, q=.65
        # oculta aproximadamente 35%), mesmo quando não há uma camada física
        # coerente à frente. Uma pequena mediana ao redor da âncora é robusta
        # a ruído de um pixel sem perder a interpretação de eixo Z.
        anchor_x, anchor_y = anchor or (fruit.width // 2, fruit.height // 2)
        patch_size = max(
            1,
            round(
                min(fruit.width, fruit.height)
                * float(placement.get("z_patch_fraction", 0.2))
            ),
        )
        half_before = patch_size // 2
        half_after = patch_size - half_before
        left = max(0, anchor_x - half_before)
        right = min(fruit.width, anchor_x + half_after)
        top = max(0, anchor_y - half_before)
        bottom = min(fruit.height, anchor_y + half_after)
        patch_opaque = opaque[top:bottom, left:right]
        placement_values = region_depth[top:bottom, left:right][patch_opaque]
        if not len(placement_values):
            placement_values = local_values
        z_offset = float(placement.get("z_offset", 0.0))
        z_offset_jitter = float(placement.get("z_offset_jitter", 0.0))
        if z_offset_jitter > 0:
            # Um offset fixo desloca o limiar igualmente em toda inserção,
            # então quase nenhuma fruta sai totalmente visível nem totalmente
            # oculta: a variação vem só da geometria local, que é estreita.
            # Sortear o offset por tentativa (mesmo rng da posição, then
            # determinístico pela seed) alarga essa distribuição para incluir
            # os dois extremos.
            sampler = rng or random.Random()
            z_offset = sampler.uniform(
                z_offset - z_offset_jitter, z_offset + z_offset_jitter
            )
        z_value = float(np.median(placement_values)) + z_offset
        z_value = float(np.clip(z_value, 0.0, 255.0))
    else:
        raise ValueError(f"z_method desconhecido: {placement['z_method']}")
    if float(np.median(placement_values)) < float(placement["min_depth"]):
        return None
    visibility = (region_depth <= z_value).astype(np.uint8) * 255
    blur = float(config["occlusion"]["edge_blur"])
    if blur > 0:
        visibility = np.asarray(
            Image.fromarray(visibility).filter(ImageFilter.GaussianBlur(blur))
        )
    mask_threshold = config["occlusion"].get("mask_threshold")
    if mask_threshold is not None:
        # O notebook de origem suaviza a topologia da máscara e depois a
        # binariza. Sem esta etapa, boa parte da fruta fica semitransparente
        # e o fundo escuro aparece como manchas em vez de oclusão geométrica.
        visibility = (visibility > round(255 * float(mask_threshold))).astype(
            np.uint8
        ) * 255
    edge_feather_radius = float(config["occlusion"].get("edge_feather_radius", 0.0))
    if edge_feather_radius > 0:
        # edge_blur suaviza a máscara ANTES do limiar e decide a forma da
        # oclusão; num recorte pequeno (poucas dezenas de px) esse blur é
        # uma fração grande do objeto, então precisa ser rebinarizado para
        # não sobrar mancha semitransparente larga. Este segundo blur, bem
        # menor, roda DEPOIS do limiar e só amacia a serrilha de poucos
        # pixels da borda já decidida — um gradiente estreito em vez de um
        # corte geométrico abrupto, sem reabrir a mancha larga.
        visibility = np.asarray(
            Image.fromarray(visibility).filter(
                ImageFilter.GaussianBlur(edge_feather_radius)
            )
        )
    new_alpha = np.rint(alpha_float * visibility / 255.0).astype(np.uint8)
    visible_pixels = int((new_alpha > 8).sum())
    if visible_pixels / original_pixels < float(placement["min_visibility"]):
        return None
    region = canvas.crop((x, y, x + fruit.width, y + fruit.height))
    fruit = _apply_appearance(fruit, region, config["appearance"], rng=rng)
    fruit = _apply_occlusion_contact_shadow(
        fruit, visibility, opaque, config["occlusion"]
    )
    if rng is not None:
        fruit = _apply_cast_shadow(fruit, x, y, depth, opaque, config["occlusion"], rng)
    fruit.putalpha(Image.fromarray(new_alpha))
    return {
        "x": x,
        "y": y,
        "image": fruit,
        "visible_mask": new_alpha,
        "amodal_mask": alpha_original,
        "rect": (0, 0, fruit.width, fruit.height),
        "z": round(z_value, 3),
        "visibility_at_insert": round(visible_pixels / original_pixels, 4),
    }


def _placement(
    canvas: Image.Image,
    depth: np.ndarray,
    fruit: Image.Image,
    config: dict,
    rng: random.Random,
) -> dict | None:
    width, height = canvas.size
    if fruit.width > width or fruit.height > height:
        return None
    alpha_original = np.asarray(fruit.getchannel("A"), dtype=np.uint8)
    alpha_float = alpha_original.astype(np.float32)
    opaque = alpha_original > 8
    original_pixels = int(opaque.sum())
    if original_pixels == 0:
        return None
    placement = config["placement"]
    exclude_bottom = float(placement.get("exclude_bottom_fraction", 0.0))
    for _ in range(int(placement["max_attempts_per_object"])):
        x = rng.randint(0, width - fruit.width)
        y = rng.randint(0, height - fruit.height)
        if exclude_bottom > 0 and (y + fruit.height / 2) > height * (
            1 - exclude_bottom
        ):
            continue
        result = _finish_placement(
            fruit,
            x,
            y,
            alpha_original,
            alpha_float,
            opaque,
            original_pixels,
            canvas,
            depth,
            config,
            anchor=(fruit.width // 2, fruit.height // 2),
            rng=rng,
        )
        if result is not None:
            return result
    return None


def _resolve_depth_scale(proximity: float, depth_scale: dict) -> float:
    near = float(depth_scale["near_scale"])
    far = float(depth_scale["far_scale"])
    return far + (near - far) * proximity


def _placement_with_depth_scale(
    canvas: Image.Image,
    depth: np.ndarray,
    fruit: Image.Image,
    config: dict,
    rng: random.Random,
    depth_scale: dict,
) -> dict | None:
    # A escala de referência (`_scale_cutout`) já fixou uma fração
    # aleatória; aqui essa fração é modulada pela profundidade local do
    # ponto de inserção escolhido, então o tamanho final só é conhecido
    # depois de sortear x,y — ao contrário de `_placement`, que recebe um
    # tamanho fixo e só sorteia a posição.
    width, height = canvas.size
    placement = config["placement"]
    exclude_bottom = float(placement.get("exclude_bottom_fraction", 0.0))
    for _ in range(int(placement["max_attempts_per_object"])):
        cx = rng.randint(0, width - 1)
        cy = rng.randint(0, height - 1)
        if exclude_bottom > 0 and cy > height * (1 - exclude_bottom):
            continue
        proximity = float(depth[cy, cx]) / 255.0
        factor = _resolve_depth_scale(proximity, depth_scale)
        scaled_width = max(1, round(fruit.width * factor))
        scaled_height = max(1, round(fruit.height * factor))
        if scaled_width > width or scaled_height > height:
            continue
        attempt = fruit.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        x = min(max(cx - scaled_width // 2, 0), width - scaled_width)
        y = min(max(cy - scaled_height // 2, 0), height - scaled_height)
        alpha_original = np.asarray(attempt.getchannel("A"), dtype=np.uint8)
        alpha_float = alpha_original.astype(np.float32)
        opaque = alpha_original > 8
        original_pixels = int(opaque.sum())
        if original_pixels == 0:
            continue
        result = _finish_placement(
            attempt,
            x,
            y,
            alpha_original,
            alpha_float,
            opaque,
            original_pixels,
            canvas,
            depth,
            config,
            anchor=(cx - x, cy - y),
            rng=rng,
        )
        if result is not None:
            return result
    return None


def _occlude_prior_instances(instances: list[dict], new_instance: dict) -> None:
    nx, ny = new_instance["x"], new_instance["y"]
    new_mask = new_instance["visible_mask"] > 8
    nh, nw = new_mask.shape
    for instance in instances:
        ix, iy = instance["x"], instance["y"]
        old = instance["visible_mask"]
        oh, ow = old.shape
        left, top = max(ix, nx), max(iy, ny)
        right, bottom = min(ix + ow, nx + nw), min(iy + oh, ny + nh)
        if left >= right or top >= bottom:
            continue
        old_slice = old[top - iy : bottom - iy, left - ix : right - ix]
        new_slice = new_mask[top - ny : bottom - ny, left - nx : right - nx]
        old_slice[new_slice] = 0


def _label_for(
    instance: dict, mode: str, canvas: tuple[int, int], min_pixels: int
) -> str | None:
    if mode == "visible":
        box = _bbox(instance["visible_mask"], threshold=8)
    elif mode == "amodal":
        box = _bbox(instance["amodal_mask"], threshold=8)
    else:
        box = instance["rect"]
    if box is None:
        return None
    left, top, right, bottom = box
    if right - left < min_pixels or bottom - top < min_pixels:
        return None
    left += instance["x"]
    right += instance["x"]
    top += instance["y"]
    bottom += instance["y"]
    width, height = canvas
    center_x = (left + right) / 2 / width
    center_y = (top + bottom) / 2 / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"


def _apply_scene_grading(canvas: Image.Image, grading: dict) -> Image.Image:
    # Os 228 fundos foram fotografados sob luz difusa/nublada, num ângulo à
    # altura dos olhos; as fotos reais anotadas são ensolaradas, céu azul
    # saturado, vistas de baixo para cima na copa. Não há como reproduzir a
    # composição/iluminação sem introduzir ativos externos (fora do escopo),
    # mas contraste, saturação e nitidez mais altos na cena inteira aproximam
    # o "punch" visual da cena composta do observado nas fotos reais.
    graded = canvas
    contrast = float(grading.get("contrast", 1.0))
    if contrast != 1.0:
        graded = ImageEnhance.Contrast(graded).enhance(contrast)
    saturation = float(grading.get("saturation", 1.0))
    if saturation != 1.0:
        graded = ImageEnhance.Color(graded).enhance(saturation)
    brightness = float(grading.get("brightness", 1.0))
    if brightness != 1.0:
        graded = ImageEnhance.Brightness(graded).enhance(brightness)
    sharpen_percent = int(grading.get("sharpen_percent", 0))
    if sharpen_percent > 0:
        graded = graded.filter(
            ImageFilter.UnsharpMask(
                radius=float(grading.get("sharpen_radius", 2.0)),
                percent=sharpen_percent,
                threshold=int(grading.get("sharpen_threshold", 2)),
            )
        )
    return graded


def _save_jpeg_atomic(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.jpg")
    image.convert("RGB").save(temporary, format="JPEG", quality=quality, subsampling=0)
    temporary.replace(path)


def _build_debug_panel(
    canvas: Image.Image, depth_image: Image.Image, instances: list[dict]
) -> Image.Image:
    # Lado a lado: imagem final | mapa de profundidade (já suavizado, o que
    # o limiar de visibilidade realmente enxerga) com a máscara de cada
    # fruta sobreposta, verde onde ficou visível e vermelho onde a
    # profundidade cortou. Isso mostra exatamente por que cada oclusão
    # ficou do jeito que ficou, sem precisar reconstruir o cálculo à mão.
    depth_rgb = depth_image.convert("RGB").resize(canvas.size)
    draw = ImageDraw.Draw(depth_rgb, "RGBA")
    for instance in instances:
        x, y = instance["x"], instance["y"]
        visible_mask = instance["visible_mask"]
        amodal_mask = instance["amodal_mask"]
        height, width = visible_mask.shape
        occluded = (amodal_mask > 8) & (visible_mask <= 8)
        visible = visible_mask > 8
        tint = np.zeros((height, width, 4), dtype=np.uint8)
        tint[visible] = (40, 220, 90, 130)
        tint[occluded] = (230, 40, 40, 160)
        tint_image = Image.fromarray(tint)
        depth_rgb.paste(tint_image, (x, y), tint_image)
        visibility_percent = round(
            100 * float(instance.get("visibility_at_insert", 0.0))
        )
        draw.rectangle(
            [x, y, x + width - 1, y + height - 1], outline=(255, 255, 0, 220), width=1
        )
        draw.text((x + 2, y + 2), f"{visibility_percent}%", fill=(255, 255, 0, 255))
    panel = Image.new("RGB", (canvas.width * 2 + 4, canvas.height), (20, 20, 20))
    panel.paste(canvas, (0, 0))
    panel.paste(depth_rgb, (canvas.width + 4, 0))
    return panel


def _render_one(task: dict) -> dict:
    split_name = task["split"]
    index = task["index"]
    output = Path(task["output"])
    name = f"{split_name}_{index:06d}"
    image_path = output / "images" / split_name / f"{name}.jpg"
    label_path = output / "labels" / split_name / f"{name}.txt"
    metadata_path = output / "metadata" / split_name / f"{name}.json"
    debug = bool(task.get("debug", False))
    debug_path = output / "images_debug" / split_name / f"{name}_debug.jpg"
    if (
        image_path.exists()
        and label_path.exists()
        and metadata_path.exists()
        and (not debug or debug_path.exists())
        and not task["force"]
    ):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    config = task["config"]
    rng = random.Random(task["sample_seed"])
    backgrounds = task["assets"]["backgrounds"]
    cutouts = task["assets"]["cutouts"]
    pair = rng.choice(backgrounds)
    canvas_size = tuple(map(int, config["canvas"]))
    canvas, depth_image = _open_background_pair(
        Path(pair["image"]), Path(pair["depth"]), canvas_size
    )
    grading = config["output"].get("scene_grading")
    if grading and grading.get("enabled", False):
        # Aplicado só no fundo, antes de colar qualquer fruta: um realce
        # aplicado na cena inteira já composta também "esculpe" as frutas
        # já ajustadas pelo hsv_cast, empilhando saturação/nitidez até
        # ficarem artificiais (achado de revisão visual). O objetivo é só
        # aproximar o "punch" do fundo nublado do observado nas fotos reais,
        # não realçar a fruta de novo.
        canvas = _apply_scene_grading(canvas, grading)
    depth_smooth_radius = float(config["occlusion"].get("depth_smooth_radius", 0.0))
    if depth_smooth_radius > 0:
        # Estimadores de profundidade de alta resolução (ex. DepthPro)
        # captam ruído pixel a pixel que gera oclusões "mosqueadas" em vez
        # de manchas coerentes de folha/galho. Um blur leve no mapa de
        # profundidade (não na imagem final) funde esse ruído em regiões
        # maiores antes do limiar de visibilidade, sem afetar a nitidez da
        # cena composta.
        depth_image = depth_image.filter(ImageFilter.GaussianBlur(depth_smooth_radius))
    depth = np.asarray(depth_image, dtype=np.uint8)
    requested = rng.randint(
        int(config["objects"]["min"]), int(config["objects"]["max"])
    )
    if requested <= len(cutouts):
        chosen = rng.sample(cutouts, requested)
    else:
        chosen = rng.sample(cutouts, len(cutouts)) + [
            rng.choice(cutouts) for _ in range(requested - len(cutouts))
        ]
    instances = []
    rejected = Counter()
    for cutout_path in chosen:
        fruit = _open_cutout_cached(cutout_path)
        fruit = _scale_cutout(fruit, config["objects"], rng, canvas_size)
        rotation = float(config["objects"]["rotation_degrees"])
        if rotation:
            fruit = fruit.rotate(
                rng.uniform(-rotation, rotation),
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            fruit = _trim_alpha(fruit)
        if fruit.width > canvas.width or fruit.height > canvas.height:
            rejected["larger_than_canvas"] += 1
            continue
        depth_scale = config["objects"].get("depth_scale")
        if depth_scale and depth_scale.get("enabled", False):
            instance = _placement_with_depth_scale(
                canvas, depth, fruit, config, rng, depth_scale
            )
        else:
            instance = _placement(canvas, depth, fruit, config, rng)
        if instance is None:
            rejected["placement_or_visibility"] += 1
            continue
        _occlude_prior_instances(instances, instance)
        canvas.paste(
            instance["image"], (instance["x"], instance["y"]), instance["image"]
        )
        instance["cutout"] = Path(cutout_path).name
        instances.append(instance)

    labels = []
    annotation_mode = config["annotation"]["mode"]
    for instance in instances:
        label = _label_for(
            instance,
            annotation_mode,
            canvas_size,
            int(config["annotation"]["min_box_pixels"]),
        )
        if label is not None:
            labels.append(label)
        else:
            rejected["final_box_too_small"] += 1
    _save_jpeg_atomic(canvas, image_path, int(config["output"]["jpeg_quality"]))
    atomic_write_text(
        label_path, "\n".join(labels) + ("\n" if labels else ""), durable=False
    )
    if debug:
        panel = _build_debug_panel(canvas, depth_image, instances)
        _save_jpeg_atomic(panel, debug_path, int(config["output"]["jpeg_quality"]))
    record = {
        "id": name,
        "split": split_name,
        "seed": task["sample_seed"],
        "background": Path(pair["image"]).name,
        "depth": Path(pair["depth"]).name,
        "requested_objects": requested,
        "inserted_objects": len(instances),
        "annotations": len(labels),
        "cutouts": [instance["cutout"] for instance in instances],
        "rejected": dict(rejected),
        "image": image_path.relative_to(output).as_posix(),
        "label": label_path.relative_to(output).as_posix(),
    }
    # Sidecars são regeneráveis e o rename continua atômico; evitar um fsync por
    # amostra é especialmente importante em volumes de rede usados por servidores.
    atomic_write_json(metadata_path, record, durable=False)
    return record


_WORKER_CONTEXT: dict | None = None


def _initialize_worker(context: dict) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    _clear_image_caches()


def _render_compact(task: tuple[str, int, int]) -> dict:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker de síntese sem contexto")
    split_name, index, sample_seed = task
    return _render_one(
        {
            "split": split_name,
            "index": index,
            "sample_seed": sample_seed,
            "output": _WORKER_CONTEXT["output"],
            "config": _WORKER_CONTEXT["config"],
            "assets": _WORKER_CONTEXT["assets"][split_name],
            "force": _WORKER_CONTEXT["force"],
            "debug": _WORKER_CONTEXT["debug"],
        }
    )


def _background_sort_key(
    task: tuple[str, int, int], assets: dict[str, dict]
) -> tuple[str, str, int]:
    split_name, index, sample_seed = task
    rng = random.Random(sample_seed)
    pair = rng.choice(assets[split_name]["backgrounds"])
    return split_name, pair["image"], index


def generate_dataset(
    asset_root: Path,
    output_root: Path,
    config: dict,
    *,
    train_ratio: float,
    split_seed: int,
    workers: int = 1,
    force: bool = False,
    debug: bool = False,
) -> dict:
    validate_synthesis_config(config)
    asset_split = create_asset_split(
        asset_root, train_ratio=train_ratio, seed=split_seed, force=False
    )
    config_hash = stable_hash(config, 24)
    config_marker = output_root / "generation_config.json"
    if config_marker.exists() and not force:
        previous = json.loads(config_marker.read_text(encoding="utf-8"))
        expected_marker = {
            "config_hash": config_hash,
            "asset_split_fingerprint": asset_split["source_fingerprint"],
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        }
        mismatched = [
            key for key, value in expected_marker.items() if previous.get(key) != value
        ]
        if mismatched:
            raise RuntimeError(
                f"{output_root} foi gerado com outro protocolo "
                f"({', '.join(mismatched)}); use --force"
            )
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        config_marker,
        {
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "config_hash": config_hash,
            "config": config,
            "asset_split_fingerprint": asset_split["source_fingerprint"],
        },
    )

    tasks: list[tuple[str, int, int]] = []
    assets_by_split = {}
    for split_name in ("train", "val"):
        source_assets = asset_split["splits"][split_name]
        assets = {
            "backgrounds": [
                {
                    "image": str(asset_root / item["image"]),
                    "depth": str(asset_root / item["depth"]),
                }
                for item in source_assets["backgrounds"]
            ],
            "cutouts": [str(asset_root / path) for path in source_assets["cutouts"]],
        }
        if not assets["backgrounds"] or not assets["cutouts"]:
            raise RuntimeError(f"split de ativos vazio: {split_name}")
        assets_by_split[split_name] = assets
        count = int(config["images"][split_name])
        for index in range(count):
            sample_seed = int(
                stable_hash(
                    [
                        int(config["seed"]),
                        config_hash,
                        asset_split["source_fingerprint"],
                        split_name,
                        index,
                    ],
                    16,
                ),
                16,
            )
            tasks.append((split_name, index, sample_seed))
    # Agrupar fundos aumenta o reaproveitamento do cache sem alterar a semente ou
    # a composição de nenhuma cena. O manifesto é ordenado novamente ao final.
    tasks.sort(key=lambda task: _background_sort_key(task, assets_by_split))
    context = {
        "output": str(output_root),
        "config": config,
        "assets": assets_by_split,
        "force": force,
        "debug": debug,
    }
    _clear_image_caches()
    if workers <= 1:
        _initialize_worker(context)
        records = [_render_compact(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(context,),
        ) as executor:
            records = list(executor.map(_render_compact, tasks, chunksize=4))
    records.sort(key=lambda item: (item["split"], item["id"]))
    manifest_text = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records
    )
    atomic_write_text(output_root / "manifest.jsonl", manifest_text)
    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "poncan"},
    }
    atomic_write_text(
        output_root / "data.yaml",
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
    )
    summary = {
        "name": config["name"],
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "config_hash": config_hash,
        "asset_split_fingerprint": asset_split["source_fingerprint"],
        "images": dict(Counter(item["split"] for item in records)),
        "annotations": {
            split_name: sum(
                item["annotations"] for item in records if item["split"] == split_name
            )
            for split_name in ("train", "val")
        },
        "negative_images": {
            split_name: sum(
                item["annotations"] == 0
                for item in records
                if item["split"] == split_name
            )
            for split_name in ("train", "val")
        },
        "manifest_sha256": sha256_file(output_root / "manifest.jsonl"),
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary
