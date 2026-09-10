from __future__ import annotations

import concurrent.futures
import json
import math
import random
import shutil
import tempfile
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
    link_or_copy,
    relative_or_absolute,
    sha256_file,
    stable_hash,
)

GENERATOR_SCHEMA_VERSION = 8
ASSET_CATALOG_SCHEMA_VERSION = 2
SCENE_SPLIT_SCHEMA_VERSION = 1


def find_depth_map(background: Path, depth_directory: Path) -> Path | None:
    stems = (f"{background.stem}_depth", f"{background.stem}_map", background.stem)
    for stem in stems:
        for suffix in sorted(IMAGE_SUFFIXES):
            candidate = depth_directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def create_asset_catalog(asset_root: Path, *, force: bool = False) -> dict:
    """Indexa todos os ativos elegíveis sem separá-los por train/val."""
    target = asset_root / "asset_catalog.json"
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

    # Conteúdo, não apenas tamanho: dois arquivos diferentes com o mesmo
    # comprimento precisam invalidar a identidade da geração.
    hashes = {path: sha256_file(path) for path in
              set(pair_lookup) | set(pair_lookup.values()) | set(cutouts)}
    source_fingerprint = stable_hash(
        [
            (
                rel(path),
                hashes[path],
                rel(pair_lookup[path]),
                hashes[pair_lookup[path]],
            )
            for path in sorted(pair_lookup)
        ]
        + [(rel(path), hashes[path]) for path in sorted(cutouts)],
        24,
    )
    if target.exists() and not force:
        previous = json.loads(target.read_text(encoding="utf-8"))
        if (
            previous.get("version") == ASSET_CATALOG_SCHEMA_VERSION
            and previous.get("source_fingerprint") == source_fingerprint
        ):
            return previous

    result = {
        "version": ASSET_CATALOG_SCHEMA_VERSION,
        "asset_root": relative_or_absolute(asset_root),
        "source_fingerprint": source_fingerprint,
        "sha256": {rel(path): digest for path, digest in sorted(hashes.items())},
        "assets": {
            "backgrounds": [
                {"image": rel(path), "depth": rel(pair_lookup[path])}
                for path in sorted(pair_lookup)
            ],
            "cutouts": [rel(path) for path in sorted(cutouts)],
        },
        "orphans": {
            "depth_maps": sorted(
                rel(path)
                for path in image_files(depth_dir)
                if path not in set(pair_lookup.values())
            )
        },
    }
    atomic_write_json(target, result)
    return result


def create_scene_split(total: int, train_ratio: float, seed: int) -> dict:
    """Particiona IDs de cenas já definidos, sem afetar sua composição."""
    if total <= 0:
        raise ValueError("o total de cenas sintéticas deve ser positivo")
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio deve estar entre 0 e 1")
    scene_indices = list(range(total))
    random.Random(seed + int(stable_hash("synthetic_scene_split", 8), 16)).shuffle(
        scene_indices
    )
    train_count = round(total * train_ratio)
    if total > 1 and 0.0 < train_ratio < 1.0:
        train_count = min(max(train_count, 1), total - 1)
    return {
        "version": SCENE_SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "train_ratio": train_ratio,
        "total": total,
        "splits": {
            "train": sorted(scene_indices[:train_count]),
            "val": sorted(scene_indices[train_count:]),
        },
    }


def validate_synthesis_config(config: dict) -> None:
    if config.get("sampling", {}).get("mode", "legacy") not in {"legacy", "paired-v1"}:
        raise ValueError("sampling.mode deve ser legacy ou paired-v1")
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
    images = config["images"]
    if set(images) != {"total"} or int(images["total"]) <= 0:
        raise ValueError(
            "images deve declarar somente total; train/val são definidos depois da geração"
        )
    width, height = map(int, config["canvas"])
    if width <= 0 or height <= 0:
        raise ValueError("canvas deve ser positivo")
    objects = config["objects"]
    if any(isinstance(objects[k], bool) or not isinstance(objects[k], int)
           for k in ("min", "max")):
        raise ValueError("objects.min e objects.max devem ser inteiros")
    if not 0 <= objects["min"] <= objects["max"]:
        raise ValueError("intervalo de objetos inválido")
    if not isinstance(config.get("augmentation", {}).get("horizontal_flip", False), bool):
        raise ValueError("augmentation.horizontal_flip deve ser booleano")
    dense = objects.get("dense")
    if dense:
        if not 0 <= float(dense.get("probability", 0.0)) <= 1:
            raise ValueError("objects.dense.probability deve estar entre 0 e 1")
        if not 0 <= int(dense["min"]) <= int(dense["max"]):
            raise ValueError("objects.dense: intervalo inválido")
        if not isinstance(dense.get("scale_with_count", False), bool):
            raise ValueError("objects.dense.scale_with_count deve ser booleano")
        if dense.get("scale_with_count", False) and int(objects["max"]) <= 0:
            raise ValueError("scale_with_count requer objects.max positivo")
    if not 0 < float(objects["min_scale"]) <= float(objects["max_scale"]):
        raise ValueError("intervalo de escala inválido")
    depth_scale = objects.get("depth_scale")
    if (
        depth_scale
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
    if not isinstance(config["placement"].get("require_vegetation", False), bool):
        raise ValueError("placement.require_vegetation deve ser booleano")
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
    hsv_cast = appearance.get("hsv_cast")
    if hsv_cast:
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
    ripeness = appearance.get("ripeness")
    if ripeness and ripeness.get("enabled", False):
        if not 0 <= float(ripeness.get("fraction_affected", 0.0)) <= 1:
            raise ValueError("appearance.ripeness.fraction_affected deve estar entre 0 e 1")
        strength_range = ripeness.get("strength_range", [0.3, 1.0])
        lo, hi = float(strength_range[0]), float(strength_range[1])
        if not 0 <= lo <= hi <= 1:
            raise ValueError(
                "appearance.ripeness.strength_range deve ser crescente dentro de 0 e 1"
            )
        if not 0 <= float(ripeness.get("green_hue_degrees", 105.0)) <= 360:
            raise ValueError("appearance.ripeness.green_hue_degrees deve estar entre 0 e 360")
        if not 0 <= float(ripeness.get("saturation_scale", 1.0)) <= 2:
            raise ValueError("appearance.ripeness.saturation_scale deve estar entre 0 e 2")
        if not 0 <= float(ripeness.get("gloss_reduction", 0.0)) <= 1:
            raise ValueError("appearance.ripeness.gloss_reduction deve estar entre 0 e 1")
    exposure = appearance.get("exposure_jitter")
    if exposure and exposure.get("enabled", False):
        if not 0 <= float(exposure.get("probability", 0.0)) <= 1:
            raise ValueError(
                "appearance.exposure_jitter.probability deve estar entre 0 e 1"
            )
        low, high = exposure.get("range", [1.0, 1.0])
        if not 0 < float(low) <= float(high):
            raise ValueError(
                "appearance.exposure_jitter.range deve ser crescente e positivo"
            )
        if not 0 <= float(exposure.get("saturation_pull", 0.0)) <= 1:
            raise ValueError(
                "appearance.exposure_jitter.saturation_pull deve estar entre 0 e 1"
            )
    grading = config["output"].get("scene_grading")
    if grading:
        for key in ("contrast", "saturation", "brightness"):
            if key in grading and float(grading[key]) <= 0:
                raise ValueError(f"output.scene_grading.{key} deve ser positivo")
        for key in ("brightness", "contrast", "saturation"):
            jitter = grading.get(f"{key}_jitter")
            if jitter is not None and not (
                len(jitter) == 2 and 0 < float(jitter[0]) <= float(jitter[1])
            ):
                raise ValueError(
                    f"output.scene_grading.{key}_jitter deve ser [min, max] positivo e crescente"
                )
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
    if contact_shadow:
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
    if cast_shadow:
        probability = float(cast_shadow.get("probability", 0.4))
        strength = float(cast_shadow.get("strength", 0.25))
        min_coverage = float(cast_shadow.get("min_coverage", 0.3))
        max_coverage = float(cast_shadow.get("max_coverage", 0.6))
        blur_radius = float(cast_shadow.get("blur_radius", 3.0))
        if not 0 <= probability <= 1:
            raise ValueError("occlusion.cast_shadow.probability deve estar entre 0 e 1")
        if not 0 <= strength <= 1:
            raise ValueError("occlusion.cast_shadow.strength deve estar entre 0 e 1")
        if not 0 < min_coverage < max_coverage <= 1:
            raise ValueError(
                "occlusion.cast_shadow requer 0 < min_coverage < max_coverage <= 1"
            )
        if blur_radius < 0:
            raise ValueError("occlusion.cast_shadow.blur_radius não pode ser negativo")
        if float(cast_shadow.get("offset_fraction", 0.35)) < 0:
            raise ValueError(
                "occlusion.cast_shadow.offset_fraction não pode ser negativo"
            )
        if float(cast_shadow.get("light_angle_jitter_degrees", 20.0)) < 0:
            raise ValueError(
                "occlusion.cast_shadow.light_angle_jitter_degrees não pode ser negativo"
            )
        shapes = cast_shadow.get("shapes", _SHADOW_SHAPES)
        unknown_shapes = set(shapes) - set(_SHADOW_SHAPES)
        if not shapes or unknown_shapes:
            raise ValueError(
                f"occlusion.cast_shadow.shapes deve ser um subconjunto não vazio "
                f"de {_SHADOW_SHAPES}"
            )


@lru_cache(maxsize=8)
def _open_background_pair_cached(
    background_path: str, depth_path: str, size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    with Image.open(background_path) as opened:
        background = ImageOps.exif_transpose(opened).convert("RGB")
    with Image.open(depth_path) as opened:
        depth = ImageOps.exif_transpose(opened).convert("L")
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


def _count_adjusted_scale(objects: dict, requested: int) -> dict:
    """Limita o crescimento da área projetada em cenas acima do regime esparso.

    Não é uma lei física: é uma hipótese de câmera mais distante, ativada
    explicitamente. Não altera a contagem nem consome aleatoriedade.
    """
    if not objects.get("dense", {}).get("scale_with_count", False):
        return objects
    if requested <= int(objects["max"]):
        return objects
    factor = math.sqrt(float(objects["max"]) / requested)
    return {
        **objects,
        "min_scale": float(objects["min_scale"]) * factor,
        "max_scale": float(objects["max_scale"]) * factor,
    }


def _scale_cutout(
    image: Image.Image, config: dict, rng: random.Random, canvas: tuple[int, int]
) -> Image.Image:
    # A escala é sempre relativa ao canvas: o tamanho aparente da fruta na
    # imagem é o que precisa casar com a distribuição real de caixas, e ele não
    # depende da resolução do recorte-fonte.
    fraction = rng.uniform(float(config["min_scale"]), float(config["max_scale"]))
    scale = fraction * min(canvas) / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


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


def _apply_ripeness_shift(
    fruit: Image.Image,
    ripeness: dict,
    rng: random.Random,
) -> Image.Image:
    # Os 127 recortes-fonte foram fotografados só em ponto de colheita
    # (maduros): sem essa etapa, nenhuma fruta sintética fica verde, mesmo
    # que o pomar real tenha frutos em vários estágios de maturação. Gira o
    # matiz de uma fração dos objetos em direção ao verde ANTES do hsv_cast
    # ambiental (que continua sendo aplicado por cima, como iluminação de
    # cena), então o resultado é "fruta verde sob a luz daquele fundo", não
    # um filtro plano sobre a fruta madura final.
    if rng.random() > float(ripeness.get("fraction_affected", 0.0)):
        return fruit
    alpha = fruit.getchannel("A")
    rgb = fruit.convert("RGB")
    h, s, v = rgb.convert("HSV").split()
    h_array = np.asarray(h, dtype=np.float32)
    strength_lo, strength_hi = ripeness.get("strength_range", [0.3, 1.0])
    strength = rng.uniform(float(strength_lo), float(strength_hi))
    target_hue = float(ripeness.get("green_hue_degrees", 105.0)) / 360.0 * 255.0
    hue_diff = ((target_hue - h_array + 128) % 256) - 128
    h_new = (h_array + hue_diff * strength) % 256
    # A folhagem já ocupa a mesma faixa de matiz alvo; sem diferenciar
    # saturação, um fruto verde fica cromaticamente equivalente a um
    # aglomerado de folhas. Reduzir a saturação da fruta abaixo da folha
    # real preserva a distinção entre as duas.
    saturation_scale = float(ripeness.get("saturation_scale", 1.0))
    s_array = np.asarray(s, dtype=np.float32) * saturation_scale
    # Fruta "de vez" real tem casca mais fosca que a madura (menos cera
    # visível) — o brilho especular concentrado é justamente o que soma com
    # o matiz/saturação pra ler como "plástico" em vez de fruta. Achata só
    # os pixels acima do percentil 75 de V do próprio recorte (o highlight),
    # não a fruta inteira, senão ela escurece de forma plana e artificial.
    gloss_reduction = float(ripeness.get("gloss_reduction", 0.0))
    v_array = np.asarray(v, dtype=np.float32)
    if gloss_reduction > 0:
        alpha_array = np.asarray(alpha, dtype=np.float32)
        opaque = alpha_array > 8
        if opaque.any():
            highlight_threshold = float(np.percentile(v_array[opaque], 75))
            excess = np.clip(v_array - highlight_threshold, 0, None)
            v_array = v_array - excess * gloss_reduction
    blended = Image.merge(
        "HSV",
        [
            Image.fromarray(np.clip(h_new, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(s_array, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(v_array, 0, 255).astype(np.uint8)),
        ],
    ).convert("RGB")
    result = blended.convert("RGBA")
    result.putalpha(alpha)
    return result


def _apply_exposure_jitter(
    fruit: Image.Image, exposure: dict, rng: random.Random
) -> Image.Image:
    # O hsv_cast puxa a fruta na direcao da cor do fundo local, entao ele so
    # sabe REDUZIR o contraste entre fruta e cena. Nas fotos reais o contraste
    # se espalha muito mais para os dois lados: fruta em sombra profunda quase
    # some no meio da folhagem, e fruta em sol direto estoura contra a copa
    # escura. Um fator de exposicao por instancia, aplicado depois do cast e
    # independente do fundo, e o unico ponto do modelo de aparencia capaz de
    # abrir as duas caudas.
    if rng.random() > float(exposure.get("probability", 0.0)):
        return fruit
    low, high = exposure.get("range", [1.0, 1.0])
    factor = rng.uniform(float(low), float(high))
    alpha = fruit.getchannel("A")
    h, s, v = fruit.convert("RGB").convert("HSV").split()
    v_array = np.asarray(v, dtype=np.float32) * factor
    # Nas fotos reais a fruta do quartil escuro e menos saturada que a do
    # quartil claro (razao 0,80 no treino manual, 0,98 no CitDet). Dessaturar
    # os dois extremos inverte isso: mede-se 1,12 no sintetico, ou seja, sol
    # lavado e sombra vivida. Com desaturate_shade_only a perda vale so para
    # fator abaixo de 1, deixando a fruta de sol intacta.
    saturation_pull = float(exposure.get("saturation_pull", 0.0))
    s_array = np.asarray(s, dtype=np.float32)
    if saturation_pull > 0:
        if exposure.get("desaturate_shade_only", False):
            deviation = max(0.0, 1.0 - factor)
        else:
            deviation = min(abs(factor - 1.0), 1.0)
        s_array = s_array * (1.0 - saturation_pull * deviation)
    blended = Image.merge(
        "HSV",
        [
            h,
            Image.fromarray(np.clip(s_array, 0, 255).astype(np.uint8)),
            Image.fromarray(np.clip(v_array, 0, 255).astype(np.uint8)),
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
    ripeness = appearance.get("ripeness")
    if ripeness and ripeness.get("enabled", False) and rng is not None:
        fruit = _apply_ripeness_shift(fruit, ripeness, rng)
    fruit = _apply_appearance_hsv_cast(
        fruit, background_region, appearance["hsv_cast"], rng
    )
    exposure = appearance.get("exposure_jitter")
    if exposure and exposure.get("enabled", False) and rng is not None:
        fruit = _apply_exposure_jitter(fruit, exposure, rng)
    return fruit


def _apply_occlusion_contact_shadow(
    fruit: Image.Image,
    visibility: np.ndarray,
    opaque: np.ndarray,
    occlusion: dict,
) -> Image.Image:
    contact_shadow = occlusion.get("contact_shadow")
    if not contact_shadow:
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


def _shadow_shape_ellipse(
    width: int, height: int, angle: float, coverage: float, offset_fraction: float
) -> np.ndarray:
    center_x = width / 2.0 + math.cos(angle) * width * offset_fraction
    center_y = height / 2.0 + math.sin(angle) * height * offset_fraction
    scale = math.sqrt(coverage) * 0.75
    radius_x = max(width * scale, 1.0)
    radius_y = max(height * scale, 1.0)
    ys, xs = np.mgrid[0:height, 0:width]
    ellipse = ((xs - center_x) / radius_x) ** 2 + ((ys - center_y) / radius_y) ** 2
    return ellipse <= 1.0


def _shadow_shape_band(
    width: int,
    height: int,
    angle: float,
    coverage: float,
    offset_fraction: float,
    rng: random.Random,
) -> np.ndarray:
    # Faixa alongada perpendicular à luz, imitando a sombra de um galho ou
    # nervura fina cruzando a fruta, em vez de uma mancha arredondada.
    center_x = width / 2.0 + math.cos(angle) * width * offset_fraction
    center_y = height / 2.0 + math.sin(angle) * height * offset_fraction
    band_angle = angle + math.pi / 2 + rng.uniform(-0.35, 0.35)
    band_width = max(width, height) * coverage
    ys, xs = np.mgrid[0:height, 0:width]
    perpendicular = -(xs - center_x) * math.sin(band_angle) + (
        ys - center_y
    ) * math.cos(band_angle)
    return np.abs(perpendicular) <= band_width / 2.0


def _shadow_shape_blob(
    width: int,
    height: int,
    angle: float,
    coverage: float,
    offset_fraction: float,
    rng: random.Random,
) -> np.ndarray:
    # União de 2-3 lóbulos circulares deslocados: silhueta mais irregular
    # que uma elipse única, mais parecida com sombra de folhas dispersas.
    base_x = width / 2.0 + math.cos(angle) * width * offset_fraction
    base_y = height / 2.0 + math.sin(angle) * height * offset_fraction
    ys, xs = np.mgrid[0:height, 0:width]
    lobes = rng.randint(2, 3)
    lobe_scale = math.sqrt(max(coverage / lobes, 0.05) * 1.6) * 0.6
    radius = max(min(width, height) * lobe_scale, 1.0)
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(lobes):
        lobe_x = base_x + rng.uniform(-width * 0.25, width * 0.25)
        lobe_y = base_y + rng.uniform(-height * 0.25, height * 0.25)
        mask |= ((xs - lobe_x) ** 2 + (ys - lobe_y) ** 2) <= radius**2
    return mask


_SHADOW_SHAPES = ("ellipse", "band", "blob")


def _apply_cast_shadow(
    fruit: Image.Image,
    opaque: np.ndarray,
    occlusion: dict,
    rng: random.Random,
) -> Image.Image:
    cast_shadow = occlusion.get("cast_shadow")
    if not cast_shadow:
        return fruit
    probability = min(max(float(cast_shadow.get("probability", 0.4)), 0.0), 1.0)
    if rng.random() > probability:
        return fruit
    height, width = opaque.shape
    # Amostrar profundidade deslocada não funcionou: numa fruta de poucas
    # dezenas de pixels o mapa não tem resolução pra desenhar uma silhueta
    # com contraste real, e a "sombra" saía quase uniforme (escurecia tudo
    # por igual, sem parte clara/parte escura) — visualmente imperceptível.
    # Formas procedurais garantem contraste real dentro da fruta; o
    # deslocamento na direção da luz ainda dá uma noção de ângulo (a sombra
    # cai do lado oposto à luz, como autossombra). Várias formas (não só
    # elipse) aumentam a variabilidade visual entre instâncias.
    base_angle = float(cast_shadow.get("light_angle_degrees", 315.0))
    angle_jitter = float(cast_shadow.get("light_angle_jitter_degrees", 20.0))
    angle = math.radians(base_angle + rng.uniform(-angle_jitter, angle_jitter))
    coverage = rng.uniform(
        float(cast_shadow.get("min_coverage", 0.3)),
        float(cast_shadow.get("max_coverage", 0.6)),
    )
    offset_fraction = float(cast_shadow.get("offset_fraction", 0.35))
    shapes = cast_shadow.get("shapes", _SHADOW_SHAPES)
    shape = rng.choice(shapes)
    if shape == "ellipse":
        mask = _shadow_shape_ellipse(width, height, angle, coverage, offset_fraction)
    elif shape == "band":
        mask = _shadow_shape_band(width, height, angle, coverage, offset_fraction, rng)
    elif shape == "blob":
        mask = _shadow_shape_blob(width, height, angle, coverage, offset_fraction, rng)
    else:
        raise ValueError(f"occlusion.cast_shadow.shapes desconhecido: {shape}")
    shadow_mask = mask.astype(np.uint8) * 255
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
    appearance_rng = rng
    if rng is not None and config.get("sampling", {}).get("mode") == "paired-v1":
        # A aparência não deve consumir os sorteios da geometria. Assim,
        # desligar uma transformação conserva as posições e as caixas.
        appearance_rng = random.Random()
        appearance_rng.setstate(rng.getstate())
    fruit = _apply_appearance(fruit, region, config["appearance"], rng=appearance_rng)
    fruit = _apply_occlusion_contact_shadow(
        fruit, visibility, opaque, config["occlusion"]
    )
    if rng is not None:
        fruit = _apply_cast_shadow(fruit, opaque, config["occlusion"], appearance_rng)
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


def _vegetation_support(canvas: Image.Image) -> np.ndarray:
    """Indício cromático de vegetação, sem rótulos nem limiar ajustável.

    Excesso de verde normalizado e separação por variância entre classes
    (Otsu). Não distingue grama de copa nem certifica suporte em um galho.
    """
    rgb = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess = (2 * green - red - blue) / np.maximum(rgb.sum(axis=2), 1)
    signal = np.rint(np.clip(excess, 0, 1) * 255).astype(np.uint8)
    histogram = np.bincount(signal.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    mass = np.cumsum(probabilities)
    moment = np.cumsum(probabilities * np.arange(256))
    variance = (moment[-1] * mass - moment) ** 2 / np.maximum(mass * (1 - mass), 1e-12)
    threshold = int(np.argmax(variance))
    return signal > threshold


def _placement(
    canvas: Image.Image,
    depth: np.ndarray,
    fruit: Image.Image,
    config: dict,
    rng: random.Random,
    support: np.ndarray | None = None,
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
        if support is not None and not support[y + fruit.height // 2, x + fruit.width // 2]:
            continue
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
    support: np.ndarray | None = None,
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
        if support is not None and not support[cy, cx]:
            continue
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


def _grading_factors(grading: dict, seed: int) -> dict:
    """Multiplicador por cena para cada eixo de grading que declarar faixa.

    Os 228 fundos foram fotografados sob luz difusa e saem quase uniformes:
    a amplitude p5-p95 medida no pool sintético é 19,0 em brilho e 17,1 em
    contraste, contra 32,5/30,3 no treino manual e 35,4/36,7 no CitDet. Um
    fator por cena espalha o conjunto sem tocar em geometria.

    Cada eixo tem seu próprio fluxo, derivado da semente da cena: ligar ou
    desligar um deles não desloca os sorteios dos outros nem os da geometria,
    o que preserva o pareamento de cenas do modo `paired-v1`.
    """
    factors = {}
    for key in ("brightness", "contrast", "saturation"):
        span = grading.get(f"{key}_jitter")
        if span:
            stream = random.Random(int(stable_hash([seed, "grading", key], 16), 16))
            factors[key] = stream.uniform(float(span[0]), float(span[1]))
    return factors


def _apply_scene_grading(
    canvas: Image.Image, grading: dict, factors: dict | None = None
) -> Image.Image:
    # Os 228 fundos foram fotografados sob luz difusa/nublada, num ângulo à
    # altura dos olhos; as fotos reais anotadas são ensolaradas, céu azul
    # saturado, vistas de baixo para cima na copa. Não há como reproduzir a
    # composição/iluminação sem introduzir ativos externos (fora do escopo),
    # mas contraste, saturação e nitidez mais altos na cena inteira aproximam
    # o "punch" visual da cena composta do observado nas fotos reais.
    graded = canvas
    factors = factors or {}
    contrast = float(grading.get("contrast", 1.0)) * factors.get("contrast", 1.0)
    saturation = float(grading.get("saturation", 1.0)) * factors.get("saturation", 1.0)
    brightness = float(grading.get("brightness", 1.0)) * factors.get("brightness", 1.0)

    def apply_contrast(image):
        return ImageEnhance.Contrast(image).enhance(contrast) if contrast != 1.0 else image

    def apply_saturation(image):
        return ImageEnhance.Color(image).enhance(saturation) if saturation != 1.0 else image

    def apply_brightness(image):
        return (
            ImageEnhance.Brightness(image).enhance(brightness)
            if brightness != 1.0
            else image
        )

    # Contraste antes de brilho satura os claros em 255 e só depois multiplica o
    # resultado já ceifado: com contrast=1.2 e brightness=0.82 nenhum pixel da
    # cena passa de 209, enquanto as duas coletas reais chegam a 243-255. A
    # ordem fotográfica normal é exposição primeiro, contraste depois, e nessa
    # ordem a cauda de destaques sobrevive sem alterar a luminância média.
    if grading.get("exposure_first", False):
        order = (apply_brightness, apply_contrast, apply_saturation)
    else:
        order = (apply_contrast, apply_saturation, apply_brightness)
    for step in order:
        graded = step(graded)
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


def _sample_object_count(objects: dict, rng: random.Random) -> int:
    """U simétrico em intervalos inteiros iguais, sem parâmetro de curvatura."""
    dense = objects.get("dense")
    if dense:
        # Leitura das receitas históricas, preservando seus sorteios.
        if rng.random() < float(dense.get("probability", 0.0)):
            return rng.randint(int(dense["min"]), int(dense["max"]))
        return rng.randint(int(objects["min"]), int(objects["max"]))
    lo, hi = objects["min"], objects["max"]
    # Inversa da CDF da distribuição arco-seno. Todos os inteiros têm
    # probabilidade positiva; mínimo e máximo recebem a mesma massa.
    u = math.sin(math.pi * rng.random() / 2) ** 2
    return lo + min(hi - lo, int((hi - lo + 1) * u))


def _mirror_rng(seed: int, subject: str, index: int = 0) -> random.Random:
    return random.Random(int(stable_hash([seed, "mirror", subject, index], 16), 16))


def _render_one(task: dict) -> dict:
    split_name = task["split"]
    index = task["index"]
    generation_index = task["generation_index"]
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
    mirror_enabled = config.get("augmentation", {}).get("horizontal_flip", False)
    background_mirrored = mirror_enabled and _mirror_rng(
        task["sample_seed"], "background"
    ).random() < 0.5
    if background_mirrored:
        canvas, depth_image = ImageOps.mirror(canvas), ImageOps.mirror(depth_image)
    grading = config["output"].get("scene_grading")
    grading_factors = {}
    if grading:
        # Aplicado só no fundo, antes de colar qualquer fruta: um realce
        # aplicado na cena inteira já composta também "esculpe" as frutas
        # já ajustadas pelo hsv_cast, empilhando saturação/nitidez até
        # ficarem artificiais (achado de revisão visual). O objetivo é só
        # aproximar o "punch" do fundo nublado do observado nas fotos reais,
        # não realçar a fruta de novo.
        grading_factors = _grading_factors(grading, task["sample_seed"])
        canvas = _apply_scene_grading(canvas, grading, grading_factors)
    support = (
        _vegetation_support(canvas)
        if config["placement"].get("require_vegetation", False)
        else None
    )
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
    requested = _sample_object_count(config["objects"], rng)
    if requested <= len(cutouts):
        chosen = rng.sample(cutouts, requested)
    else:
        chosen = rng.sample(cutouts, len(cutouts)) + [
            rng.choice(cutouts) for _ in range(requested - len(cutouts))
        ]
    instances = []
    rejected = Counter()
    scale_config = _count_adjusted_scale(config["objects"], requested)
    for object_index, cutout_path in enumerate(chosen):
        if config.get("sampling", {}).get("mode") == "paired-v1":
            rng = random.Random(int(stable_hash([task["sample_seed"], "object", object_index], 16), 16))
        fruit = _open_cutout_cached(cutout_path)
        if mirror_enabled and _mirror_rng(task["sample_seed"], "fruit", object_index).random() < 0.5:
            fruit = ImageOps.mirror(fruit)
        fruit = _scale_cutout(fruit, scale_config, rng, canvas_size)
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
        if depth_scale:
            instance = _placement_with_depth_scale(
                canvas, depth, fruit, config, rng, depth_scale, support=support
            )
        else:
            instance = _placement(canvas, depth, fruit, config, rng, support=support)
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
        "config_hash": stable_hash(config, 24),
        "generator_sha256": sha256_file(Path(__file__)),
        "id": name,
        "split": split_name,
        "generation_index": generation_index,
        "generation_id": f"scene_{generation_index:06d}",
        "seed": task["sample_seed"],
        "background": Path(pair["image"]).name,
        "background_mirrored": background_mirrored,
        "grading_factors": {k: round(v, 6) for k, v in grading_factors.items()},
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


def _render_compact(task: tuple[str, int, int, int]) -> dict:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker de síntese sem contexto")
    split_name, index, generation_index, sample_seed = task
    return _render_one(
        {
            "split": split_name,
            "index": index,
            "generation_index": generation_index,
            "sample_seed": sample_seed,
            "output": _WORKER_CONTEXT["output"],
            "config": _WORKER_CONTEXT["config"],
            "assets": _WORKER_CONTEXT["assets"],
            "force": _WORKER_CONTEXT["force"],
            "debug": _WORKER_CONTEXT["debug"],
        }
    )


def _background_sort_key(
    task: tuple[str, int, int, int], assets: dict
) -> tuple[str, str, int]:
    split_name, _index, generation_index, sample_seed = task
    rng = random.Random(sample_seed)
    pair = rng.choice(assets["backgrounds"])
    return pair["image"], split_name, generation_index


def scene_seed(config: dict, asset_fingerprint: str, generation_index: int) -> int:
    """Separa identidade da cena e parâmetros nos experimentos pareados."""
    identity = (
        "paired-v1" if config.get("sampling", {}).get("mode") == "paired-v1"
        else stable_hash(config, 24)
    )
    return int(stable_hash([int(config["seed"]), identity, asset_fingerprint,
                           "scene", generation_index], 16), 16)


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
    asset_catalog = create_asset_catalog(asset_root, force=False)
    scene_split = create_scene_split(
        int(config["images"]["total"]), train_ratio, split_seed
    )
    config_hash = stable_hash(config, 24)
    generator_sha256 = sha256_file(Path(__file__))
    config_marker = output_root / "generation_config.json"
    if config_marker.exists() and not force:
        previous = json.loads(config_marker.read_text(encoding="utf-8"))
        expected_marker = {
            "generator_sha256": generator_sha256,
            "config_hash": config_hash,
            "asset_catalog_fingerprint": asset_catalog["source_fingerprint"],
            "scene_split": {
                "version": scene_split["version"],
                "seed": scene_split["seed"],
                "train_ratio": scene_split["train_ratio"],
            },
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
            "generator_sha256": generator_sha256,
            "config_hash": config_hash,
            "config": config,
            "asset_catalog_fingerprint": asset_catalog["source_fingerprint"],
            "scene_split": {
                "version": scene_split["version"],
                "seed": scene_split["seed"],
                "train_ratio": scene_split["train_ratio"],
            },
        },
    )
    atomic_write_json(output_root / "scene_split.json", scene_split)

    source_assets = asset_catalog["assets"]
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
        raise RuntimeError("catálogo de ativos sintéticos vazio")

    tasks: list[tuple[str, int, int, int]] = []
    for split_name in ("train", "val"):
        for index, generation_index in enumerate(scene_split["splits"][split_name]):
            sample_seed = scene_seed(config, asset_catalog["source_fingerprint"], generation_index)
            tasks.append((split_name, index, generation_index, sample_seed))
    # Agrupar fundos aumenta o reaproveitamento do cache sem alterar a semente ou
    # a composição de nenhuma cena. O manifesto é ordenado novamente ao final.
    tasks.sort(key=lambda task: _background_sort_key(task, assets))
    context = {
        "output": str(output_root),
        "config": config,
        "assets": assets,
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
        "asset_catalog_fingerprint": asset_catalog["source_fingerprint"],
        "scene_split": {
            "seed": scene_split["seed"],
            "train_ratio": scene_split["train_ratio"],
            "sha256": sha256_file(output_root / "scene_split.json"),
        },
        "asset_counts": {
            "backgrounds": len(assets["backgrounds"]),
            "cutouts": len(assets["cutouts"]),
        },
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


def materialize_nested_subsets(
    pool_root: Path,
    target_root: Path,
    multipliers: list[int],
    *,
    base_size: int = 104,
    base_val_size: int = 26,
    prefix: str = "synthetic-",
    force: bool = False,
) -> dict:
    """Recorta prefixos aninhados de treino e validação do pool sintético.

    Depois do split determinístico, cada subconjunto usa prefixos crescentes das
    duas partições. Assim, `2x` contém todo o treino e toda a validação de `1x`,
    sem gerar amostras independentes ou reamostrar frutas, fundos e parâmetros."""
    images_dir = pool_root / "images" / "train"
    images = image_files(images_dir)
    if not images:
        raise FileNotFoundError(f"pool sem imagens de treino: {images_dir}")
    val_images = image_files(pool_root / "images" / "val")
    if not val_images:
        raise FileNotFoundError(
            f"pool sem imagens de validação: {pool_root / 'images' / 'val'}"
        )
    train_sizes = {
        int(multiplier): base_size * int(multiplier) for multiplier in multipliers
    }
    val_sizes = {
        int(multiplier): base_val_size * int(multiplier) for multiplier in multipliers
    }
    if any(multiplier <= 0 for multiplier in train_sizes):
        raise ValueError("multiplicadores devem ser inteiros positivos")
    if base_size <= 0 or base_val_size <= 0:
        raise ValueError("os tamanhos-base de treino e validação devem ser positivos")
    if train_sizes and max(train_sizes.values()) > len(images):
        raise ValueError(
            f"treino pedido {max(train_sizes.values())} excede o pool ({len(images)})"
        )
    if val_sizes and max(val_sizes.values()) > len(val_images):
        raise ValueError(
            f"validação pedida {max(val_sizes.values())} excede o pool ({len(val_images)})"
        )
    manifest_path = pool_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifesto do pool ausente: {manifest_path}")
    pool_records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_image = {Path(item["image"]).name: item for item in pool_records}
    pool_manifest_sha256 = sha256_file(manifest_path)
    target_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for multiplier, train_size in sorted(train_sizes.items()):
        val_size = val_sizes[multiplier]
        name = f"{prefix}{multiplier}x"
        target = target_root / name
        summary_path = target / "summary.json"
        if target.exists() and not force:
            if not summary_path.exists():
                raise FileExistsError(f"subconjunto incompleto: {target}; use --force")
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                existing.get("pool_manifest_sha256") != pool_manifest_sha256
                or int(existing.get("train_images", -1)) != train_size
                or int(existing.get("val_images", -1)) != val_size
                or int(existing.get("base_val_size", -1)) != base_val_size
                or int(existing.get("nested_val_prefix", -1)) != val_size
            ):
                raise RuntimeError(
                    f"subconjunto congelado não corresponde ao pool: {target}"
                )
            summary[name] = existing
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=target_root))
        try:
            selected = {
                "train": images[:train_size],
                "val": val_images[:val_size],
            }
            selected_records = []
            for split_name, split_images in selected.items():
                for image_path in split_images:
                    label_path = (
                        pool_root / "labels" / split_name / f"{image_path.stem}.txt"
                    )
                    if not label_path.exists():
                        raise FileNotFoundError(f"rótulo do pool ausente: {label_path}")
                    link_or_copy(
                        image_path,
                        temporary / "images" / split_name / image_path.name,
                    )
                    link_or_copy(
                        label_path,
                        temporary / "labels" / split_name / label_path.name,
                    )
                    try:
                        selected_records.append(by_image[image_path.name])
                    except KeyError as error:
                        raise ValueError(
                            f"imagem sem registro no manifesto do pool: {image_path.name}"
                        ) from error
            selected_records.sort(key=lambda item: (item["split"], item["image"]))
            atomic_write_text(
                temporary / "manifest.jsonl",
                "".join(
                    json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n"
                    for item in selected_records
                ),
            )
            data_yaml = {
                "path": str(temporary.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "poncan"},
            }
            atomic_write_text(
                temporary / "data.yaml",
                yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
            )
            subset_summary = {
                "name": name,
                "multiplier": multiplier,
                "base_size": base_size,
                "base_val_size": base_val_size,
                "train_images": train_size,
                "val_images": val_size,
                "pool": relative_or_absolute(pool_root),
                "pool_manifest_sha256": pool_manifest_sha256,
                "manifest_sha256": sha256_file(temporary / "manifest.jsonl"),
                "nested_train_prefix": train_size,
                "nested_val_prefix": val_size,
            }
            atomic_write_json(temporary / "summary.json", subset_summary)
            if target.exists():
                shutil.rmtree(target)
            temporary.replace(target)
            data_yaml["path"] = str(target.resolve())
            atomic_write_text(
                target / "data.yaml",
                yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
            )
            summary[name] = subset_summary
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return summary
