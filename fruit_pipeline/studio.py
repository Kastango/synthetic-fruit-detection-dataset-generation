"""Estúdio local: a prévia chama o mesmo compositor usado no dataset."""

from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import threading
import zipfile
import re
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import yaml
from PIL import Image, ImageDraw, ImageOps
from .common import ROOT, load_yaml, stable_hash, sha256_file, atomic_write_json
from .synthesis import (
    create_asset_catalog,
    _render_one,
    scene_seed,
    validate_synthesis_config,
    generate_dataset,
    _open_background_pair,
)
from .studio_assets import install_demo_assets
from .similarity import read_boxes, image_features, merge_features, FEATURES
import numpy as np

# Controles de pesquisa. Os detalhes mecânicos continuam explícitos no YAML
# exportado; restringir a superfície de ajuste não apaga sua existência.
SAMPLE_SCENES = 16
SAMPLE_CROPS = 48

CONTROLS = [
    (
        "fruit_min",
        "Cena",
        "Mínimo de frutas",
        0,
        150,
        1,
        10,
        "Quantidades próximas dos extremos aparecem mais que as do centro.",
    ),
    (
        "fruit_max",
        "Cena",
        "Máximo de frutas",
        0,
        150,
        1,
        100,
        "Frutas solicitadas; rejeições e oclusões podem reduzir o total visível.",
    ),
    (
        "min_scale",
        "Geometria",
        "Tamanho mínimo (%)",
        0.5,
        8,
        0.1,
        1,
        "Maior lado do recorte em relação ao menor lado da imagem, antes da profundidade.",
    ),
    (
        "max_scale",
        "Geometria",
        "Tamanho máximo (%)",
        1,
        15,
        0.1,
        6.5,
        "Não equivale diretamente ao tamanho da caixa visível.",
    ),
    (
        "z_offset",
        "Geometria",
        "Posição na profundidade",
        -30,
        30,
        1,
        -5,
        "Valores negativos colocam a fruta atrás da folhagem.",
    ),
    (
        "min_visibility",
        "Geometria",
        "Visibilidade mínima (%)",
        5,
        80,
        1,
        15,
        "Limiar na inserção; frutas posteriores ainda podem ocluir.",
    ),
    (
        "green_fraction",
        "Aparência",
        "Frutas em maturação (%)",
        0,
        50,
        1,
        18,
        "Recoloração aproximada de recortes maduros; revise os detalhes.",
    ),
    (
        "light_match",
        "Aparência",
        "Influência da luz local (%)",
        0,
        90,
        1,
        45,
        "Adoção da luminância do alvo ambiental no HSV cast.",
    ),
    (
        "exposure_spread",
        "Aparência",
        "Variação de exposição (%)",
        0,
        100,
        1,
        100,
        "Zero remove a exposição aleatória independente da cena.",
    ),
    (
        "contact_shadow",
        "Aparência",
        "Sombra de contato (%)",
        0,
        80,
        1,
        45,
        "Escurecimento próximo à borda de oclusão.",
    ),
    (
        "mirror_probability",
        "Cena",
        "Espelhamento (%)",
        0,
        100,
        1,
        50,
        "Chance de espelhar o fundo e cada fruta; cria composições novas a partir do mesmo catálogo.",
    ),
    (
        "light_angle",
        "Luz",
        "Direção da luz (graus)",
        0,
        359,
        1,
        315,
        "Ângulo do sol na cena. As sombras caem do lado oposto.",
    ),
    (
        "light_spread",
        "Luz",
        "Variação da direção (graus)",
        0,
        180,
        1,
        20,
        "Quanto a direção muda entre cenas. O ângulo é único dentro de cada cena.",
    ),
    (
        "brightness_spread",
        "Variação",
        "Variação de brilho (%)",
        0,
        40,
        1,
        0,
        "Faixa em torno do brilho do fundo, sorteada por cena.",
    ),
    (
        "contrast_spread",
        "Variação",
        "Variação de contraste (%)",
        0,
        60,
        1,
        0,
        "Faixa em torno do contraste do fundo. Valores baixos achatam a cena.",
    ),
    (
        "saturation_spread",
        "Variação",
        "Variação de saturação (%)",
        0,
        40,
        1,
        0,
        "Faixa em torno da saturação do fundo, sorteada por cena.",
    ),
    (
        "sharpness",
        "Variação",
        "Nitidez do fundo (%)",
        0,
        100,
        1,
        25,
        "Realce de bordas aplicado ao fundo antes de inserir as frutas.",
    ),
    (
        "sharpness_spread",
        "Variação",
        "Variação de nitidez (%)",
        0,
        100,
        1,
        0,
        "Faixa em torno da nitidez, sorteada por cena.",
    ),
    (
        "background_brightness",
        "Fundo",
        "Brilho do fundo (%)",
        60,
        130,
        1,
        82,
        "Ajuste somente do fundo, antes da composição.",
    ),
    (
        "background_contrast",
        "Fundo",
        "Contraste do fundo (%)",
        80,
        140,
        1,
        120,
        "Ajuste somente do fundo, antes da composição.",
    ),
]
DEFAULTS = {row[0]: row[6] for row in CONTROLS}
SIMPLE = {**DEFAULTS, "exposure_spread": 0}


def resolve_recipe(base: dict, controls: dict, preset: str, seed: int) -> dict:
    if preset not in {"reference", "essential"}:
        raise ValueError("Receita desconhecida")
    if set(controls) - set(DEFAULTS):
        raise ValueError("Controle desconhecido")
    values = {**(SIMPLE if preset == "essential" else DEFAULTS), **controls}
    for key, _, _, lo, hi, step, _, _ in CONTROLS:
        value = values[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not lo <= value <= hi
        ):
            raise ValueError(f"{key}: valor deve estar entre {lo} e {hi}")
        if step == 1 and int(value) != value:
            raise ValueError(f"{key}: use um valor inteiro")
    if values["min_scale"] > values["max_scale"]:
        raise ValueError("Tamanho mínimo não pode superar o máximo")
    if values["fruit_min"] > values["fruit_max"]:
        raise ValueError("Mínimo de frutas não pode superar o máximo")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**31 - 1
    ):
        raise ValueError("Semente deve ser um inteiro entre 0 e 2147483647")
    c = deepcopy(base)
    c.update(name="studio_candidate", seed=seed, sampling={"mode": "paired-v1"})
    c["objects"].pop("dense", None)
    c["objects"].update(min=int(values["fruit_min"]), max=int(values["fruit_max"]))
    c["augmentation"] = {"horizontal_flip": values["mirror_probability"] / 100}
    c["objects"].update(
        min_scale=values["min_scale"] / 100, max_scale=values["max_scale"] / 100
    )
    c["placement"].update(
        z_offset=values["z_offset"], min_visibility=values["min_visibility"] / 100
    )
    c["appearance"]["ripeness"]["fraction_affected"] = values["green_fraction"] / 100
    c["appearance"]["hsv_cast"]["value_power"] = values["light_match"] / 100
    spread = values["exposure_spread"] / 100
    if spread:
        c["appearance"]["exposure_jitter"]["range"] = [
            1 - 0.6 * spread,
            1 + 0.9 * spread,
        ]
    else:
        c["appearance"].pop("exposure_jitter", None)
    if preset == "essential":
        c["occlusion"].pop("cast_shadow", None)
    c["occlusion"]["contact_shadow"]["strength"] = values["contact_shadow"] / 100
    cast = c["occlusion"].get("cast_shadow")
    if cast:
        # Um ângulo por cena, e não por fruta: numa fotografia o sol está num
        # lugar só. Com isso a variação entre cenas pode ser bem maior.
        cast.update(
            light_angle_degrees=float(values["light_angle"]),
            light_angle_jitter_degrees=float(values["light_spread"]),
            light_angle_per_scene=True,
        )
    grading = c["output"]["scene_grading"]
    grading["sharpen_percent"] = int(values["sharpness"])
    # Cada faixa é simétrica em torno do valor base: o controle é a amplitude,
    # não os extremos. Amplitude zero mantém o eixo constante, como antes.
    for control, key in (
        ("brightness_spread", "brightness"),
        ("contrast_spread", "contrast"),
        ("saturation_spread", "saturation"),
        ("sharpness_spread", "sharpen_percent"),
    ):
        spread = values[control] / 100
        if spread:
            # O piso fica acima de zero: um fator nulo zeraria o eixo em vez
            # de reduzi-lo, e a validação da receita recusa faixas em zero.
            grading[f"{key}_jitter"] = [
                round(max(0.01, 1 - spread), 4),
                round(1 + spread, 4),
            ]
        else:
            grading.pop(f"{key}_jitter", None)
    c["output"]["scene_grading"].update(
        brightness=values["background_brightness"] / 100,
        contrast=values["background_contrast"] / 100,
    )
    validate_synthesis_config(c)
    return c


def picture(image: Image.Image, max_side: int = 960) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def illustration(image: Image.Image, boxes: list, name: str) -> dict:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = image.size
    for x, y, bw, bh in boxes:
        draw.rectangle(
            ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h),
            outline="#ffce54",
            width=max(1, round(w / 500)),
        )
    crops = []
    # Ordem do arquivo: nenhuma seleção pelas predições ou pela aparência.
    for x, y, bw, bh in boxes[:SAMPLE_CROPS]:
        side = max(bw * w, bh * h) * 2
        crop = image.crop(
            (
                max(0, int(x * w - side / 2)),
                max(0, int(y * h - side / 2)),
                min(w, int(x * w + side / 2)),
                min(h, int(y * h + side / 2)),
            )
        )
        crops.append(
            picture(
                ImageOps.pad(
                    crop, (160, 160), method=Image.Resampling.NEAREST, color="#d6dfda"
                )
            )
        )
    return dict(
        name=name,
        image=picture(image),
        annotated=picture(annotated),
        crops=crops,
        count=len(boxes),
        width=w,
        height=h,
    )


class Studio:
    def __init__(self, asset_root: Path, output: Path):
        self.base = load_yaml(ROOT / "configs/synthesis/studio.yaml")
        self.output = output
        catalog = create_asset_catalog(asset_root)
        self.fingerprint = catalog["source_fingerprint"]
        self.asset_hashes = catalog["sha256"]
        self.assets = {
            "backgrounds": [
                {k: str(asset_root / v) for k, v in pair.items()}
                for pair in catalog["assets"]["backgrounds"]
            ],
            "cutouts": [str(asset_root / p) for p in catalog["assets"]["cutouts"]],
        }
        self.asset_root = asset_root
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs_root = output.parent / "datasets"

    def start_job(self, body):
        config = resolve_recipe(
            self.base,
            body.get("controls", {}),
            body.get("preset", "reference"),
            body.get("seed", 42),
        )
        total = body.get("total", 390)
        ratio = body.get("train_ratio", 0.8)
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or not 2 <= total <= 5000
        ):
            raise ValueError("Use entre 2 e 5.000 imagens")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (float, int))
            or not 0.5 <= ratio <= 0.95
        ):
            raise ValueError("A fração de treino deve ficar entre 50% e 95%")
        config["images"]["total"] = total
        job_id = stable_hash(
            [
                config,
                ratio,
                self.fingerprint,
                sha256_file(Path(__file__).with_name("synthesis.py")),
            ],
            24,
        )
        with self.jobs_lock:
            if job_id in self.jobs and self.jobs[job_id]["status"] in {
                "running",
                "complete",
            }:
                return self.job_status(job_id)
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError(
                    "Um dataset está sendo gerado. Aguarde a conclusão para iniciar outro."
                )
            self.jobs[job_id] = {
                "id": job_id,
                "status": "running",
                "total": total,
                "completed": 0,
            }
        self.executor.submit(self._generate_job, job_id, config, ratio)
        return self.job_status(job_id)

    def _generate_job(self, job_id, config, ratio):
        root = self.jobs_root / job_id
        try:
            summary = generate_dataset(
                self.asset_root,
                root,
                config,
                train_ratio=ratio,
                split_seed=config["seed"],
                workers=1,
            )
            (root / "recipe.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            )
            import platform
            import PIL

            atomic_write_json(
                root / "provenance.json",
                {
                    "assets_sha256": self.asset_hashes,
                    "generator_sha256": sha256_file(
                        Path(__file__).with_name("synthesis.py")
                    ),
                    "runtime": {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "Pillow": PIL.__version__,
                    },
                    "sampling": config["sampling"],
                    "seed": config["seed"],
                },
            )
            archive = root.with_suffix(".zip")
            temporary = archive.with_suffix(".zip.part")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as z:
                for path in sorted(root.rglob("*")):
                    if path.is_file() and path.name not in {"data.yaml", "job.json"}:
                        z.write(path, path.relative_to(root).as_posix())
                # Sem caminho absoluto da máquina que gerou o dataset.
                z.writestr(
                    "data.yaml",
                    yaml.safe_dump(
                        {
                            "train": "images/train",
                            "val": "images/val",
                            "names": {0: "poncan"},
                        },
                        sort_keys=False,
                    ),
                )
            temporary.replace(archive)
            self.jobs[job_id].update(
                status="complete",
                completed=config["images"]["total"],
                summary=summary,
                download=f"/api/jobs/{job_id}/download",
            )
            atomic_write_json(root / "job.json", self.jobs[job_id])
        except Exception as error:
            import traceback

            traceback.print_exc()
            self.jobs[job_id].update(
                status="error", error=f"Não foi possível gerar o dataset: {error}"
            )

    def job_status(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{24}", job_id):
            raise ValueError("Dataset desconhecido")
        if job_id not in self.jobs:
            saved = self.jobs_root / job_id / "job.json"
            if not saved.exists():
                raise ValueError("Dataset desconhecido")
            self.jobs[job_id] = json.loads(saved.read_text())
        job = dict(self.jobs[job_id])
        if job["status"] == "running":
            job["completed"] = sum(
                1 for _ in (self.jobs_root / job_id / "metadata").rglob("*.json")
            )
        return job

    def render(self, body):
        with self.lock:
            config = resolve_recipe(
                self.base,
                body.get("controls", {}),
                body.get("preset", "reference"),
                body.get("seed", 42),
            )
            index = body.get("index", 0)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index <= 100000
            ):
                raise ValueError("Índice de cena inválido")
            code_hash = sha256_file(Path(__file__).with_name("synthesis.py"))
            key = stable_hash([config, index, self.fingerprint, code_hash], 24)
            if key not in self.cache:
                output = self.output / key
                output.mkdir(parents=True, exist_ok=True)
                views, features = [], []
                for i in range(index, index + SAMPLE_SCENES):
                    record = _render_one(
                        dict(
                            split="preview",
                            index=i,
                            generation_index=i,
                            output=str(output),
                            sample_seed=scene_seed(config, self.fingerprint, i),
                            config=config,
                            assets=self.assets,
                            force=True,
                        )
                    )
                    boxes = read_boxes(output / record["label"])
                    with Image.open(output / record["image"]) as im:
                        view = illustration(im, boxes, record["id"])
                        pair = next(
                            p
                            for p in self.assets["backgrounds"]
                            if Path(p["image"]).name == record["background"]
                        )
                        background, _ = _open_background_pair(
                            Path(pair["image"]),
                            Path(pair["depth"]),
                            tuple(config["canvas"]),
                        )
                        if record["background_mirrored"]:
                            background = ImageOps.mirror(background)
                        view["background"] = picture(background)
                        views.append(view)
                        features.append(image_features(im, boxes))
                self.cache[key] = (views, merge_features(features))
                # Arquivos da prévia são efêmeros; o YAML exportado reproduz as cenas.
                import shutil

                shutil.rmtree(output)
                while len(self.cache) > 4:
                    self.cache.popitem(last=False)
            views, features = self.cache[key]
            metrics = [
                dict(
                    key=key,
                    label=label,
                    quantiles=(
                        (np.quantile(features[key], [0.1, 0.5, 0.9]) * scale).tolist()
                        if features[key]
                        else None
                    ),
                )
                for key, (label, scale) in FEATURES.items()
            ]
            return dict(
                synthetic=views,
                metrics=metrics,
                config=config,
                config_hash=stable_hash(config, 24),
                generator_sha256=code_hash,
                asset_fingerprint=self.fingerprint,
                yaml=yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                sample_images=SAMPLE_SCENES,
                sample_boxes=sum(v["count"] for v in views),
            )


def serve(host="127.0.0.1", port=8765, asset_root=None, output=None):
    full_assets = ROOT / "data/assets/regenerated"
    asset_root = (
        Path(asset_root)
        if asset_root
        else (full_assets if full_assets.exists() else ROOT / "data/studio-demo")
    )
    output = output or ROOT / "artifacts/studio/preview"
    try:
        studio = Studio(asset_root, output)
    except FileNotFoundError:
        studio = None
    installation_lock = threading.Lock()
    static = Path(__file__).with_name("studio_static")

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path.startswith("/api/") and studio is None:
                return self.send(
                    409, b'{"error":"Prepare os dados primeiro."}', "application/json"
                )
            if path == "/api/controls":
                value = dict(
                    controls=[
                        dict(
                            zip(
                                [
                                    "key",
                                    "group",
                                    "label",
                                    "min",
                                    "max",
                                    "step",
                                    "default",
                                    "help",
                                ],
                                r,
                            )
                        )
                        for r in CONTROLS
                    ],
                    defaults=DEFAULTS,
                    essential=SIMPLE,
                )
                return self.send(200, json.dumps(value).encode(), "application/json")
            if path.startswith("/api/jobs/"):
                try:
                    parts = path.strip("/").split("/")
                    job = studio.job_status(parts[2])
                    if len(parts) == 4 and parts[3] == "download":
                        if job["status"] != "complete":
                            raise ValueError("Aguarde a geração terminar")
                        archive = (studio.jobs_root / job["id"]).with_suffix(".zip")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/zip")
                        self.send_header(
                            "Content-Disposition",
                            f'attachment; filename="pomar-{job["id"][:8]}.zip"',
                        )
                        self.send_header("Content-Length", str(archive.stat().st_size))
                        self.end_headers()
                        try:
                            with archive.open("rb") as source:
                                while chunk := source.read(1024 * 1024):
                                    self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        return
                    return self.send(200, json.dumps(job).encode(), "application/json")
                except (ValueError, IndexError, FileNotFoundError) as error:
                    return self.send(
                        404,
                        json.dumps({"error": str(error)}).encode(),
                        "application/json",
                    )
            names = {
                "/": "index.html",
                "/app.js": "app.js",
                "/style.css": "style.css",
                "/pomar-regular.woff": "pomar-regular.woff",
                "/pomar-bold.woff": "pomar-bold.woff",
            }
            if path not in names:
                return self.send(404, b"Not found", "text/plain")
            p = static / (
                "setup.html" if path == "/" and studio is None else names[path]
            )
            mime = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".woff": "font/woff",
            }[p.suffix]
            self.send(200, p.read_bytes(), mime)

        def do_POST(self):
            nonlocal studio
            if self.path not in {"/api/preview", "/api/generate", "/api/assets"}:
                return self.send(404, b"Not found", "text/plain")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                return self.send(403, b"Forbidden", "text/plain")
            try:
                if not self.headers.get("Content-Type", "").startswith(
                    "application/json"
                ):
                    raise ValueError("Use application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Requisição inválida")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Requisição deve ser um objeto")
                if self.path == "/api/assets":
                    with installation_lock:
                        if studio is None:
                            install_demo_assets(asset_root)
                            studio = Studio(asset_root, output)
                    return self.send(200, b'{"ready":true}', "application/json")
                if studio is None:
                    raise ValueError("Prepare os dados primeiro.")
                result = (
                    studio.start_job(body)
                    if self.path == "/api/generate"
                    else studio.render(body)
                )
                self.send(
                    200,
                    json.dumps(result, allow_nan=False).encode(),
                    "application/json",
                )
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send(
                    400, json.dumps({"error": str(error)}).encode(), "application/json"
                )
            except Exception:
                import traceback

                traceback.print_exc()
                self.send(
                    500,
                    b'{"error":"Falha ao gerar a cena. Consulte o terminal."}',
                    "application/json",
                )

    print(f"Estúdio disponível em http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
