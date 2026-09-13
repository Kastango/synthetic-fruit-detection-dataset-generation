"""Auditoria visual rápida de um gabarito real.

Mostra uma imagem por vez com as caixas anotadas desenhadas e registra o
veredito humano por tecla. Quando recebe um checkpoint, ordena as imagens pela
discordância entre o gabarito e o detector: onde os dois discordam é onde o
erro de anotação se concentra, então a fila começa pelo que vale olhar.

O que sai daqui é um JSONL de vereditos — não corrige caixa nenhuma. Corrigir
coordenada é outro trabalho, e misturar os dois esconde qual dos dois foi feito.
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import random
import threading
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

from .common import ROOT

STATIC = Path(__file__).parent / "audit_static"

# As marcas espelham os casos que a revisão manual já produziu. Uma imagem
# pode ter mais de uma; "ok" é exclusiva e significa gabarito sem reparos.
MARKS = [
    ("ok", "gabarito coerente"),
    ("faltando", "fruta visível sem caixa"),
    ("caixa-frouxa", "caixa não acompanha a fruta"),
    ("duplicada", "mais de uma caixa no mesmo fruto"),
    ("oclusao-extrema", "anotada com quase nada visível"),
    ("nao-e-fruta", "caixa sobre galho, folha ou fundo"),
]

GT_COLOR = "#00d9ff"
PRED_COLOR = "#ffaf38"


def read_boxes(path: Path, width: int, height: int) -> np.ndarray:
    linhas = [l for l in path.read_text().splitlines() if l.strip()]
    if not linhas:
        return np.zeros((0, 4))
    v = np.array([[float(x) for x in l.split()[1:5]] for l in linhas])
    cx, cy, w, h = v[:, 0] * width, v[:, 1] * height, v[:, 2] * width, v[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _draw(image: Image.Image, boxes: np.ndarray, color: str, width: int) -> None:
    desenho = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in boxes:
        desenho.rectangle([x0, y0, x1, y1], outline=color, width=width)


def _encode(image: Image.Image, max_side: int = 1100) -> str:
    escala = max_side / max(image.size)
    if escala < 1:
        image = image.resize(
            (round(image.width * escala), round(image.height * escala)),
            Image.LANCZOS,
        )
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class Fila:
    """Os itens a auditar, já na ordem em que vale a pena olhá-los."""

    def __init__(
        self,
        images: Path,
        labels: Path,
        output: Path,
        sample: int | None,
        seed: int,
        checkpoint: str | None,
        conf: float,
        iou: float,
    ):
        self.images, self.labels, self.output = images, labels, output
        self.iou = iou
        self.conf = conf
        self.checkpoint = checkpoint
        caminhos = [
            p
            for p in sorted(images.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (labels / f"{p.stem}.txt").exists()
        ]
        if not caminhos:
            raise SystemExit(f"nenhuma imagem com rótulo em {images}")
        if sample and sample < len(caminhos):
            caminhos = random.Random(seed).sample(caminhos, sample)
            caminhos.sort()
        self.itens = caminhos
        self.discordancia: dict[str, dict] = {}
        if checkpoint:
            self._ordenar_por_discordancia()
        self.vereditos = self._carregar()

    def _ordenar_por_discordancia(self) -> None:
        from ultralytics import YOLO

        modelo = YOLO(self.checkpoint)
        pontos = {}
        for caminho in self.itens:
            with Image.open(caminho) as im:
                largura, altura = im.size
            gt = read_boxes(self.labels / f"{caminho.stem}.txt", largura, altura)
            r = modelo.predict(
                source=str(caminho),
                conf=self.conf,
                imgsz=960,
                max_det=1000,
                verbose=False,
            )[0]
            pred = r.boxes.xyxy.cpu().numpy()
            m = iou_matrix(gt, pred)
            sem_par = int((m.max(axis=1) < self.iou).sum()) if len(pred) else len(gt)
            sobrando = int((m.max(axis=0) < self.iou).sum()) if len(gt) else len(pred)
            self.discordancia[caminho.stem] = {
                "gabarito": len(gt),
                "previstas": len(pred),
                "sem_par": sem_par,
                "sobrando": sobrando,
            }
            # Normalizar pelo tamanho do gabarito evita que imagens cheias
            # dominem a fila só por terem mais caixas.
            pontos[caminho] = (sem_par + sobrando) / max(len(gt), 1)
        self.itens.sort(key=lambda p: -pontos[p])

    def _carregar(self) -> dict:
        if not self.output.exists():
            return {}
        return {
            registro["imagem"]: registro
            for registro in (
                json.loads(linha)
                for linha in self.output.read_text().splitlines()
                if linha.strip()
            )
        }

    def salvar(self, registro: dict) -> None:
        self.vereditos[registro["imagem"]] = registro
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w") as arquivo:
            for chave in sorted(self.vereditos):
                arquivo.write(json.dumps(self.vereditos[chave], ensure_ascii=False) + "\n")

    def item(self, indice: int, mostrar_previsoes: bool) -> dict:
        caminho = self.itens[indice]
        with Image.open(caminho) as im:
            imagem = im.convert("RGB")
        largura, altura = imagem.size
        gt = read_boxes(self.labels / f"{caminho.stem}.txt", largura, altura)
        espessura = max(2, round(max(imagem.size) / 500))
        if mostrar_previsoes and self.checkpoint:
            _draw(imagem, self._prever(caminho), PRED_COLOR, espessura)
        _draw(imagem, gt, GT_COLOR, espessura)
        return {
            "indice": indice,
            "total": len(self.itens),
            "imagem": caminho.stem,
            "caixas": len(gt),
            "resolucao": f"{largura}x{altura}",
            "discordancia": self.discordancia.get(caminho.stem),
            "veredicto": self.vereditos.get(caminho.stem),
            "figura": _encode(imagem),
        }

    def _prever(self, caminho: Path) -> np.ndarray:
        from ultralytics import YOLO

        if not hasattr(self, "_modelo"):
            self._modelo = YOLO(self.checkpoint)
        r = self._modelo.predict(
            source=str(caminho), conf=self.conf, imgsz=960, max_det=1000, verbose=False
        )[0]
        return r.boxes.xyxy.cpu().numpy()


def serve(
    host: str,
    port: int,
    images: Path,
    labels: Path,
    output: Path,
    sample: int | None,
    seed: int,
    checkpoint: str | None,
    conf: float,
    iou: float,
) -> None:
    fila = Fila(images, labels, output, sample, seed, checkpoint, conf, iou)
    trava = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _responder(self, corpo: bytes, tipo: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def _json(self, dados: dict, status: int = 200) -> None:
            self._responder(
                json.dumps(dados, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self):
            rota = urlparse(self.path).path
            if rota in ("/", "/index.html"):
                return self._responder(
                    (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8"
                )
            if rota in ("/app.js", "/style.css"):
                tipo = "text/javascript" if rota.endswith(".js") else "text/css"
                return self._responder(
                    (STATIC / rota.lstrip("/")).read_bytes(), f"{tipo}; charset=utf-8"
                )
            if rota == "/api/setup":
                return self._json(
                    {
                        "marcas": [{"chave": k, "rotulo": r} for k, r in MARKS],
                        "total": len(fila.itens),
                        "julgadas": len(fila.vereditos),
                        "tem_detector": bool(fila.checkpoint),
                        "saida": str(fila.output),
                    }
                )
            if rota.startswith("/api/item/"):
                try:
                    indice = int(rota.rsplit("/", 1)[1])
                except ValueError:
                    return self._json({"erro": "índice inválido"}, 400)
                if not 0 <= indice < len(fila.itens):
                    return self._json({"erro": "fora da fila"}, 404)
                mostrar = urlparse(self.path).query != "previsoes=0"
                with trava:
                    return self._json(fila.item(indice, mostrar))
            self._json({"erro": "rota desconhecida"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/veredicto":
                return self._json({"erro": "rota desconhecida"}, 404)
            tamanho = int(self.headers.get("Content-Length", 0))
            corpo = json.loads(self.rfile.read(tamanho) or b"{}")
            registro = {
                "imagem": corpo["imagem"],
                "marcas": [m for m in corpo.get("marcas", []) if m in dict(MARKS)],
                "nota": (corpo.get("nota") or "").strip(),
            }
            with trava:
                fila.salvar(registro)
                return self._json({"julgadas": len(fila.vereditos)})

    servidor = ThreadingHTTPServer((host, port), Handler)
    print(f"auditoria em http://{host}:{port}  ({len(fila.itens)} imagens)")
    print(f"vereditos em {output}")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado")


def resumo(caminho: Path) -> None:
    """Conta as marcas de um arquivo de vereditos."""
    registros = [
        json.loads(l) for l in caminho.read_text().splitlines() if l.strip()
    ]
    if not registros:
        raise SystemExit(f"{caminho} está vazio")
    print(f"{len(registros)} imagens julgadas")
    for chave, rotulo in MARKS:
        n = sum(chave in r["marcas"] for r in registros)
        print(f"  {chave:18} {n:4d}  ({100 * n / len(registros):5.1f}%)  {rotulo}")
    notas = [r for r in registros if r["nota"]]
    if notas:
        print(f"\n{len(notas)} com observação:")
        for r in notas:
            print(f"  {r['imagem']}: {r['nota']}")
