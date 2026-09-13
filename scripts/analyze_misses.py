#!/usr/bin/env python3
"""Extrai o que o detector deixa de encontrar, para orientar o gerador.

Roda um checkpoint sobre um conjunto real, casa as predições com o gabarito
por IoU e separa as caixas que ficaram sem par. Emite recortes das falhas e
o perfil delas — tamanho, densidade local e posição —, que é o que permite
dizer ao gerador o que produzir mais.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fruit_pipeline.common import ROOT

FONT = ROOT / "docs/figures/fonts/Geist[wght].ttf"


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


def vizinhos(caixas: np.ndarray, raio: float) -> np.ndarray:
    """Quantas outras caixas de gabarito há em volta de cada uma."""
    if len(caixas) == 0:
        return np.zeros(0)
    centros = np.stack(
        [(caixas[:, 0] + caixas[:, 2]) / 2, (caixas[:, 1] + caixas[:, 3]) / 2], axis=1
    )
    dist = np.linalg.norm(centros[:, None] - centros[None, :], axis=2)
    return (dist < raio).sum(axis=1) - 1


def folha(recortes, caminho: Path, colunas=8, celula=150):
    if not recortes:
        return
    fonte = ImageFont.truetype(str(FONT), 15)
    linhas = (len(recortes) + colunas - 1) // colunas
    rotulo = 22
    sheet = Image.new(
        "RGB",
        (colunas * (celula + 8) + 8, linhas * (celula + rotulo + 8) + 8),
        "white",
    )
    d = ImageDraw.Draw(sheet)
    for i, (legenda, img) in enumerate(recortes):
        col, lin = i % colunas, i // colunas
        x, y = 8 + col * (celula + 8), 8 + lin * (celula + rotulo + 8)
        d.text((x, y), legenda, font=fonte, fill="#233A33")
        img = img.resize((celula, celula), Image.LANCZOS)
        sheet.paste(img, (x, y + rotulo))
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(caminho, quality=92)
    print(f"  {caminho}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--amostras", type=int, default=24)
    args = p.parse_args()
    from ultralytics import YOLO

    modelo = YOLO(args.checkpoint)
    perdidas, achadas, recortes = [], [], []
    for caminho in sorted(Path(args.images).glob("*.jpg")):
        rotulo = Path(args.labels) / f"{caminho.stem}.txt"
        if not rotulo.exists():
            continue
        with Image.open(caminho) as im:
            im = im.convert("RGB")
            W, H = im.size
            gt = read_boxes(rotulo, W, H)
            if len(gt) == 0:
                continue
            r = modelo.predict(
                source=str(caminho), conf=args.conf, imgsz=960, max_det=1000,
                device="0", verbose=False,
            )[0]
            pred = r.boxes.xyxy.cpu().numpy()
            melhor = iou_matrix(gt, pred).max(axis=1) if len(pred) else np.zeros(len(gt))
            dens = vizinhos(gt, raio=0.08 * max(W, H))
            lado = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
            for i in range(len(gt)):
                registro = dict(
                    lado=float(lado[i] / W),
                    densidade=int(dens[i]),
                    altura=float((gt[i, 1] + gt[i, 3]) / 2 / H),
                    iou=float(melhor[i]),
                )
                (perdidas if melhor[i] < args.iou else achadas).append(registro)
                if melhor[i] < args.iou:
                    cx, cy = (gt[i, 0] + gt[i, 2]) / 2, (gt[i, 1] + gt[i, 3]) / 2
                    meia = max(48, float(lado[i]) * 2.4)
                    crop = im.crop(
                        (
                            max(0, int(cx - meia)), max(0, int(cy - meia)),
                            min(W, int(cx + meia)), min(H, int(cy + meia)),
                        )
                    ).copy()
                    escala = 150 / max(crop.size)
                    dd = ImageDraw.Draw(crop)
                    dd.rectangle(
                        [gt[i, 0] - max(0, cx - meia), gt[i, 1] - max(0, cy - meia),
                         gt[i, 2] - max(0, cx - meia), gt[i, 3] - max(0, cy - meia)],
                        outline="#ff4d4d", width=max(1, int(2 / escala)),
                    )
                    recortes.append((float(lado[i] / W), f"{lado[i]:.0f}px", crop))
    total = len(perdidas) + len(achadas)
    print(f"\ngabarito: {total} caixas | encontradas {len(achadas)} | perdidas {len(perdidas)} ({100*len(perdidas)/total:.1f}%)")
    saida = Path(args.output)
    saida.mkdir(parents=True, exist_ok=True)
    json.dump(
        dict(total=total, perdidas=len(perdidas), detalhe=perdidas),
        (saida / "misses.json").open("w"),
    )

    def perfil(nome, chave, faixas):
        print(f"\ntaxa de perda por {nome}:")
        for lo, hi in faixas:
            sel_p = [r for r in perdidas if lo <= r[chave] < hi]
            sel_a = [r for r in achadas if lo <= r[chave] < hi]
            n = len(sel_p) + len(sel_a)
            if n < 20:
                continue
            print(f"  {lo:6.3f}–{hi:6.3f}: {100*len(sel_p)/n:5.1f}% perdidas  (n={n})")

    perfil("tamanho (lado/largura)", "lado", [(0, .015), (.015, .025), (.025, .04), (.04, .07), (.07, 1)])
    perfil("densidade local (vizinhos)", "densidade", [(0, 3), (3, 8), (8, 15), (15, 100)])
    perfil("altura na imagem", "altura", [(0, .25), (.25, .5), (.5, .75), (.75, 1)])

    recortes.sort(key=lambda r: r[0])
    n = args.amostras
    for nome, fatia in (
        ("perdidas-menores", recortes[:n]),
        ("perdidas-medianas", recortes[len(recortes) // 2 - n // 2 : len(recortes) // 2 + n // 2]),
        ("perdidas-maiores", recortes[-n:]),
    ):
        folha([(c[1], c[2]) for c in fatia], saida / f"{nome}.jpg")


if __name__ == "__main__":
    main()
