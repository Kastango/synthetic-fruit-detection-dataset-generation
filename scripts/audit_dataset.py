#!/usr/bin/env python3
"""Auditoria visual de um gabarito real, uma imagem por vez.

Com --checkpoint, a fila começa pelas imagens em que o detector e o gabarito
mais discordam, que é onde o erro de anotação se concentra. Com --resumo, só
conta as marcas de um arquivo de vereditos já preenchido.
"""

import argparse
from pathlib import Path

from fruit_pipeline.audit import resumo, serve

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, help="pasta com as imagens")
    p.add_argument("--labels", type=Path, help="pasta com os .txt no formato YOLO")
    p.add_argument("--output", type=Path, required=True, help="JSONL de vereditos")
    p.add_argument("--sample", type=int, help="auditar só uma amostra de N imagens")
    p.add_argument("--seed", type=int, default=42, help="semente da amostra")
    p.add_argument("--checkpoint", help="best.pt para ordenar a fila por discordância")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--resumo", action="store_true", help="só contar as marcas e sair")
    a = p.parse_args()

    if a.resumo:
        resumo(a.output)
    elif not a.images or not a.labels:
        p.error("--images e --labels são obrigatórios fora do --resumo")
    else:
        serve(
            a.host, a.port, a.images, a.labels, a.output,
            a.sample, a.seed, a.checkpoint, a.conf, a.iou,
        )
