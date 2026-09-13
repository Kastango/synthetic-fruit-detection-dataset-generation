#!/usr/bin/env python3
"""Seleciona o subconjunto de avaliação da coleta de laranja em árvore.

A coleta publicada tem 5.025 recortes 640x640 extraídos de 865 fotos por um
algoritmo de corte. Duas propriedades desse corte impedem usá-la inteira como
teste de detecção em copa:

1. Parte dos recortes fica tão perto que uma única fruta ocupa o quadro. A
   pergunta ali não é "onde estão as frutas nesta copa", é "isto é uma fruta".
   Recorte com qualquer caixa acima de `--teto` do lado é descartado.
2. As nove condições de tempo e hora estão muito desbalanceadas — só a de
   manhã ensolarada tem 2.768 dos 5.025 recortes. Como a diversidade de luz é
   justamente o motivo de usar esta coleta, cada condição entra com no máximo
   `--cap` recortes.

O corte por condição percorre as fotos de origem em rodadas, então o limite
não vira 250 recortes da mesma árvore. Nenhum dos dois critérios olha para a
distribuição da coleta própria: ajustar o teste ao domínio de treino tornaria
a comparação circular.

A saída é uma pasta YOLO plana, para ser entregue a
`scripts/import_external_test.py --source`, que faz a verificação e o
manifesto.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import random
import shutil


def maior_lado(texto: str) -> float:
    lados = [
        max(float(linha.split()[3]), float(linha.split()[4]))
        for linha in texto.splitlines()
        if linha.strip()
    ]
    return max(lados) if lados else 0.0


def foto_de_origem(stem: str) -> str:
    """`AC168_2024-12-18_06` veio da foto `AC168_2024-12-18`."""
    partes = stem.split("_")
    return "_".join(partes[:2]) if len(partes) > 2 else stem


def selecionar(raiz: Path, teto: float, cap: int, seed: int) -> tuple[list[Path], dict]:
    escolhidas: list[Path] = []
    relatorio = {}
    for pasta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        elegiveis: dict[str, list[Path]] = defaultdict(list)
        descartadas = 0
        for rotulo in sorted((pasta / "labels").glob("*.txt")):
            texto = rotulo.read_text()
            lado = maior_lado(texto)
            if lado == 0.0 or lado > teto:
                descartadas += 1
                continue
            elegiveis[foto_de_origem(rotulo.stem)].append(rotulo)
        if not elegiveis:
            relatorio[pasta.name] = {"selecionadas": 0, "descartadas": descartadas}
            continue
        embaralhador = random.Random(seed)
        fotos = sorted(elegiveis)
        for foto in fotos:
            embaralhador.shuffle(elegiveis[foto])
        da_condicao: list[Path] = []
        rodada = 0
        while len(da_condicao) < cap:
            nesta = [elegiveis[f][rodada] for f in fotos if len(elegiveis[f]) > rodada]
            if not nesta:
                break
            da_condicao += nesta[: cap - len(da_condicao)]
            rodada += 1
        escolhidas += da_condicao
        relatorio[pasta.name] = {
            "selecionadas": len(da_condicao),
            "descartadas_por_teto": descartadas,
            "fotos_de_origem": len({foto_de_origem(p.stem) for p in da_condicao}),
        }
    return sorted(escolhidas), relatorio


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="pasta 'Oranges in the field' já extraída")
    p.add_argument("--output", type=Path, required=True, help="pasta YOLO plana de saída")
    p.add_argument("--teto", type=float, default=0.25,
                   help="maior lado de caixa aceito, fração da imagem (padrão 0,25)")
    p.add_argument("--cap", type=int, default=250, help="recortes por condição (padrão 250)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--excluir", nargs="*", default=["IN_CLIP_INDOOR"],
                   help="condições fora do escopo (padrão: a de interior)")
    a = p.parse_args()

    raiz = a.source
    if not (raiz / "MS_CLIP_MOR_SUN").exists():
        candidata = raiz / "Oranges in the field"
        if candidata.exists():
            raiz = candidata
        else:
            raise SystemExit(f"não encontrei as pastas de condição em {a.source}")

    escolhidas, relatorio = selecionar(raiz, a.teto, a.cap, a.seed)
    escolhidas = [
        r for r in escolhidas if r.parent.parent.name not in set(a.excluir)
    ]

    imagens = a.output / "images"
    rotulos = a.output / "labels"
    for pasta in (imagens, rotulos):
        pasta.mkdir(parents=True, exist_ok=True)
    caixas = 0
    for rotulo in escolhidas:
        origem = rotulo.parent.parent / "images" / f"{rotulo.stem}.jpg"
        shutil.copy2(origem, imagens / origem.name)
        shutil.copy2(rotulo, rotulos / rotulo.name)
        caixas += sum(1 for l in rotulo.read_text().splitlines() if l.strip())

    resumo = {
        "criterio": {
            "teto_de_lado": a.teto,
            "recortes_por_condicao": a.cap,
            "seed": a.seed,
            "condicoes_excluidas": a.excluir,
        },
        "imagens": len(escolhidas),
        "caixas": caixas,
        "fotos_de_origem": len({foto_de_origem(r.stem) for r in escolhidas}),
        "por_condicao": relatorio,
    }
    (a.output / "selecao.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(escolhidas)} imagens e {caixas} caixas em {a.output}")
    print(f"critério e contagens por condição em {a.output / 'selecao.json'}")


if __name__ == "__main__":
    main()
