"""A auditoria registra veredito humano; não altera gabarito nenhum."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from fruit_pipeline.audit import MARKS, Fila, read_boxes


def _conjunto(tmp_path, nomes):
    imagens, rotulos = tmp_path / "imagens", tmp_path / "rotulos"
    imagens.mkdir(exist_ok=True)
    rotulos.mkdir(exist_ok=True)
    for i, nome in enumerate(nomes):
        Image.new("RGB", (80, 60), (30, 90, 40)).save(imagens / f"{nome}.jpg")
        linhas = "\n".join("0 0.5 0.5 0.2 0.2" for _ in range(i + 1))
        (rotulos / f"{nome}.txt").write_text(linhas + "\n")
    return imagens, rotulos


def _fila(tmp_path, nomes, **kwargs):
    imagens, rotulos = _conjunto(tmp_path, nomes)
    padrao = dict(
        sample=None, seed=42, checkpoint=None, conf=0.25, iou=0.5
    )
    return Fila(imagens, rotulos, tmp_path / "vereditos.jsonl", **{**padrao, **kwargs})


def test_le_caixas_em_pixels(tmp_path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    caixas = read_boxes(tmp_path / "a.txt", 100, 200)
    assert np.allclose(caixas, [[25, 50, 75, 150]])


def test_item_traz_a_imagem_com_as_caixas_desenhadas(tmp_path):
    fila = _fila(tmp_path, ["a", "b"])
    item = fila.item(0, mostrar_previsoes=False)
    assert item["imagem"] == "a"
    assert item["caixas"] == 1
    assert item["total"] == 2
    assert item["figura"].startswith("data:image/jpeg;base64,")
    assert item["veredicto"] is None


def test_veredito_persiste_e_e_relido(tmp_path):
    fila = _fila(tmp_path, ["a", "b"])
    fila.salvar({"imagem": "b", "marcas": ["faltando"], "nota": "olhar de novo"})

    relida = _fila(tmp_path, ["a", "b"])
    guardado = relida.item(1, mostrar_previsoes=False)["veredicto"]
    assert guardado["marcas"] == ["faltando"]
    assert guardado["nota"] == "olhar de novo"


def test_amostra_e_reprodutivel(tmp_path):
    primeira = _fila(tmp_path, [f"img{i}" for i in range(10)], sample=4)
    assert len(primeira.itens) == 4
    segunda = _fila(tmp_path, [f"img{i}" for i in range(10)], sample=4)
    assert [p.name for p in primeira.itens] == [p.name for p in segunda.itens]


def test_marcas_sao_estaveis_e_incluem_o_caso_sem_reparo():
    chaves = [chave for chave, _ in MARKS]
    assert chaves[0] == "ok"
    assert len(set(chaves)) == len(chaves)
