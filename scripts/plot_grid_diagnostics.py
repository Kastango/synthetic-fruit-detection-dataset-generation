#!/usr/bin/env python3
"""Três leituras da grade que as tabelas de resumo não mostram.

Contagem prevista contra contagem real, AP por limiar de IoU e progressão da
validação por época. Tudo sai dos CSV em `artifacts/confirmatory/analysis_csv/`,
sem inferência nova.

Não participa do pipeline reprodutível (`run_pipeline.sh`); é um script de
análise executado manualmente sobre artefatos já gerados.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "artifacts" / "confirmatory" / "analysis_csv"
FIGURES = ROOT / "docs" / "figures" / "results"

DETECTOR_ORDER = ["yolov8s", "rtdetr-l", "yolo26s"]
SYNTHETIC_ORDER = [
    "synthetic-1x",
    "synthetic-2x",
    "synthetic-3x",
    "synthetic-5x",
    "synthetic-10x",
]
# O volume sintético é uma grandeza ordenada, não um conjunto de categorias:
# uma rampa de um matiz só, do claro ao escuro, diz "mais dados" sem pedir
# que o leitor decore cinco cores. L do OKLab: 0,769 / 0,668 / 0,575 / 0,479
# / 0,363, monótona, passo mínimo 0,061.
SYNTHETIC_RAMP = ["#8fb8e8", "#619ade", "#2a78d6", "#1f5fae", "#143f74"]
# A referência anotada à mão não pertence à rampa: ela é outra origem de dado,
# então ganha o segundo slot categórico validado, em traço interrompido.
MANUAL_COLOR = "#eb6834"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

IOU_STEPS = ["0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"]


def read_csv(name: str) -> list[dict]:
    with (CSV_DIR / f"{name}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def condition_color(condition: str) -> str:
    if condition == "manual-full":
        return MANUAL_COLOR
    return SYNTHETIC_RAMP[SYNTHETIC_ORDER.index(condition)]


def save(fig, nome: str) -> Path:
    destino = FIGURES / nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    # Sem Date o SVG sai idêntico a cada geração, então o diff do git só mexe
    # quando o dado mexe.
    fig.savefig(
        destino, format="svg", facecolor=SURFACE, dpi=200, metadata={"Date": None}
    )
    plt.close(fig)
    return destino


def build_counting_chart(detector: str = "yolo26s") -> Path:
    """Contagem prevista contra contagem real, uma condição por painel.

    A pergunta é se o erro de contagem é proporcional. Se for, um fator de
    correção único funciona; se a razão andar com a densidade, não funciona.
    """
    plt.rcParams["svg.hashsalt"] = "synthetic-fruit-diagnostics"
    dados = collections.defaultdict(list)
    for linha in read_csv("counting_by_image"):
        if linha["model"] != detector:
            continue
        dados[linha["condition"]].append(
            (float(linha["target"]), float(linha["predicted"]))
        )

    condicoes = ["manual-full", *SYNTHETIC_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0), facecolor=SURFACE)
    fig.subplots_adjust(
        wspace=0.20, hspace=0.34, top=0.83, bottom=0.10, left=0.07, right=0.98
    )
    teto = max(t for pares in dados.values() for t, _ in pares)
    for ax, condicao in zip(axes.ravel(), condicoes, strict=True):
        style(ax)
        alvo, previsto = (np.array(v) for v in zip(*dados[condicao], strict=True))
        cor = condition_color(condicao)
        # Reta pela origem: é a forma que um fator de correção teria.
        inclinacao = float((alvo * previsto).sum() / (alvo * alvo).sum())
        residuo = ((previsto - inclinacao * alvo) ** 2).sum()
        variacao = ((previsto - previsto.mean()) ** 2).sum()
        r2 = 1 - residuo / variacao
        ax.plot(
            [0, teto], [0, teto], color=BASELINE, linewidth=1.2, zorder=1,
            linestyle=(0, (4, 3)),
        )
        # 2.486 pontos por painel viram 2.486 elementos no SVG e o arquivo passa
        # de 1,7 MB. Rasterizar só a camada de pontos deixa eixo, reta e texto
        # em vetor.
        ax.scatter(
            alvo, previsto, s=7, color=cor, alpha=0.20, linewidths=0, zorder=2,
            rasterized=True,
        )
        ax.plot(
            [0, teto], [0, inclinacao * teto], color=cor, linewidth=2, zorder=3
        )
        ax.set_xlim(0, teto * 1.02)
        ax.set_ylim(0, teto * 1.02)
        ax.set_aspect("equal")
        ax.set_title(condicao, fontsize=10, color=INK_PRIMARY, loc="left", pad=8)
        ax.annotate(
            f"fator {inclinacao:.2f}\nR² {r2:.2f}".replace(".", ","),
            xy=(0.97, 0.06),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )
    for ax in axes[1]:
        ax.set_xlabel("Frutas no gabarito", fontsize=9, color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Frutas detectadas", fontsize=9, color=INK_SECONDARY)

    fig.suptitle(
        "Toda condição conta para baixo, e o fator depende do conjunto de treino",
        x=0.07,
        ha="left",
        y=0.965,
        fontsize=16,
        color=INK_PRIMARY,
    )
    fig.text(
        0.07,
        0.895,
        f"{detector}, coleta externa, 1.243 imagens por semente, as duas sementes juntas. "
        "Cada ponto é uma foto. O fator ajustado vai de 0,23 a 0,59 conforme a condição.",
        fontsize=10,
        color=INK_SECONDARY,
    )
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=BASELINE, linewidth=1.2,
                       linestyle=(0, (4, 3)), label="contagem exata"),
            plt.Line2D([0], [0], color=INK_SECONDARY, linewidth=2,
                       label="fator ajustado pela origem"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        ncol=2,
    )
    fig.text(
        0.07,
        0.028,
        "O limiar de confiança é o de melhor F1 de cada execução, entre 0,24 e 0,62, "
        "e não um valor comum. Comparar o fator entre condições depende dessa ressalva.",
        fontsize=8.5,
        color=INK_MUTED,
    )
    return save(fig, "contagem-prevista-x-real.svg")


def build_iou_chart() -> Path:
    """AP por limiar de IoU, um detector por painel.

    O mAP@0.50:0.95 é uma média sobre dez limiares. Separá-los mostra se a
    vantagem vem de achar mais fruta ou de encaixar melhor a caixa.
    """
    plt.rcParams["svg.hashsalt"] = "synthetic-fruit-diagnostics"
    dados = collections.defaultdict(list)
    for linha in read_csv("run_metrics"):
        dados[(linha["model"], linha["condition"])].append(
            [float(linha[f"test.ap_by_iou.{i}"]) for i in IOU_STEPS]
        )

    xs = [float(i) for i in IOU_STEPS]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), facecolor=SURFACE, sharey=True)
    fig.subplots_adjust(wspace=0.10, top=0.72, bottom=0.22, left=0.06, right=0.99)
    for ax, detector in zip(axes, DETECTOR_ORDER, strict=True):
        style(ax)
        for condicao in ["manual-full", *SYNTHETIC_ORDER]:
            ys = np.mean(dados[(detector, condicao)], axis=0)
            manual = condicao == "manual-full"
            ax.plot(
                xs,
                ys,
                color=condition_color(condicao),
                linewidth=2.2 if manual else 2,
                linestyle=(0, (5, 3)) if manual else "-",
                marker="o",
                markersize=4,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                zorder=4 if manual else 3,
            )
        ax.set_xlim(0.48, 0.97)
        ax.set_ylim(0, None)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_xlabel("Limiar de IoU", fontsize=9, color=INK_SECONDARY)
        ax.set_title(detector, fontsize=10, color=INK_PRIMARY, loc="left", pad=8)
    axes[0].set_ylabel("AP", fontsize=10, color=INK_SECONDARY)

    fig.suptitle(
        "A vantagem sintética cresce conforme a caixa precisa encaixar melhor",
        x=0.06,
        ha="left",
        y=0.965,
        fontsize=16,
        color=INK_PRIMARY,
    )
    fig.text(
        0.06,
        0.865,
        "Coleta externa, média das duas sementes. Em IoU 0,50 a melhor sintética vale de 0,99 a "
        "1,16 vez o manual-full. Em IoU 0,75, de 1,34 a 1,64.",
        fontsize=10,
        color=INK_SECONDARY,
    )
    handles = [
        plt.Line2D([0], [0], color=MANUAL_COLOR, linewidth=2.2,
                   linestyle=(0, (5, 3)), label="manual-full")
    ] + [
        plt.Line2D([0], [0], color=condition_color(c), linewidth=2, label=c)
        for c in SYNTHETIC_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.0),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        ncol=6,
    )
    return save(fig, "ap-por-iou.svg")


def build_convergence_chart() -> Path:
    """Validação por época, cada execução relativa ao próprio pico.

    Cada condição valida no seu próprio conjunto, com contagem de imagens
    diferente, então os valores absolutos não se comparam entre painéis nem
    entre curvas. Relativo ao pico de cada execução, sobra uma pergunta que
    a curva responde: quando o treino para de subir?
    """
    plt.rcParams["svg.hashsalt"] = "synthetic-fruit-diagnostics"
    historico = collections.defaultdict(dict)
    for linha in read_csv("training_history"):
        chave = (linha["model"], linha["condition"], linha["seed"])
        historico[chave][int(linha["epoch"])] = float(linha["metrics/mAP50(B)"] or 0.0)

    # `controlled` valida numa tarefa trivial, chega a mAP50 0,995 e satura na
    # época 16. Somá-lo às estatísticas de convergência mistura duas coisas.
    reais = {k: v for k, v in historico.items() if k[1] != "controlled"}
    piso = min(v[40] / max(v.values()) for v in reais.values())

    def epoca_de_95(serie: dict[int, float]) -> int:
        alvo = 0.95 * max(serie.values())
        return next(e for e in sorted(serie) if serie[e] >= alvo)

    lento = int(np.median([epoca_de_95(v) for k, v in reais.items()
                           if k[1] == "manual-full"]))
    rapido = int(np.median([epoca_de_95(v) for k, v in reais.items()
                            if k[1] == "synthetic-10x"]))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), facecolor=SURFACE, sharey=True)
    fig.subplots_adjust(wspace=0.10, top=0.72, bottom=0.22, left=0.06, right=0.99)
    for ax, detector in zip(axes, DETECTOR_ORDER, strict=True):
        style(ax)
        for condicao in ["manual-full", *SYNTHETIC_ORDER]:
            curvas = []
            for semente in ("41", "42"):
                serie = historico[(detector, condicao, semente)]
                epocas = sorted(serie)
                pico = max(serie.values())
                curvas.append([serie[e] / pico for e in epocas])
            ys = np.mean(curvas, axis=0)
            manual = condicao == "manual-full"
            ax.plot(
                range(1, len(ys) + 1),
                ys,
                color=condition_color(condicao),
                linewidth=2.2 if manual else 1.8,
                linestyle=(0, (5, 3)) if manual else "-",
                zorder=4 if manual else 3,
            )
        # A linha de 95% do pico transforma a leitura "quando estabiliza" num
        # cruzamento, em vez de pedir que o olho estime o joelho da curva.
        ax.axhline(0.95, color=BASELINE, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_xlim(1, 50)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("Época", fontsize=9, color=INK_SECONDARY)
        ax.set_title(detector, fontsize=10, color=INK_PRIMARY, loc="left", pad=8)
    axes[0].set_ylabel(
        "mAP@0.50 da validação, relativo ao pico\nda própria execução",
        fontsize=9,
        color=INK_SECONDARY,
    )
    axes[0].annotate(
        "95% do pico",
        xy=(49, 0.938),
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK_MUTED,
    )

    fig.suptitle(
        "Mais volume sintético chega antes ao platô, e as 50 épocas bastaram",
        x=0.06,
        ha="left",
        y=0.965,
        fontsize=16,
        color=INK_PRIMARY,
    )
    fig.text(
        0.06,
        0.865,
        f"Mediana da época em que a execução cruza 95% do próprio pico: {lento} no manual-full, "
        f"{rapido} no synthetic-10x. Na época 40 nenhuma está abaixo de {piso:.0%} do seu pico.",
        fontsize=10,
        color=INK_SECONDARY,
    )
    handles = [
        plt.Line2D([0], [0], color=MANUAL_COLOR, linewidth=2.2,
                   linestyle=(0, (5, 3)), label="manual-full")
    ] + [
        plt.Line2D([0], [0], color=condition_color(c), linewidth=1.8, label=c)
        for c in SYNTHETIC_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.0),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        ncol=6,
    )
    return save(fig, "convergencia-por-epoca.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        default="yolo26s",
        choices=DETECTOR_ORDER,
        help="detector usado no gráfico de contagem",
    )
    args = parser.parse_args()
    for caminho in (
        build_counting_chart(args.detector),
        build_iou_chart(),
        build_convergence_chart(),
    ):
        print(caminho.relative_to(ROOT))


if __name__ == "__main__":
    main()
