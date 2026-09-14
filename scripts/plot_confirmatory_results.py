#!/usr/bin/env python3
"""Consolida test_results_{oranges_field,manual_full_val}.json num Markdown único
com tabelas comparativas e um gráfico de tendência (volume de dados sintéticos
x mAP), para a coleta externa e a validação manual reutilizada na avaliação.

Não participa do pipeline reprodutível (`run_pipeline.sh`); é um script de
análise executado manualmente sobre artefatos já gerados.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "confirmatory"
FIGURES = ROOT / "docs" / "figures" / "results"
HEATMAPS_OUT = FIGURES / "heatmaps"

TESTS = {
    "oranges_field": "Laranja em árvore · coleta externa",
    "manual_full_val": "Validação manual · reusada na seleção de manual-full",
}

CONDITION_ORDER = [
    "manual-full",
    "controlled",
    "synthetic-1x",
    "synthetic-2x",
    "synthetic-3x",
    "synthetic-5x",
    "synthetic-10x",
]
TRAIN_IMAGES = {
    "manual-full": 104,
    "controlled": 284,
    "synthetic-1x": 104,
    "synthetic-2x": 208,
    "synthetic-3x": 312,
    "synthetic-5x": 520,
    "synthetic-10x": 1040,
}
SYNTHETIC_CONDITIONS = [c for c in CONDITION_ORDER if c.startswith("synthetic-")]
# `controlled` existe para mostrar que fruta recortada sem cena não treina
# detector: fica perto de zero e esmagaria a escala de qualquer gráfico. Ela
# permanece nas tabelas, onde o número zero é lido sem distorcer os outros.
CHART_CONDITIONS = [c for c in CONDITION_ORDER if c != "controlled"]

DETECTOR_ORDER = ["yolov8s", "rtdetr-l", "yolo26s"]
DETECTOR_COLOR = {
    "yolov8s": "#2a78d6",  # categorical slot 1 (blue)
    "rtdetr-l": "#eb6834",  # categorical slot 2 (orange)
    "yolo26s": "#1baf7a",  # categorical slot 3 (aqua)
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def detectores_presentes(*summaries: dict) -> list[str]:
    """Só as arquiteturas com resultado, na ordem canônica: a grade pode ser
    rodada uma arquitetura de cada vez e consolidada depois."""
    return [d for d in DETECTOR_ORDER if any(s.get(d) for s in summaries)]


def load_results(name: str, directory: Path) -> dict:
    path = directory / f"test_results_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def markdown_table(name: str, summary: dict) -> str:
    lines = [
        f"### {TESTS[name]}",
        "",
        "| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | Count MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for detector in detectores_presentes(summary):
        best_condition = max(
            (c for c in CONDITION_ORDER if c in summary.get(detector, {})),
            key=lambda c: summary[detector][c]["map50_95_mean"],
        )
        for condition in CONDITION_ORDER:
            row = summary.get(detector, {}).get(condition)
            if not row:
                continue
            flag = (
                " [val]"
                if (name == "manual_full_val" and condition == "manual-full")
                else ""
            )
            best = condition == best_condition
            label = f"**{condition}**{flag}" if best else f"{condition}{flag}"
            map_value = (
                f"**{row['map50_95_mean']:.3f}**"
                if best
                else f"{row['map50_95_mean']:.3f}"
            )
            lines.append(
                f"| {detector} | {label} | {row['precision_mean']:.3f} | "
                f"{row['recall_mean']:.3f} | {row['f1_mean']:.3f} | {row['map50_mean']:.3f} | "
                f"{row['map75_mean']:.3f} | {map_value} | "
                f"{row['count_mae_mean']:.1f} |"
            )
    lines.append("")
    lines.append(
        "Negrito = maior média observada; não indica significância estatística."
    )
    lines.append("")
    return "\n".join(lines)


def build_trend_chart(
    results_by_test: dict[str, dict], *, rounded: bool = False, metric: str = "map50_95"
) -> Path:
    # Fixed SVG IDs and no timestamp make the snapshot reproducible.
    plt.rcParams["svg.hashsalt"] = "synthetic-fruit-results"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.22, top=0.73, bottom=0.22, left=0.07, right=0.98)
    for ax, (test_name, summary) in zip(axes, results_by_test.items(), strict=True):
        # Cada painel tem escala própria. Um eixo comum comprime a coleta
        # externa contra o piso e esconde a única comparação que interessa
        # ali: a distância entre o treino sintético e o anotado à mão.
        valores = [
            summary[d][c][f"{metric}_mean"]
            for d in detectores_presentes(summary)
            for c in CHART_CONDITIONS
        ]
        desvios = [
            summary[d][c][f"{metric}_std"]
            for d in detectores_presentes(summary)
            for c in CHART_CONDITIONS
        ]
        folga = max(desvios) + 0.02
        ax.set_facecolor(SURFACE)
        ax.set_ylim(max(0, min(valores) - folga), min(1.0, max(valores) + folga))
        ax.set_xlim(70, 1330)  # espaço à direita para o rótulo direto
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        rotulos: list[list] = []
        for detector in detectores_presentes(summary):
            color = DETECTOR_COLOR[detector]
            xs = [TRAIN_IMAGES[c] for c in SYNTHETIC_CONDITIONS]
            ys = [summary[detector][c][f"{metric}_mean"] for c in SYNTHETIC_CONDITIONS]
            es = [summary[detector][c][f"{metric}_std"] for c in SYNTHETIC_CONDITIONS]
            ax.fill_between(
                xs,
                [y - e for y, e in zip(ys, es)],
                [y + e for y, e in zip(ys, es)],
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                xs, ys, color=color, linewidth=2, marker="o", markersize=5, zorder=3
            )
            # A referência anotada à mão vira faixa, não linha: a comparação é
            # com o intervalo entre sementes dela, não com um ponto exato.
            base = summary[detector]["manual-full"]
            alvo_m, alvo_s = base[f"{metric}_mean"], base[f"{metric}_std"]
            ax.axhspan(alvo_m - alvo_s, alvo_m + alvo_s, color=color, alpha=0.08, zorder=1)
            ax.axhline(alvo_m, color=color, linewidth=1.3, linestyle=(0, (5, 3)), alpha=0.9)
            rotulos.append([ys[-1], detector, color])
        # Dois detectores podem terminar quase no mesmo valor. Sem isto, os
        # rótulos se sobrepõem e nenhum dos dois se lê.
        baixo, alto = ax.get_ylim()
        folga_rotulo = (alto - baixo) * 0.045
        rotulos.sort()
        for anterior, atual in zip(rotulos, rotulos[1:]):
            if atual[0] - anterior[0] < folga_rotulo:
                atual[0] = anterior[0] + folga_rotulo
        for altura, detector, color in rotulos:
            ax.annotate(
                detector,
                xy=(xs[-1], altura),
                xytext=(7, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
                va="center",
                zorder=4,
            )
        ax.set_title(
            TESTS[test_name], fontsize=10, color=INK_PRIMARY, loc="left", pad=12
        )
        ax.set_xticks(
            [104, 208, 312, 520, 1040],
            ["1x\n104", "2x\n208", "3x\n312", "5x\n520", "10x\n1.040"],
        )
        ax.set_xlabel("Volume sintético / imagens de treino", fontsize=10, labelpad=10)
    labels = {
        "map50_95": "mAP@0.50:0.95",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    axes[0].set_ylabel(labels[metric], fontsize=10)
    # As linhas levam o nome do detector ao lado do último ponto, então a
    # legenda só precisa explicar o que cada traço significa.
    style_handles = [
        plt.Line2D([0], [0], color=INK_SECONDARY, linewidth=1.4, label="synthetic-Nx"),
        plt.Line2D(
            [0],
            [0],
            color=INK_SECONDARY,
            linewidth=1.4,
            linestyle=(0, (5, 3)),
            label="manual-full ± desvio entre sementes",
        ),
    ]
    fig.suptitle(
        f"{labels[metric]} por volume de dados sintéticos",
        x=0.07,
        ha="left",
        y=0.98,
        fontsize=16,
        color=INK_PRIMARY,
    )
    fig.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.92),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    note = (
        "Linha e faixa: média das duas sementes e o desvio entre elas. "
        "`controlled` fica fora — perto de zero, comprimiria a escala."
    )
    if rounded:
        note += " Fonte: médias publicadas, arredondadas a 3 casas."
    fig.text(0.07, 0.04, note, fontsize=9, color=INK_SECONDARY)
    FIGURES.mkdir(parents=True, exist_ok=True)
    suffix = "map" if metric == "map50_95" else metric
    out_path = FIGURES / f"synthetic-volume-vs-{suffix}.svg"
    fig.savefig(out_path, facecolor=SURFACE, metadata={"Date": None})
    plt.close(fig)
    out_path.write_text(
        "\n".join(line.rstrip() for line in out_path.read_text().splitlines()) + "\n"
    )
    return out_path


def build_ranking_chart(test_name: str, summary: dict) -> Path:
    """Barras ordenadas por mAP, uma linha por condição e detector.

    O gráfico de tendência responde "quanto volume sintético compensa"; este
    responde "quem ficou na frente de quem", que é a pergunta da tese. A barra
    de erro é o desvio entre as duas sementes: onde ela cobre a barra vizinha,
    as duas condições não estão separadas.
    """
    plt.rcParams["svg.hashsalt"] = "synthetic-fruit-results"
    linhas = sorted(
        (
            (summary[d][c], d, c)
            for d in detectores_presentes(summary)
            for c in CHART_CONDITIONS
        ),
        key=lambda item: item[0]["map50_95_mean"],
    )
    altura = 0.34 * len(linhas) + 1.9
    fig, ax = plt.subplots(figsize=(11, altura), facecolor=SURFACE)
    fig.subplots_adjust(left=0.30, right=0.78, top=1 - 1.15 / altura, bottom=0.9 / altura)
    ax.set_facecolor(SURFACE)
    teto = max(r["map50_95_mean"] + r["map50_95_std"] for r, _, _ in linhas)
    for i, (row, detector, condition) in enumerate(linhas):
        cor = DETECTOR_COLOR[detector]
        ax.barh(i, row["map50_95_mean"], height=0.62, color=cor, zorder=3)
        ax.errorbar(
            row["map50_95_mean"], i, xerr=row["map50_95_std"], color=INK_PRIMARY,
            elinewidth=1.1, capsize=3.5, capthick=1.1, fmt="none", zorder=4,
        )
        referencia = condition == "manual-full"
        # A arquitetura aparece escrita, não só na cor da barra: quem não
        # distingue as duas cores ainda lê a tabela inteira.
        ax.text(
            -0.02, i, f"[{detector}]", ha="right", va="center", fontsize=8.5,
            color=INK_MUTED, transform=ax.get_yaxis_transform(),
        )
        ax.text(
            -0.135, i, condition, ha="right", va="center", fontsize=9.5,
            color=INK_PRIMARY, fontweight="bold" if referencia else "normal",
            transform=ax.get_yaxis_transform(),
        )
        for coluna, valor in enumerate(
            (row["map50_95_mean"], row["precision_mean"], row["recall_mean"], row["f1_mean"])
        ):
            ax.text(
                1.035 + coluna * 0.075, i, f"{valor:.3f}",
                ha="right", va="center", fontsize=9,
                color=INK_PRIMARY if coluna == 0 else INK_SECONDARY,
                fontweight="bold" if coluna == 0 else "normal",
                transform=ax.get_yaxis_transform(),
            )
    for coluna, titulo in enumerate(("mAP", "P", "R", "F1")):
        ax.text(
            1.035 + coluna * 0.075, len(linhas) - 0.35, titulo, ha="right", va="bottom",
            fontsize=8.5, color=INK_MUTED, transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(linhas) - 0.3)
    ax.set_xlim(0, teto * 1.04)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_xlabel("mAP@0.50:0.95", fontsize=10, labelpad=8)
    fig.suptitle(TESTS[test_name], x=0.015, ha="left", y=0.985, fontsize=14, color=INK_PRIMARY)
    marcas = [
        plt.Line2D([0], [0], color=DETECTOR_COLOR[d], linewidth=6, label=d)
        for d in detectores_presentes(summary)
    ]
    fig.legend(
        handles=marcas, loc="upper left", bbox_to_anchor=(0.015, 1 - 0.55 / altura),
        ncol=3, frameon=False, fontsize=9,
    )
    fig.text(
        0.015, 0.32 / altura,
        "Barra de erro: desvio entre as duas sementes. `controlled` fica fora — "
        "perto de zero, comprimiria a escala.",
        fontsize=8.5, color=INK_SECONDARY,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    saida = FIGURES / f"ranking-{test_name.replace('_', '-')}.svg"
    fig.savefig(saida, facecolor=SURFACE, metadata={"Date": None})
    plt.close(fig)
    saida.write_text(
        "\n".join(linha.rstrip() for linha in saida.read_text().splitlines()) + "\n"
    )
    return saida


def copy_heatmaps(directory: Path) -> list[str]:
    HEATMAPS_OUT.mkdir(parents=True, exist_ok=True)
    keep = CONDITION_ORDER + ["oranges_field", "manual_full_val"]
    copied = []
    for name in keep:
        src = directory / "annotation_heatmaps" / f"{name}.png"
        if src.exists():
            dst = HEATMAPS_OUT / src.name
            shutil.copy2(src, dst)
            copied.append(src.name)
    return copied


def copy_detection_examples() -> list[str]:
    src_dir = ROOT / "docs" / "figures" / "results" / "examples"
    return sorted(p.name for p in src_dir.glob("*.jpg")) if src_dir.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Diretório com os test_results_*.json (padrão: artifacts/confirmatory)",
    )
    args = parser.parse_args()
    # Sem snapshot versionado: o gráfico sai dos JSONs por execução, que são
    # a única fonte que acompanha o gerador vigente.
    results_dir = args.results_dir or ROOT / "artifacts/confirmatory"
    results = {name: load_results(name, results_dir) for name in TESTS}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tables = "\n".join(
        markdown_table(name, summary) for name, summary in results.items()
    )
    chart_paths = [
        build_trend_chart(results, rounded=args.results_dir is None, metric=metric)
        for metric in ("map50_95", "precision", "recall", "f1")
    ] + [build_ranking_chart(name, summary) for name, summary in results.items()]
    heatmaps = copy_heatmaps(args.results_dir) if args.results_dir else []
    examples = copy_detection_examples()
    print(f"tabelas geradas para: {list(results)}")
    for chart_path in chart_paths:
        print(f"gráfico salvo em: {chart_path.relative_to(ROOT)}")
    print(f"heatmaps copiados: {heatmaps}")
    print(
        f"exemplos de detecção presentes: {examples or '(rode scripts/render_detection_examples.py antes)'}"
    )
    (ARTIFACTS / "results_tables.md").write_text(tables, encoding="utf-8")
    print(
        f"tabelas markdown intermediárias em: {(ARTIFACTS / 'results_tables.md').relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
