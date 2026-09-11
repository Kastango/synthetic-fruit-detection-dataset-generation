#!/usr/bin/env python3
"""Consolida test_results_{citdet,manual_full_val}.json num Markdown único
com tabelas comparativas e um gráfico de tendência (volume de dados sintéticos
x mAP), para CitDet e a validação manual reutilizada na avaliação.

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
    "citdet": "CitDet · coleta externa usada na calibração",
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
    for detector in DETECTOR_ORDER:
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.18, top=0.73, bottom=0.22, left=0.07, right=0.98)
    maximum = max(
        row[f"{metric}_mean"]
        for summary in results_by_test.values()
        for conditions in summary.values()
        for row in conditions.values()
    )
    upper = min(1.0, max(0.6, maximum + 0.05))
    for ax, (test_name, summary) in zip(axes, results_by_test.items(), strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_ylim(0, upper)
        ax.set_xlim(70, 1080)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        for detector in DETECTOR_ORDER:
            color = DETECTOR_COLOR[detector]
            xs = [TRAIN_IMAGES[c] for c in SYNTHETIC_CONDITIONS]
            ys = [summary[detector][c][f"{metric}_mean"] for c in SYNTHETIC_CONDITIONS]
            ax.plot(
                xs, ys, color=color, linewidth=2, marker="o", markersize=5, zorder=3
            )
            ax.axhline(
                summary[detector]["manual-full"][f"{metric}_mean"],
                color=color,
                linewidth=1.3,
                linestyle=(0, (5, 3)),
                alpha=0.8,
            )
            ax.axhline(
                summary[detector]["controlled"][f"{metric}_mean"],
                color=color,
                linewidth=1.4,
                linestyle=(0, (1, 3)),
                alpha=0.9,
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
    model_handles = [
        plt.Line2D([0], [0], color=DETECTOR_COLOR[d], linewidth=2, marker="o", label=d)
        for d in DETECTOR_ORDER
    ]
    style_handles = [
        plt.Line2D([0], [0], color=INK_SECONDARY, linewidth=1.4, label="synthetic-Nx"),
        plt.Line2D(
            [0],
            [0],
            color=INK_SECONDARY,
            linewidth=1.4,
            linestyle=(0, (5, 3)),
            label="manual-full",
        ),
        plt.Line2D(
            [0],
            [0],
            color=INK_SECONDARY,
            linewidth=1.4,
            linestyle=(0, (1, 3)),
            label="controlled",
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
        handles=model_handles + style_handles,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.92),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    note = "Médias de duas sementes; sem intervalos de incerteza."
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


def copy_heatmaps(directory: Path) -> list[str]:
    HEATMAPS_OUT.mkdir(parents=True, exist_ok=True)
    keep = CONDITION_ORDER + ["citdet", "manual_full_val"]
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
    ]
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
