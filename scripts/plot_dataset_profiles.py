#!/usr/bin/env python3
"""Exporta distribuições medidas como figura compartilhável (requer matplotlib)."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/dataset_profiles/images.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/dataset_profiles/distributions.png")
    )
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open()))
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {
        "manual-full": "#226c4f",
        "oranges_field": "#d67a16",
        "synthetic": "#596bb1",
    }
    titles = [
        "Caixas por imagem",
        "Tamanho mediano da caixa por imagem (%)",
        "Brilho da imagem (0–255)",
        "Área somada das caixas (% da imagem)",
    ]
    for ax, metric, title, factor in zip(
        axes.flat,
        ["boxes", "median_max_side", "brightness", "sum_box_area"],
        titles,
        [1, 100, 1, 100],
    ):
        for name, color in colors.items():
            vals = np.sort(
                [
                    float(r[metric]) * factor
                    for r in rows
                    if r["dataset"] == name and r[metric]
                ]
            )
            ax.step(
                vals,
                np.arange(1, len(vals) + 1) / len(vals) * 100,
                where="post",
                label=name,
                color=color,
                linewidth=2,
            )
        # The mixed CDF gives total weight 1/2 to each real domain.
        pairs = []
        for name in ("manual-full", "oranges_field"):
            vals = [
                float(r[metric]) * factor
                for r in rows
                if r["dataset"] == name and r[metric]
            ]
            pairs.extend((v, 0.5 / len(vals)) for v in vals)
        pairs.sort()
        ax.step(
            [v for v, w in pairs],
            np.cumsum([w for v, w in pairs]) * 100,
            where="post",
            label="Alvo 50/50",
            color="#292929",
            linestyle="--",
        )
        ax.set(xlabel=title, ylabel="Imagens acumuladas (%)", ylim=(0, 101))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Perfis dos dados e alvo 50/50 — pool oficial medido em 13/09/2026", fontsize=15
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
