#!/usr/bin/env python3
"""Publica todas as tentativas do ciclo, sem escolher a melhor semente."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fruit_pipeline.common import ROOT, atomic_write_json, sha256_file


def main():
    sources = []
    rows = []
    for domain in ("citdet", "manual_full_val"):
        path = ROOT / f"artifacts/confirmatory/test_results_{domain}.json"
        sources.append(path)
        for row in json.loads(path.read_text())["results"]:
            if row["model_name"] == "yolov8s" and row["condition"] == "manual-full":
                rows.append(dict(row, domain=domain))
    for folder in ("similarity_yolov8", "realism_yolov8"):
        path = ROOT / "artifacts" / folder / "paired_results.json"
        if path.exists():
            sources.append(path)
            rows.extend(json.loads(path.read_text()))
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    records = []
    for condition in conditions:
        for domain in ("citdet", "manual_full_val"):
            runs = sorted(
                [
                    r
                    for r in rows
                    if r["condition"] == condition and r["domain"] == domain
                ],
                key=lambda r: r["seed"],
            )
            if [r["seed"] for r in runs] != [41, 42]:
                raise ValueError(f"Duas sementes obrigatórias: {condition}, {domain}")
            records.append(
                {
                    "condition": condition,
                    "domain": domain,
                    "map50_95_mean": float(
                        np.mean([r["test"]["map50_95"] for r in runs])
                    ),
                    "runs": [
                        {
                            "seed": r["seed"],
                            "run_id": r["run_id"],
                            "checkpoint_sha256": r["checkpoint_sha256"],
                            "metrics": r["test"],
                        }
                        for r in runs
                    ],
                }
            )
    output = ROOT / "docs/figures/results/realism"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "realism-yolov8"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    upper = min(1.0, max(0.65, max(r["test"]["map50_95"] for r in rows) + 0.05))
    for ax, domain, title in zip(
        axes, ("citdet", "manual_full_val"), ("CitDet", "Validação manual")
    ):
        for x, condition in enumerate(conditions):
            record = next(
                r
                for r in records
                if r["condition"] == condition and r["domain"] == domain
            )
            for run, marker, color in zip(
                record["runs"], ("o", "D"), ("#2874b5", "#d15d27")
            ):
                ax.scatter(
                    x,
                    run["metrics"]["map50_95"],
                    marker=marker,
                    color=color,
                    s=35,
                    zorder=3,
                )
            ax.plot(
                [x - 0.2, x + 0.2], [record["map50_95_mean"]] * 2, color="black", lw=2
            )
        real = next(
            r["map50_95_mean"]
            for r in records
            if r["condition"] == "manual-full" and r["domain"] == domain
        )
        ax.axhline(real, color="gray", linestyle="--", lw=1)
        ax.set_title(title, loc="left")
        ax.set_xticks(
            range(len(conditions)),
            [c.replace("paired_", "").replace("_", "\n") for c in conditions],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(0, upper)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mAP@0.50:0.95")
    fig.suptitle("YOLOv8s: treino real e ciclos sintéticos", x=0.07, ha="left")
    fig.text(
        0.07,
        0.02,
        "Círculo azul: semente 41. Losango laranja: 42. Traço preto: média. Linha cinza: média do treino real.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output / "map.svg", metadata={"Date": None})
    plt.close(fig)
    similarity = {}
    for condition in conditions:
        name = condition.removeprefix("paired_")
        folder = "studio" if name in ("reference", "essential") else "realism_yolov8"
        path = ROOT / "artifacts" / folder / f"{name}_similarity.json"
        if path.exists():
            sources.append(path)
            similarity[condition] = json.loads(path.read_text())
    atomic_write_json(
        ROOT / "docs/realism-results.json",
        {
            "interpretation": "Desenvolvimento exploratório; duas sementes não demonstram significância. Todos os ciclos incluídos.",
            "sources": [
                {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}
                for p in sources
            ],
            "records": records,
            "similarity": similarity,
        },
    )
    lines = [
        "| Receita | CitDet 41 | CitDet 42 | Média | Local 41 | Local 42 | Média |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        values = []
        for domain in ("citdet", "manual_full_val"):
            record = next(
                r
                for r in records
                if r["condition"] == condition and r["domain"] == domain
            )
            values.extend(
                [r["metrics"]["map50_95"] for r in record["runs"]]
                + [record["map50_95_mean"]]
            )
        lines.append(
            f"| {condition} | " + " | ".join(f"{v:.6f}" for v in values) + " |"
        )
    artifact = ROOT / "artifacts/realism_yolov8/report-table.md"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("\n".join(lines) + "\n")
    start, end = "<!-- realism-metrics:start -->", "<!-- realism-metrics:end -->"
    block = "\n".join(
        [
            start,
            "![YOLOv8s por receita e semente nos dois cenários](figures/results/realism/map.svg)",
            "",
            *lines,
            "",
            "Valores de mAP@.50:.95. Os pontos mostram as duas sementes de treino,",
            "não intervalos de confiança. Todos os ciclos são exploratórios; ambos",
            "os conjuntos reais já participaram do desenvolvimento. Os checkpoints",
            "sintéticos foram selecionados somente na validação sintética.",
            "",
            "O [snapshot completo](realism-results.json) registra métricas, similaridade",
            "e hashes das fontes. Reproduza este bloco com",
            "`.venv/bin/python scripts/report_realism.py` após a avaliação dos dois cenários.",
            end,
        ]
    )
    doc = ROOT / "docs/RESULTS.md"
    text = doc.read_text()
    if start in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = prefix + block + suffix
    else:
        text += "\n## Ciclo de verossimilhança com YOLOv8s\n\n" + block + "\n"
    doc.write_text(text)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
