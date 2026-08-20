from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .common import atomic_write_text
from .training import scoped_experiment_root


def _number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _duration(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_markdown_report(
    experiment: dict,
    pipeline: dict,
    *,
    output: Path | None = None,
) -> Path:
    artifacts = scoped_experiment_root(
        Path(pipeline["paths"]["artifacts"]), experiment, "artifact"
    )
    if not artifacts.is_absolute():
        from .common import project_path

        artifacts = project_path(artifacts)
    external_name = str(experiment["protocol"]["external_test"])
    selection_path = artifacts / "model_selection.json"
    results_path = artifacts / f"test_results_{external_name}.json"
    if not selection_path.exists() or not results_path.exists():
        raise FileNotFoundError(
            "relatório exige model_selection.json e os resultados externos congelados"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    by_run = {item["run_id"]: item for item in results["results"]}
    conditions = list(experiment["conditions"])
    lines = [
        "# Resultados confirmatórios",
        "",
        f"Gerado em {datetime.now(UTC).isoformat(timespec='seconds')}. ",
        f"Teste externo: `{external_name}`. Métrica primária: mAP@0.50:0.95.",
        "",
        (
            "Os deltas são sempre calculados contra `manual-full` dentro da mesma "
            "família de detector. Um valor positivo é um destaque descritivo; sozinho, "
            "não constitui teste de superioridade estatística."
        ),
        "",
        "## Resumo por família e condição",
        "",
        "| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | Δ vs manual | Count MAE | Tempo total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wins = []
    for model_name, by_condition in results["summary"].items():
        for condition in conditions:
            summary = by_condition[condition]
            selected = selection["selected"][model_name][condition]
            training_seconds = sum(
                float(run.get("training_seconds", 0.0)) for run in selected["runs"]
            )
            delta = summary.get("delta_vs_manual_full")
            display_condition = condition
            if condition.startswith("synthetic-") and delta is not None and delta > 0:
                display_condition = f"**{condition}** ★"
                wins.append((model_name, condition, float(delta)))
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        display_condition,
                        _number(summary.get("precision_mean")),
                        _number(summary.get("recall_mean")),
                        _number(summary.get("f1_mean")),
                        _number(summary.get("map50_mean")),
                        _number(summary.get("map75_mean")),
                        _number(summary.get("map50_95_mean")),
                        _number(delta),
                        _number(summary.get("count_mae_mean"), 2),
                        _duration(training_seconds),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Destaques dos conjuntos completamente sintéticos", ""])
    if wins:
        for model_name, condition, delta in sorted(wins, key=lambda item: -item[2]):
            lines.append(
                f"- `{condition}` superou descritivamente `manual-full` em "
                f"`{model_name}` por {_number(delta)} mAP@.50:.95."
            )
    else:
        lines.append(
            "Nenhuma condição completamente sintética teve média de mAP@.50:.95 "
            "maior que `manual-full` dentro da mesma família."
        )
    lines.extend(
        [
            "",
            "## Execuções individuais",
            "",
            "| Detector | Condição | Seed | Tempo | Val mAP@.50:.95 | Teste mAP@.50:.95 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_name, by_condition in selection["selected"].items():
        for condition in conditions:
            for run in by_condition[condition]["runs"]:
                evaluated = by_run[run["run_id"]]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            model_name,
                            condition,
                            str(run["seed"]),
                            _duration(float(run.get("training_seconds", 0.0))),
                            _number(run["validation"].get("map50_95")),
                            _number(evaluated["test"].get("map50_95")),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Rastreabilidade",
            "",
            f"- SHA-256 da seleção: `{results['selection_sha256']}`",
            f"- SHA-256 do manifesto externo: `{results['external_manifest_sha256']}`",
            f"- Checkpoints avaliados: {len(results['results'])}",
            (
                "- O teste foi aberto somente após a seleção: "
                f"`{str(results['test_opened_after_selection']).lower()}`"
            ),
            "",
        ]
    )
    destination = output or artifacts / f"RESULTS_{external_name}.md"
    atomic_write_text(destination, "\n".join(lines))
    return destination
