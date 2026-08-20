from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .common import atomic_write_text, project_path
from .real_data import validate_yolo_text
from .training import scoped_experiment_root

_HEATMAP_GRID_SIZE = 256
_HEATMAP_OUTPUT_SIZE = 512


def _number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _duration(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _count(value: int, singular: str, plural: str) -> str:
    return f"{value} {singular if value == 1 else plural}"


def _label_directory(image_directory: Path) -> Path:
    if image_directory.parent.name != "images":
        raise ValueError(f"diretório YOLO inválido: {image_directory}")
    return image_directory.parent.parent / "labels" / image_directory.name


def _annotation_grid(
    image_directories: list[Path],
) -> tuple[np.ndarray, list[dict], list[dict]]:
    label_paths = []
    for image_directory in image_directories:
        label_directory = _label_directory(image_directory)
        if not label_directory.is_dir():
            raise FileNotFoundError(f"rótulos ausentes: {label_directory}")
        label_paths.extend(
            (image_directory.name, path)
            for path in sorted(label_directory.glob("*.txt"))
        )
    if not label_paths:
        raise FileNotFoundError(
            f"nenhum rótulo encontrado para {', '.join(map(str, image_directories))}"
        )

    difference = np.zeros(
        (_HEATMAP_GRID_SIZE + 1, _HEATMAP_GRID_SIZE + 1), dtype=np.int64
    )
    annotations = []
    split_stats = defaultdict(lambda: {"images": 0, "boxes": 0})
    for split, path in label_paths:
        split_stats[split]["images"] += 1
        text = path.read_text(encoding="utf-8")
        validate_yolo_text(text, str(path))
        for annotation_index, line in enumerate(
            line for line in text.splitlines() if line.strip()
        ):
            class_id, center_x, center_y, width, height = line.split()
            center_x, center_y, width, height = map(
                float, (center_x, center_y, width, height)
            )
            left = max(0, int(np.floor((center_x - width / 2) * _HEATMAP_GRID_SIZE)))
            top = max(0, int(np.floor((center_y - height / 2) * _HEATMAP_GRID_SIZE)))
            right = min(
                _HEATMAP_GRID_SIZE,
                max(
                    left + 1, int(np.ceil((center_x + width / 2) * _HEATMAP_GRID_SIZE))
                ),
            )
            bottom = min(
                _HEATMAP_GRID_SIZE,
                max(
                    top + 1, int(np.ceil((center_y + height / 2) * _HEATMAP_GRID_SIZE))
                ),
            )
            difference[top, left] += 1
            difference[top, right] -= 1
            difference[bottom, left] -= 1
            difference[bottom, right] += 1
            split_stats[split]["boxes"] += 1
            annotations.append(
                {
                    "split": split,
                    "image": path.stem,
                    "annotation_index": annotation_index,
                    "class_id": int(class_id),
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height,
                }
            )

    occupancy = difference[:-1, :-1].cumsum(axis=0).cumsum(axis=1)
    stats = [
        {"split": split, **values} for split, values in sorted(split_stats.items())
    ]
    return occupancy / len(label_paths), stats, annotations


def _colorize_heatmap(grid: np.ndarray, shared_maximum: float) -> Image.Image:
    if shared_maximum <= 0:
        normalized = np.zeros_like(grid, dtype=np.float64)
    else:
        normalized = np.clip(grid / shared_maximum, 0.0, 1.0)
    positions = np.array([0.0, 0.05, 0.35, 0.7, 1.0])
    colors = np.array(
        [
            [255, 255, 255],
            [210, 220, 255],
            [35, 70, 255],
            [0, 220, 100],
            [255, 235, 0],
        ],
        dtype=np.float64,
    )
    indices = np.minimum(np.searchsorted(positions, normalized, side="right") - 1, 3)
    indices = np.maximum(indices, 0)
    start = positions[indices]
    fraction = (normalized - start) / (positions[indices + 1] - start)
    rgb = colors[indices] + fraction[..., None] * (
        colors[indices + 1] - colors[indices]
    )
    return Image.fromarray(np.rint(rgb).astype(np.uint8)).resize(
        (_HEATMAP_OUTPUT_SIZE, _HEATMAP_OUTPUT_SIZE), Image.Resampling.BILINEAR
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def _annotation_datasets(
    experiment: dict, pipeline: dict, external_name: str
) -> list[tuple[str, str, list[Path]]]:
    datasets = []
    default_validation = experiment["protocol"]["validation"]
    for condition, values in experiment["conditions"].items():
        directories = [project_path(path) for path in values["train"]]
        directories.append(project_path(values.get("validation", default_validation)))
        datasets.append((condition, "treino + validação", directories))
    external_root = project_path(pipeline["paths"]["external_tests"])
    datasets.append(
        (
            external_name,
            "teste externo",
            [external_root / external_name / "images" / "test"],
        )
    )
    return datasets


def _generate_annotation_heatmaps(
    experiment: dict, pipeline: dict, external_name: str, output_directory: Path
) -> list[dict]:
    grids = []
    for name, splits, directories in _annotation_datasets(
        experiment, pipeline, external_name
    ):
        grid, split_stats, annotations = _annotation_grid(directories)
        for annotation in annotations:
            annotation["dataset"] = name
        grids.append(
            {
                "name": name,
                "splits": splits,
                "grid": grid,
                "images": sum(item["images"] for item in split_stats),
                "boxes": len(annotations),
                "split_stats": split_stats,
                "annotations": annotations,
            }
        )
    shared_maximum = max(float(item["grid"].max()) for item in grids)
    output_directory.mkdir(parents=True, exist_ok=True)
    for item in grids:
        path = output_directory / f"{_slug(item['name'])}.png"
        temporary = path.with_name(f".{path.stem}.tmp.png")
        _colorize_heatmap(item["grid"], shared_maximum).save(temporary, format="PNG")
        temporary.replace(path)
        item["path"] = path
        del item["grid"]
    return grids


def _write_csv(path: Path, rows: list[dict], preferred_fields: list[str]) -> Path:
    fields = preferred_fields + sorted(
        {key for row in rows for key in row} - set(preferred_fields)
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    return path


def _flatten(value: object, prefix: str = "") -> dict:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, child_prefix))
        return flattened
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value, ensure_ascii=False, sort_keys=True)}
    return {prefix: value}


def _experiment_root(pipeline: dict, experiment: dict, kind: str) -> Path:
    root = scoped_experiment_root(Path(pipeline["paths"][kind]), experiment, kind)
    return project_path(root)


def _training_exports(
    selection: dict, results: dict, runs_root: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    evaluations = {item["run_id"]: item for item in results["results"]}
    history_rows = []
    run_rows = []
    counting_rows = []
    for model_name, by_condition in selection["selected"].items():
        for condition, selected in by_condition.items():
            for selected_run in selected["runs"]:
                run_id = selected_run["run_id"]
                run_directory = runs_root / "training" / run_id
                result_path = run_directory / "result.json"
                history_path = run_directory / "results.csv"
                if not result_path.is_file() or not history_path.is_file():
                    raise FileNotFoundError(
                        f"histórico de treinamento incompleto para {run_id}"
                    )
                training = json.loads(result_path.read_text(encoding="utf-8"))
                evaluation = evaluations[run_id]
                metadata = {
                    "model": model_name,
                    "condition": condition,
                    "seed": int(selected_run["seed"]),
                    "run_id": run_id,
                }
                row = {
                    **metadata,
                    "candidate_id": training.get(
                        "candidate_id", selected.get("candidate_id")
                    ),
                    "dataset_fingerprint": training.get(
                        "dataset_fingerprint", selected.get("dataset_fingerprint")
                    ),
                    "training_seconds": training.get(
                        "training_seconds", selected_run.get("training_seconds")
                    ),
                    "checkpoint": training.get(
                        "checkpoint", selected_run.get("checkpoint")
                    ),
                    "checkpoint_sha256": training.get(
                        "checkpoint_sha256", selected_run.get("checkpoint_sha256")
                    ),
                    "input_checkpoint": training.get("input_checkpoint"),
                    "input_checkpoint_sha256": training.get("input_checkpoint_sha256"),
                    "ultralytics_version": training.get("ultralytics_version"),
                    "torch_version": training.get("torch_version"),
                }
                row.update(_flatten(training.get("parameters", {}), "parameter"))
                row.update(_flatten(training.get("hardware", {}), "hardware"))
                row.update(
                    _flatten(
                        training.get("validation", selected_run.get("validation", {})),
                        "validation",
                    )
                )
                row.update(_flatten(evaluation.get("test", {}), "test"))
                counting = {
                    key: value
                    for key, value in evaluation.get("counting", {}).items()
                    if key != "per_image"
                }
                row.update(_flatten(counting, "counting"))
                run_rows.append(row)

                with history_path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    rows_before = len(history_rows)
                    for source in reader:
                        history_rows.append(
                            {
                                **metadata,
                                **{
                                    str(key).strip(): str(value).strip()
                                    for key, value in source.items()
                                    if key is not None
                                },
                            }
                        )
                    if len(history_rows) == rows_before:
                        raise ValueError(f"histórico vazio: {history_path}")

                for item in evaluation.get("counting", {}).get("per_image", []):
                    counting_rows.append({**metadata, **item})
    return history_rows, run_rows, counting_rows


def _metric_column(row: dict) -> str:
    for key in row:
        normalized = re.sub(r"\s+", "", key).lower()
        if normalized.startswith("metrics/") and "map50-95" in normalized:
            return key
    raise ValueError("results.csv não contém a curva metrics/mAP50-95")


def _curve_svg(
    path: Path, title: str, series: dict[str, list[tuple[float, float]]]
) -> None:
    left, top, plot_width, plot_height = 70, 55, 720, 430
    all_epochs = [epoch for values in series.values() for epoch, _ in values]
    minimum_epoch, maximum_epoch = min(all_epochs), max(all_epochs)
    if maximum_epoch == minimum_epoch:
        maximum_epoch += 1
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
    ]

    def x_position(epoch: float) -> float:
        return (
            left
            + (epoch - minimum_epoch) / (maximum_epoch - minimum_epoch) * plot_width
        )

    def y_position(value: float) -> float:
        return top + (1 - min(max(value, 0.0), 1.0)) * plot_height

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 560">',
        '<rect width="1080" height="560" fill="white"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="20" font-weight="bold">{html.escape(title)}</text>',
    ]
    for index in range(6):
        value = index / 5
        y = y_position(value)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#dddddd"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>',
            ]
        )
    for index in range(6):
        epoch = minimum_epoch + index * (maximum_epoch - minimum_epoch) / 5
        x = x_position(epoch)
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#eeeeee"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{epoch:.0f}</text>',
            ]
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222222"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222222"/>',
            f'<text x="{left + plot_width / 2:.1f}" y="535" text-anchor="middle" font-family="sans-serif" font-size="14">Época</text>',
            f'<text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2:.1f})" font-family="sans-serif" font-size="14">mAP@0.50:0.95</text>',
        ]
    )
    for index, (name, values) in enumerate(series.items()):
        color = palette[index % len(palette)]
        points = " ".join(
            f"{x_position(epoch):.1f},{y_position(value):.1f}"
            for epoch, value in values
        )
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        legend_y = 78 + index * 34
        svg.extend(
            [
                f'<line x1="825" y1="{legend_y}" x2="855" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="865" y="{legend_y + 5}" font-family="sans-serif" font-size="14">{html.escape(name)}</text>',
            ]
        )
    svg.append("</svg>")
    atomic_write_text(path, "\n".join(svg))


def _generate_training_curves(
    history_rows: list[dict], experiment: dict, output_directory: Path
) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in history_rows:
        try:
            epoch = float(row["epoch"])
            value = float(row[_metric_column(row)])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"linha inválida em training_history.csv: {row}"
            ) from error
        grouped[row["model"]][row["condition"]][epoch].append(value)
    output_directory.mkdir(parents=True, exist_ok=True)
    curves = []
    for model_name, by_condition in grouped.items():
        series = {}
        for condition in experiment["conditions"]:
            epochs = by_condition.get(condition)
            if not epochs:
                raise ValueError(f"curva ausente para {model_name}/{condition}")
            series[condition] = [
                (epoch, float(np.mean(values)))
                for epoch, values in sorted(epochs.items())
            ]
        path = output_directory / f"validation_map50_95_{_slug(model_name)}.svg"
        _curve_svg(path, f"Validação por época — {model_name}", series)
        curves.append({"model": model_name, "path": path})
    return curves


def _analysis_bundle(
    experiment: dict,
    pipeline: dict,
    selection: dict,
    results: dict,
    heatmaps: list[dict],
    destination: Path,
) -> tuple[list[dict], Path]:
    runs_root = _experiment_root(pipeline, experiment, "runs")
    history_rows, run_rows, counting_rows = _training_exports(
        selection, results, runs_root
    )
    output_directory = destination.parent / "analysis_csv"
    output_directory.mkdir(parents=True, exist_ok=True)
    files = [
        _write_csv(
            output_directory / "training_history.csv",
            history_rows,
            ["model", "condition", "seed", "run_id", "epoch"],
        ),
        _write_csv(
            output_directory / "run_metrics.csv",
            run_rows,
            ["model", "condition", "seed", "run_id", "candidate_id"],
        ),
        _write_csv(
            output_directory / "counting_by_image.csv",
            counting_rows,
            ["model", "condition", "seed", "run_id", "id"],
        ),
    ]
    summary_rows = []
    for model_name, by_condition in results["summary"].items():
        for condition, summary in by_condition.items():
            summary_rows.append(
                {"model": model_name, "condition": condition, **_flatten(summary)}
            )
    files.append(
        _write_csv(
            output_directory / "result_summary.csv",
            summary_rows,
            ["model", "condition"],
        )
    )
    dataset_rows = []
    annotation_rows = []
    for heatmap in heatmaps:
        for stats in heatmap["split_stats"]:
            dataset_rows.append(
                {
                    "dataset": heatmap["name"],
                    "role": heatmap["splits"],
                    **stats,
                }
            )
        annotation_rows.extend(heatmap["annotations"])
    files.extend(
        [
            _write_csv(
                output_directory / "datasets.csv",
                dataset_rows,
                ["dataset", "role", "split", "images", "boxes"],
            ),
            _write_csv(
                output_directory / "annotations.csv",
                annotation_rows,
                [
                    "dataset",
                    "split",
                    "image",
                    "annotation_index",
                    "class_id",
                    "center_x",
                    "center_y",
                    "width",
                    "height",
                ],
            ),
        ]
    )
    protocol_rows = [
        {"key": key, "value": value}
        for key, value in sorted(
            _flatten({"experiment": experiment, "pipeline": pipeline}).items()
        )
    ]
    provenance_rows = [
        {"key": "analysis_schema_version", "value": 1},
        {"key": "selection_sha256", "value": results["selection_sha256"]},
        {
            "key": "external_manifest_sha256",
            "value": results["external_manifest_sha256"],
        },
        {"key": "external_dataset", "value": results["external_dataset"]},
    ]
    files.extend(
        [
            _write_csv(
                output_directory / "protocol.csv",
                protocol_rows,
                ["key", "value"],
            ),
            _write_csv(
                output_directory / "provenance.csv",
                provenance_rows,
                ["key", "value"],
            ),
        ]
    )
    curves = _generate_training_curves(
        history_rows, experiment, destination.parent / "training_curves"
    )
    archive = destination.parent / "analysis_csv.zip"
    temporary = archive.with_name(f".{archive.stem}.tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=f"analysis_csv/{path.name}")
    temporary.replace(archive)
    return curves, archive


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
    destination = output or artifacts / f"RESULTS_{external_name}.md"
    heatmaps = _generate_annotation_heatmaps(
        experiment,
        pipeline,
        external_name,
        destination.parent / "annotation_heatmaps",
    )
    curves, analysis_archive = _analysis_bundle(
        experiment, pipeline, selection, results, heatmaps, destination
    )
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
        "## Dados para análise",
        "",
        (
            "Os dados por época, execução, imagem, conjunto e anotação estão em "
            "CSVs independentes. O pacote também preserva parâmetros, hashes e "
            "métricas detalhadas de validação e teste."
        ),
        "",
        f"[Baixar pacote de análise]({Path(os.path.relpath(analysis_archive, destination.parent)).as_posix()})",
        "",
        "## Distribuição espacial das anotações",
        "",
        (
            "Os mapas acumulam a área das caixas em coordenadas normalizadas. "
            "A intensidade representa a média por imagem e a escala de cores é "
            "compartilhada entre todos os conjuntos: branco indica ausência e "
            "azul a amarelo indica densidade crescente."
        ),
        "",
    ]
    for item in heatmaps:
        relative_path = Path(
            os.path.relpath(item["path"], start=destination.parent)
        ).as_posix()
        lines.extend(
            [
                f"### `{item['name']}`",
                "",
                (
                    f"{_count(item['images'], 'imagem', 'imagens')} e "
                    f"{_count(item['boxes'], 'caixa', 'caixas')} "
                    f"({item['splits']})."
                ),
                "",
                f"![Mapa de calor das anotações de {item['name']}]({relative_path})",
                "",
            ]
        )
    lines.extend(
        [
            "## Curvas de treinamento",
            "",
            (
                "As curvas mostram a média das sementes em cada época disponível. "
                "Épocas ausentes após parada antecipada não são imputadas. Cada "
                "condição usa sua validação de origem, portanto estas curvas são "
                "diagnósticas; a comparação final permanece no teste externo."
            ),
            "",
        ]
    )
    for curve in curves:
        relative_path = Path(
            os.path.relpath(curve["path"], start=destination.parent)
        ).as_posix()
        lines.extend(
            [
                f"### `{curve['model']}`",
                "",
                f"![Curva de validação por época de {curve['model']}]({relative_path})",
                "",
            ]
        )
    lines.extend(
        [
            "## Resumo por família e condição",
            "",
            "| Detector | Condição | P | R | F1 | mAP@.50 | mAP@.75 | mAP@.50:.95 | Δ vs manual | Count MAE | Tempo total |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
    atomic_write_text(destination, "\n".join(lines))
    return destination
