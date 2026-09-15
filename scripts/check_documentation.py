#!/usr/bin/env python3
"""Confere links, grade e procedência da documentação versionada.

Compara P, R, F1 e mAP das tabelas com os rótulos dos rankings SVG. Os JSONs
locais de avaliação permitem conferir todas as métricas com --results-dir.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fruit_pipeline.common import ROOT, load_yaml, stable_hash


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def anchors(text: str) -> set[str]:
    result = set()
    for heading in re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE):
        heading = heading.replace("`", "").replace("*", "").lower()
        result.add(re.sub(r"[^\w\- ]", "", heading).replace(" ", "-"))
    return result


def check_links(path: Path) -> int:
    text = path.read_text()
    require(not re.search(r"^(<<<<<<<|=======|>>>>>>>)", text, re.MULTILINE),
            f"{path}: marcador de conflito")
    links = re.findall(r"\]\(([^\s)]+)\)", text)
    links += re.findall(r'(?:src|href)="([^"]+)"', text)
    checked = 0
    for link in links:
        url = urlsplit(html.unescape(link))
        if url.scheme or url.netloc:
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        require(target.exists(), f"{path.relative_to(ROOT)}: link ausente {link}")
        if url.fragment and target.suffix == ".md":
            require(unquote(url.fragment) in anchors(target.read_text()),
                    f"{path.relative_to(ROOT)}: seção ausente {link}")
        checked += 1
    return checked


def result_rows(text: str) -> list[list[str]]:
    return [
        [cell.strip().replace("**", "").replace(" [val]", "")
         for cell in line.strip("|").split("|")]
        for line in text.splitlines()
        if re.match(r"\| (yolov8s|yolo26s|rtdetr-l) \|", line)
    ]


def check_results(text: str, experiment: dict, results_dir: Path | None) -> int:
    rows = result_rows(text)
    expected = {(model["name"], condition) for model in experiment["models"]
                for condition in experiment["conditions"]}
    size = len(expected)
    require(len(rows) == size * 2, "RESULTS: quantidade de linhas difere da grade")
    fields = ["precision", "recall", "f1", "map50", "map75", "map50_95", "count_mae"]
    for index, dataset in enumerate(("oranges_field", "manual_full_val")):
        group = rows[index * size:(index + 1) * size]
        require({tuple(row[:2]) for row in group} == expected,
                f"RESULTS: condições ou detectores diferentes em {dataset}")
        svg = ROOT / "docs/figures/results" / f"ranking-{dataset.replace('_', '-')}.svg"
        labels = [html.unescape(label.strip())
                  for label in re.findall(r"<!--(.*?)-->", svg.read_text(), re.DOTALL)]
        summary = None
        if results_dir:
            summary = json.loads((results_dir / f"test_results_{dataset}.json").read_text())["summary"]
        for row in group:
            model, condition = row[:2]
            # Each ranking row contains detector, condition, mAP, P, R, F1.
            position = next((i for i in range(len(labels) - 5)
                             if labels[i:i + 2] == [f"[{model}]", condition]), None)
            require(position is not None, f"{svg.name}: linha ausente {model} {condition}")
            require(labels[position + 2:position + 6] == [row[7], *row[2:5]],
                    f"{dataset}: tabela e SVG divergem em {model} {condition}")
            if summary:
                for field, value in zip(fields, row[2:], strict=True):
                    precision = 1 if field == "count_mae" else 3
                    expected_value = f"{summary[model][condition][field + '_mean']:.{precision}f}"
                    require(value == expected_value,
                            f"{dataset}: JSON diverge em {model} {condition} {field}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        help="diretório com os dois test_results_*.json da rodada")
    args = parser.parse_args()
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    links = sum(check_links(path) for path in docs)
    experiment = load_yaml(ROOT / "configs/confirmatory.yaml")
    pipeline = load_yaml(ROOT / "configs/pipeline.yaml")
    recipe = load_yaml(ROOT / "configs/synthesis/confirmatory_pool.yaml")
    studio = load_yaml(ROOT / "configs/synthesis/studio.yaml")
    results = (ROOT / "docs/RESULTS.md").read_text()
    recipe_hash = stable_hash(recipe, 24)
    require(f"config_hash {recipe_hash}" in results,
            "RESULTS: hash da receita difere da configuração atual")
    for key in recipe.keys() | studio.keys():
        if key not in {"name", "images"}:
            require(recipe.get(key) == studio.get(key), f"Studio e pool divergem: {key}")
    for path in (ROOT / "docs/figures/results/examples").rglob("provenance.json"):
        record = json.loads(path.read_text())
        require(record["selection_sha256"][:16] in results,
                f"RESULTS: seleção diferente da usada em {path.relative_to(ROOT)}")
        require({run["condition"] for run in record["runs"]} == set(experiment["conditions"]),
                f"Exemplo incompleto: {path.relative_to(ROOT)}")
    count = check_results(results, experiment, args.results_dir)
    runs = len(experiment["conditions"]) * len(experiment["models"]) * len(experiment["seeds"])
    for path in (ROOT / "README.md", ROOT / "docs/RESULTS.md"):
        require(re.search(rf"\b{runs} (treinos|execuções)\b", path.read_text()) is not None,
                f"{path.name}: número de execuções diverge da configuração")
    print(f"OK: {len(docs)} Markdown, {links} links locais, {runs} execuções, {count} linhas de métricas.")
    print(f"Receita: {recipe_hash}. Studio e exemplos correspondem ao protocolo descrito.")
    for multiple in pipeline["synthetic_subsets"]["multipliers"]:
        base = pipeline["synthetic_subsets"]
        print(f"synthetic-{multiple}x: {multiple * base['base_train_images']} treino, "
              f"{multiple * base['base_val_images']} validação")
    if args.results_dir:
        print("Métricas conferidas também com os JSONs de avaliação fornecidos.")
    else:
        print("Métricas conferidas com os SVGs publicados. Para comparação com as "
              "avaliações, use --results-dir com os JSONs da rodada.")


if __name__ == "__main__":
    main()
