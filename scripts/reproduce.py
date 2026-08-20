#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fruit_pipeline.common import ROOT, automatic_workers, load_yaml, project_path


class Workflow:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workers = (
            max(1, args.workers) if args.workers is not None else automatic_workers()
        )
        self.pipeline = load_yaml(project_path(args.pipeline_config))
        self.experiment = load_yaml(project_path(args.experiment_config))
        self.external_name = args.external_name or str(
            self.experiment["protocol"]["external_test"]
        )
        self.asset_root = (
            args.asset_root.expanduser().resolve()
            if args.asset_root
            else project_path(self.pipeline["paths"]["assets"])
        )

    def command(self, script: str, *values: str) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / script), *values]

    def run(self, title: str, command: list[str]) -> None:
        print(f"\n== {title} ==\n{' '.join(command)}", flush=True)
        if not self.args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    def download(self, source: str) -> None:
        command = self.command("download_data.py", source)
        if self.args.accept_data_terms:
            command.append("--accept-data-terms")
        if self.args.force:
            command.append("--force")
        self.run(f"Obter dados ({source})", command)

    def import_real(self) -> None:
        source = self.args.real_source
        if source is None:
            manifest = (
                project_path(self.pipeline["paths"]["real_source"]) / "manifest.json"
            )
            if manifest.exists() or self.args.dry_run:
                print(f"\n== Base real ==\nreutilizando {manifest}")
                return
            if not self.args.accept_data_terms and not self.args.dry_run:
                raise SystemExit("confirme os termos dos dados com --accept-data-terms")
            download = self.command("download_real.py")
            if self.args.accept_data_terms:
                download.append("--accept-data-terms")
            if self.args.force:
                download.append("--force")
            self.run("Baixar base real anotada", download)
            configured_source = self.pipeline["real_dataset"]["source"]
            source = (
                project_path(self.pipeline["paths"]["archives"])
                / configured_source["archive_name"]
            )
        command = self.command(
            "import_real_dataset.py",
            "--source",
            str(source.expanduser().resolve()),
        )
        if self.args.force:
            command.append("--force")
        self.run("Importar e auditar base real original", command)

    def split_real(self) -> None:
        command = self.command("split_real.py")
        if self.args.force:
            command.append("--force")
        self.run("Congelar split real", command)

    def materialize_controlled(self) -> None:
        command = self.command("materialize_controlled.py")
        if self.args.force:
            command.append("--force")
        self.run("Materializar condição controlled", command)

    def preprocess(self) -> None:
        command = self.command(
            "preprocess_assets.py", "--stage", "all", "--device", self.args.device
        )
        if self.args.force:
            command.append("--force")
        self.run("Regenerar recortes e profundidade", command)

    def preprocess_controlled(self) -> None:
        for stage in ("normalize", "segment"):
            command = self.command(
                "preprocess_assets.py", "--stage", stage, "--device", self.args.device
            )
            if self.args.force:
                command.append("--force")
            self.run(f"Pré-processar condição controlled ({stage})", command)

    def split_assets(self) -> None:
        command = self.command("split_assets.py", "--asset-root", str(self.asset_root))
        if self.args.force:
            command.append("--force")
        self.run("Congelar ativos sintéticos de treino/validação", command)

    def synthesis_configs(self) -> list[Path]:
        return self.args.synthesis_config or [
            ROOT / "configs" / "synthesis" / "confirmatory_pool.yaml"
        ]

    def synthesize(self) -> None:
        for config in self.synthesis_configs():
            command = self.command(
                "generate_synthetic.py",
                "--synthesis-config",
                str(config.expanduser().resolve()),
                "--asset-root",
                str(self.asset_root),
                "--workers",
                str(self.workers),
            )
            if self.args.force:
                command.append("--force")
            self.run(f"Gerar dataset sintético ({config.stem})", command)

    def materialize_subsets(self) -> None:
        command = self.command("materialize_nx_subsets.py")
        if self.args.force:
            command.append("--force")
        self.run("Materializar subconjuntos sintéticos aninhados 1x–10x", command)

    def validate(self, stage: str = "all") -> None:
        command = self.command(
            "validate_data.py",
            "--stage",
            stage,
            "--asset-root",
            str(self.asset_root),
        )
        self.run(f"Auditar pipeline ({stage})", command)

    def train(self) -> None:
        command = self.command(
            "train_grid.py",
            "--config",
            str(project_path(self.args.experiment_config)),
            "--device",
            self.args.device,
            "--workers",
            str(self.workers),
        )
        if self.args.max_runs is not None:
            command.extend(["--max-runs", str(self.args.max_runs)])
        if self.args.dry_run:
            subprocess.run(command + ["--dry-run"], cwd=ROOT, check=True)
            return
        if self.args.force:
            command.append("--force")
        self.run("Treinar grade YOLO", command)

    def select(self) -> None:
        self.run(
            "Congelar checkpoints pela validação de origem",
            self.command(
                "select_models.py",
                "--config",
                str(project_path(self.args.experiment_config)),
            ),
        )

    def prepare_external(self) -> None:
        name = self.external_name
        artifacts = project_path(self.pipeline["paths"]["artifacts"])
        subdir = self.experiment["protocol"].get("artifact_subdir")
        selection = (
            artifacts / str(subdir) if subdir else artifacts
        ) / "model_selection.json"
        if not selection.exists() and not self.args.dry_run:
            raise FileNotFoundError(
                "teste externo permanece bloqueado até model_selection.json ser congelado"
            )
        download = self.command("download_external.py", name)
        if self.args.external_source:
            download.extend(
                ["--source", str(self.args.external_source.expanduser().resolve())]
            )
        if self.args.force:
            download.append("--force")
        self.run(f"Obter teste externo ({name})", download)
        dataset = self.pipeline["external_datasets"][name]
        archive = (
            project_path(self.pipeline["paths"]["archives"])
            / "external"
            / dataset["archive_name"]
        )
        command = self.command(
            "import_external_test.py", name, "--source", str(archive)
        )
        if self.args.force:
            command.append("--force")
        self.run(f"Importar teste externo ({name})", command)

    def test(self) -> None:
        if not self.args.unlock_test:
            raise SystemExit(
                "use --unlock-test somente depois de congelar model_selection.json"
            )
        command = self.command(
            "evaluate_test.py",
            "--config",
            str(project_path(self.args.experiment_config)),
            "--external-name",
            self.external_name,
            "--device",
            self.args.device,
            "--unlock-test",
        )
        if self.args.force:
            command.append("--force")
        self.run("Avaliação final no teste real", command)

    def report(self) -> None:
        self.run(
            "Gerar relatório Markdown",
            self.command(
                "generate_report.py",
                "--config",
                str(project_path(self.args.experiment_config)),
                "--external-name",
                self.external_name,
            ),
        )

    def prepare(self) -> None:
        self.download("prepared")
        self.download("raw")
        self.import_real()
        self.split_real()
        self.preprocess_controlled()
        self.materialize_controlled()
        self.split_assets()
        self.synthesize()
        self.materialize_subsets()
        self.validate("all")

    def all(self) -> None:
        self.prepare()
        self.train()
        self.select()
        if self.args.unlock_test:
            self.prepare_external()
            self.test()
            self.report()
        else:
            print(
                "\nTeste mantido fechado. Revise artifacts/confirmatory/model_selection.json "
                "e execute ./run_pipeline.sh all --unlock-test para autorizar a abertura.",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestrador da reprodução do experimento."
    )
    parser.add_argument(
        "stage",
        choices=(
            "help",
            "download-prepared",
            "download-raw",
            "download-real",
            "preprocess",
            "import-real",
            "split-real",
            "materialize-controlled",
            "split-assets",
            "synthesize",
            "materialize-subsets",
            "validate",
            "train",
            "select",
            "test",
            "prepare-test",
            "report",
            "prepare",
            "all",
        ),
    )
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--experiment-config", default="configs/confirmatory.yaml")
    parser.add_argument("--real-source", type=Path)
    parser.add_argument("--external-source", type=Path)
    parser.add_argument(
        "--external-name",
        help="teste externo configurado; padrão: protocol.external_test",
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--synthesis-config", action="append", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--accept-data-terms", action="store_true")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workflow = Workflow(args)
    actions = {
        "download-prepared": lambda: workflow.download("prepared"),
        "download-raw": lambda: workflow.download("raw"),
        "download-real": workflow.import_real,
        "preprocess": workflow.preprocess,
        "import-real": lambda: (workflow.import_real(), workflow.split_real()),
        "split-real": workflow.split_real,
        "materialize-controlled": workflow.materialize_controlled,
        "split-assets": workflow.split_assets,
        "synthesize": workflow.synthesize,
        "materialize-subsets": workflow.materialize_subsets,
        "validate": workflow.validate,
        "train": workflow.train,
        "select": workflow.select,
        "test": workflow.test,
        "prepare-test": workflow.prepare_external,
        "report": workflow.report,
        "prepare": workflow.prepare,
        "all": workflow.all,
    }
    if args.stage == "help":
        parser.print_help()
        return
    actions[args.stage]()


if __name__ == "__main__":
    main()
