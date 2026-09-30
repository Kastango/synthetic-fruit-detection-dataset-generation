#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fruit_pipeline.common import ROOT, automatic_workers, load_yaml, project_path
from fruit_pipeline.progress import run_stage
from fruit_pipeline.training import scoped_experiment_root


class Workflow:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workers = (
            max(1, args.workers) if args.workers is not None else automatic_workers()
        )
        self.pipeline = load_yaml(project_path(args.pipeline_config))
        self.experiment = load_yaml(project_path(args.experiment_config))
        self.state_path = (
            project_path(self.pipeline["paths"]["artifacts"]) / "pipeline_state.json"
        )
        self.external_name = args.external_name or str(
            self.experiment["protocol"]["external_test"]
        )
        self.asset_root = (
            args.asset_root.expanduser().resolve()
            if args.asset_root
            else project_path(self.pipeline["paths"]["assets"])
        )

    def command(self, script: str, *values: str, force: bool = False) -> list[str]:
        """Monta a chamada; `force=True` repassa `--force` quando ele foi pedido."""
        command = [sys.executable, str(ROOT / "scripts" / script), *values]
        if force and self.args.force:
            command.append("--force")
        return command

    def run(self, title: str, command: list[str]) -> None:
        print(f"\n== {title} ==\n{' '.join(command)}", flush=True)
        if not self.args.dry_run:
            run_stage(title, command, cwd=ROOT, state_path=self.state_path)

    def download(self, source: str) -> None:
        terms = ["--accept-data-terms"] if self.args.accept_data_terms else []
        command = self.command("download_data.py", source, *terms, force=True)
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
            terms = ["--accept-data-terms"] if self.args.accept_data_terms else []
            download = self.command("download_real.py", *terms, force=True)
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
            force=True,
        )
        self.run("Importar e auditar base real original", command)

    def split_real(self) -> None:
        self.run("Congelar split real", self.command("split_real.py", force=True))

    def materialize_controlled(self) -> None:
        self.run(
            "Materializar condição controlled",
            self.command("materialize_controlled.py", force=True),
        )

    def preprocess(self) -> None:
        command = self.command(
            "preprocess_assets.py",
            "--stage",
            "all",
            "--device",
            self.args.device,
            force=True,
        )
        self.run("Regenerar recortes e profundidade", command)

    def catalog_assets(self) -> None:
        command = self.command(
            "catalog_assets.py", "--asset-root", str(self.asset_root), force=True
        )
        self.run("Catalogar todos os ativos da síntese", command)

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
                force=True,
            )
            self.run(f"Gerar dataset sintético ({config.stem})", command)

    def materialize_subsets(self) -> None:
        self.run(
            "Materializar subconjuntos sintéticos aninhados 1x–10x",
            self.command("materialize_nx_subsets.py", force=True),
        )

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
        selection = (
            scoped_experiment_root(artifacts, self.experiment, "artifact")
            / "model_selection.json"
        )
        if not selection.exists() and not self.args.dry_run:
            raise FileNotFoundError(
                "teste externo permanece bloqueado até model_selection.json ser congelado"
            )
        source = (
            ["--source", str(self.args.external_source.expanduser().resolve())]
            if self.args.external_source
            else []
        )
        download = self.command("download_external.py", name, *source, force=True)
        self.run(f"Obter teste externo ({name})", download)
        dataset = self.pipeline["external_datasets"][name]
        archive = (
            project_path(self.pipeline["paths"]["archives"])
            / "external"
            / dataset["archive_name"]
        )
        command = self.command(
            "import_external_test.py", name, "--source", str(archive), force=True
        )
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
            force=True,
        )
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
        self.download("raw")
        self.import_real()
        self.split_real()
        self.preprocess()
        self.materialize_controlled()
        self.catalog_assets()
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


STAGES = {
    "download-raw": lambda workflow: workflow.download("raw"),
    "download-real": Workflow.import_real,
    "preprocess": Workflow.preprocess,
    "import-real": lambda workflow: (workflow.import_real(), workflow.split_real()),
    "split-real": Workflow.split_real,
    "materialize-controlled": Workflow.materialize_controlled,
    "catalog-assets": Workflow.catalog_assets,
    "synthesize": Workflow.synthesize,
    "materialize-subsets": Workflow.materialize_subsets,
    "validate": Workflow.validate,
    "train": Workflow.train,
    "select": Workflow.select,
    "test": Workflow.test,
    "prepare-test": Workflow.prepare_external,
    "report": Workflow.report,
    "prepare": Workflow.prepare,
    "all": Workflow.all,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestrador da reprodução do experimento."
    )
    parser.add_argument("stage", choices=("help", *STAGES))
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
    if args.stage == "help":
        parser.print_help()
        return
    STAGES[args.stage](Workflow(args))


if __name__ == "__main__":
    main()
