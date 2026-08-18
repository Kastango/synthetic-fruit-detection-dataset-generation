#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fruit_pipeline.common import ROOT, load_yaml, project_path


class Workflow:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pipeline = load_yaml(project_path(args.pipeline_config))
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
        if self.args.real_source is None:
            manifest = (
                project_path(self.pipeline["paths"]["real_source"]) / "manifest.json"
            )
            if manifest.exists() or self.args.dry_run:
                print(f"\n== Base real ==\nreutilizando {manifest}")
                return
            raise FileNotFoundError("informe --real-source /caminho/datanotation.zip")
        command = self.command(
            "import_real_dataset.py",
            "--source",
            str(self.args.real_source.expanduser().resolve()),
        )
        if self.args.force:
            command.append("--force")
        self.run("Importar e auditar base real original", command)

    def split_real(self) -> None:
        command = self.command("split_real.py")
        if self.args.force:
            command.append("--force")
        self.run("Congelar split real 100/15/15", command)

    def preprocess(self) -> None:
        command = self.command(
            "preprocess_assets.py", "--stage", "all", "--device", self.args.device
        )
        if self.args.force:
            command.append("--force")
        self.run("Regenerar recortes e profundidade", command)

    def split_assets(self) -> None:
        command = self.command("split_assets.py", "--asset-root", str(self.asset_root))
        if self.args.force:
            command.append("--force")
        self.run("Congelar ativos sintéticos de treino/validação", command)

    def synthesis_configs(self) -> list[Path]:
        return self.args.synthesis_config or [
            ROOT / "configs" / "synthesis" / "depth_robust.yaml"
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
                str(self.args.workers),
            )
            if self.args.force:
                command.append("--force")
            self.run(f"Gerar dataset sintético ({config.stem})", command)

    def validate(self, stage: str = "all") -> None:
        command = self.command(
            "validate_data.py", "--stage", stage, "--asset-root", str(self.asset_root)
        )
        self.run(f"Auditar pipeline ({stage})", command)

    def train(self) -> None:
        command = self.command("train_grid.py", "--device", self.args.device)
        if self.args.max_runs is not None:
            command.extend(["--max-runs", str(self.args.max_runs)])
        if self.args.dry_run:
            # O orquestrador executa o dry-run interno para listar a grade.
            subprocess.run(command + ["--dry-run"], cwd=ROOT, check=True)
            return
        if self.args.force:
            command.append("--force")
        self.run("Treinar grade YOLO", command)

    def select(self) -> None:
        self.run("Selecionar por validação real", self.command("select_models.py"))

    def test(self) -> None:
        if not self.args.unlock_test:
            raise SystemExit(
                "use --unlock-test somente depois de congelar model_selection.json"
            )
        command = self.command(
            "evaluate_test.py", "--device", self.args.device, "--unlock-test"
        )
        if self.args.force:
            command.append("--force")
        self.run("Avaliação final no teste real", command)

    def prepare(self) -> None:
        self.download("prepared")
        self.import_real()
        self.split_real()
        self.split_assets()
        self.synthesize()
        self.validate("all")

    def all(self) -> None:
        self.prepare()
        self.train()
        self.select()
        if self.args.unlock_test:
            self.test()
        else:
            print(
                "\nTeste mantido fechado. Revise artifacts/model_selection.json e execute "
                "./run_pipeline.sh test --unlock-test.",
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
            "preprocess",
            "import-real",
            "split-real",
            "split-assets",
            "synthesize",
            "validate",
            "train",
            "select",
            "test",
            "prepare",
            "all",
        ),
    )
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--real-source", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--synthesis-config", action="append", type=Path)
    parser.add_argument("--workers", type=int, default=4)
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
        "preprocess": workflow.preprocess,
        "import-real": lambda: (workflow.import_real(), workflow.split_real()),
        "split-real": workflow.split_real,
        "split-assets": workflow.split_assets,
        "synthesize": workflow.synthesize,
        "validate": workflow.validate,
        "train": workflow.train,
        "select": workflow.select,
        "test": workflow.test,
        "prepare": workflow.prepare,
        "all": workflow.all,
    }
    if args.stage == "help":
        parser.print_help()
        return
    actions[args.stage]()


if __name__ == "__main__":
    main()
