#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from fruit_pipeline.common import load_yaml, project_path
from fruit_pipeline.download import obtain_source, selected_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa, retoma, valida e extrai os dados públicos do projeto."
    )
    parser.add_argument(
        "source",
        choices=("prepared", "raw", "all", "fruits", "backgrounds"),
        help="prepared é o pacote pronto; raw permite reconstruir recortes/profundidade",
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--accept-data-terms",
        action="store_true",
        help="Confirma que o usuário possui autorização para usar os dados",
    )
    args = parser.parse_args()
    if not args.verify_only and not args.accept_data_terms:
        raise SystemExit(
            "a licença dos arquivos de campo não está declarada no repositório; "
            "revise a procedência e use --accept-data-terms para confirmar o uso"
        )
    config = load_yaml(project_path(args.config))
    reports = [
        obtain_source(
            source,
            config,
            force=args.force,
            keep_archive=args.keep_archive,
            verify_only=args.verify_only,
        )
        for source in selected_sources(args.source)
    ]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
