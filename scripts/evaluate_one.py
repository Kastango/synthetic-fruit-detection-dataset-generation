#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.training import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia um checkpoint selecionado no teste real."
    )
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_checkpoint(json.loads(args.spec.read_text(encoding="utf-8")))
    print(json.dumps(result["test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
