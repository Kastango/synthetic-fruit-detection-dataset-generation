#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.training import train_one


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa uma célula isolada da grade YOLO."
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = train_one(spec, args.runs_root.resolve(), force=args.force)
    print(json.dumps(result["validation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
