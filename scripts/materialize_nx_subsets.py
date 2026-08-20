#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fruit_pipeline.synthesis import materialize_nested_subsets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recorta os subconjuntos aninhados 1x-10x do pool sintético confirmatório."
    )
    parser.add_argument(
        "--pool-root", type=Path, default=Path("data/generated/confirmatory_pool")
    )
    parser.add_argument("--target-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--base-size", type=int, default=104)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = materialize_nested_subsets(
        args.pool_root,
        args.target_root,
        args.multipliers,
        base_size=args.base_size,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
