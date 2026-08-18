#!/usr/bin/env python3
"""Atalho para o gerador; grades devem usar scripts/generate_synthetic.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    print(
        "setimages.py agora usa o gerador reprodutível. "
        "Para novos experimentos, prefira ./run_pipeline.sh synthesize.",
        file=sys.stderr,
    )
    arguments = list(sys.argv[1:])
    if "--synthesis-config" not in arguments:
        arguments.extend(
            [
                "--synthesis-config",
                str(ROOT / "configs" / "synthesis" / "depth_robust.yaml"),
            ]
        )
    os.execv(
        sys.executable,
        [sys.executable, str(ROOT / "scripts" / "generate_synthetic.py"), *arguments],
    )
