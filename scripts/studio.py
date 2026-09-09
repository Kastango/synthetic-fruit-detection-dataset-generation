#!/usr/bin/env python3
import argparse
from pathlib import Path
from fruit_pipeline.studio import serve

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Interface local para criar e inspecionar cenas sintéticas com sliders."
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--asset-root", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    serve(a.host, a.port, a.asset_root, a.output)
