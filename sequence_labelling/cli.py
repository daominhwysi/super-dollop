"""Small command-line entry point for processing health checks."""

from __future__ import annotations

import argparse

from .config import DATA_DIR, MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(prog="sequence-label")
    parser.add_argument("command", choices=["paths"])
    args = parser.parse_args()
    if args.command == "paths":
        print(f"data={DATA_DIR}")
        print(f"models={MODEL_DIR}")
