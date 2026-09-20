#!/usr/bin/env python3
"""
Launcher script for Azozo Sequence Labelling Review & Editor Studio.

Usage:
  uv run python scripts/run_review_studio.py
  uv run python scripts/run_review_studio.py --port 8050
"""

import argparse
import sys
from pathlib import Path
import uvicorn

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Azozo Sequence Labelling Review & Editor Studio Launcher"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to listen on (default: 8050)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code change",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("🚀  AZOZO SEQUENCE LABELLING REVIEW & EDITOR STUDIO")
    print("=" * 65)
    print(f"  URL     : http://{args.host}:{args.port}")
    print(f"  Dataset : /home/daominhwysi/project/vietnamese-exam-seq-labelling/output/real_annotated")
    print(f"  Raw OCR : {WORKSPACE_DIR / 'data' / 'sequence_labelling_input_data'}")
    print("=" * 65)

    uvicorn.run(
        "review_studio.review_studio:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
