#!/usr/bin/env python3
"""
CLI Runner for Batch Review of Sequence Labelling Annotated Dataset.

Usage:
  # Run reviewer using OpenAI Codex with GPT-5.6 Luna High
  uv run python tools/batch_review_annotated.py --provider codex --model gpt-5.6-luna --thinking high

  # Test on first 5 documents
  uv run python tools/batch_review_annotated.py --provider codex --model gpt-5.6-luna --thinking high --limit 5

  # Resume review skipping already reviewed files
  uv run python tools/batch_review_annotated.py --provider codex --model gpt-5.6-luna --thinking high

  # Force re-evaluation of all files
  uv run python tools/batch_review_annotated.py --provider codex --model gpt-5.6-luna --thinking high --overwrite
  uv run python tools/batch_review_annotated.py --provider agy --model gemini-3.8-flash --thinking high --overwrite
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sequence_labelling.annotator.reviewer import main

if __name__ == "__main__":
    main()
