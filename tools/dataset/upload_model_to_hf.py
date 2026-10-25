#!/usr/bin/env python3
"""
Upload trained sequence labelling model weights and comprehensive Model Card README.md
to Hugging Face Hub.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from huggingface_hub import HfApi
from model.train import generate_model_readme


def parse_args():
    parser = argparse.ArgumentParser(description="Upload trained model weights and update README to Hugging Face Hub.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="results/mmbert_seqlabel_v2",
        help="Path to trained model directory containing weights and configs (default: results/mmbert_seqlabel_v2)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="daominhwysi/sealior-modernbert-vi-exam-seq-labeling",
        help="Hugging Face model repository ID (default: daominhwysi/sealior-modernbert-vi-exam-seq-labeling)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="aisingapore/SEA-LION-ModernBERT-300M",
        help="Base model architecture name (default: aisingapore/SEA-LION-ModernBERT-300M)"
    )
    parser.add_argument(
        "--macro-f1",
        type=float,
        default=None,
        help="Validation Macro F1 score (e.g. 0.852 for 85.20%%)"
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=None,
        help="Validation Precision score"
    )
    parser.add_argument(
        "--recall",
        type=float,
        default=None,
        help="Validation Recall score"
    )
    parser.add_argument(
        "--micro-f1",
        type=float,
        default=None,
        help="Validation Micro F1 score"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model weights and updated comprehensive Model Card README.md",
        help="Commit message for Hugging Face"
    )
    parser.add_argument(
        "--only-readme",
        action="store_true",
        help="Only generate and upload README.md to update model card documentation without re-uploading weights"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token (default: reads from HF_TOKEN env or cached token)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = WORKSPACE_DIR / args.model_dir
    repo_id = args.repo_id
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"=== Hugging Face Model Upload Utility ===")
    print(f"Target Repository : https://huggingface.co/{repo_id}")
    print(f"Source Directory  : {model_dir}")
    print(f"Only Readme Mode  : {args.only_readme}\n")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # 1. Gather model metadata and configs if present
    tag_to_id = None
    label_map_file = model_dir / "label_mapping.json"
    if label_map_file.exists():
        try:
            data = json.loads(label_map_file.read_text(encoding="utf-8"))
            tag_to_id = data.get("tag_to_id", None)
        except Exception:
            pass

    metrics = {}
    if args.macro_f1 is not None:
        metrics["macro_f1"] = args.macro_f1
    if args.precision is not None:
        metrics["precision"] = args.precision
    if args.recall is not None:
        metrics["recall"] = args.recall
    if args.micro_f1 is not None:
        metrics["micro_f1"] = args.micro_f1

    # Look for evaluation metrics in model_dir if not specified via CLI
    metric_candidates = [
        model_dir / "gold_test_metrics.json",
        model_dir / "best" / "gold_test_metrics.json",
        model_dir / "best" / "metrics.json",
        model_dir / "metrics.json",
        model_dir / "best_dev_metrics.json",
        model_dir / "trainer_state.json",
    ]
    for cand in metric_candidates:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                for key in ["macro_f1", "precision", "recall", "micro_f1"]:
                    if key in data and key not in metrics:
                        metrics[key] = float(data[key])
                if "best_macro_f1" in data and "macro_f1" not in metrics:
                    metrics["macro_f1"] = float(data["best_macro_f1"])
                if metrics:
                    print(f"Loaded evaluation metrics from '{cand}': {metrics}")
                    break
            except Exception:
                pass

    # 2. Generate and save Model Card README.md
    print("Generating comprehensive Model Card README.md...")
    readme_content = generate_model_readme(
        repo_id=repo_id,
        metrics=metrics,
        tag_to_id=tag_to_id,
        base_model_name=args.base_model
    )

    readme_path = model_dir / "README.md"
    if model_dir.exists():
        readme_path.write_text(readme_content, encoding="utf-8")
        print(f"Saved local README.md ({len(readme_content):,} bytes) at '{readme_path}'.")

    # 3. Upload to Hugging Face Hub
    if args.only_readme:
        print(f"Uploading README.md directly to {repo_id}...")
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update Model Card README.md documentation"
        )
        print(f"\nModel Card README successfully updated at: https://huggingface.co/{repo_id}")
        return

    if not model_dir.exists():
        print(f"Error: Model directory '{model_dir}' does not exist.")
        sys.exit(1)

    print(f"Uploading full model bundle from '{model_dir}' to '{repo_id}'...")
    t0 = time.time()
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=["checkpoint-*", "checkpoint-*/*", "*.tmp", "*.log"]
    )
    elapsed = time.time() - t0
    print(f"\nModel weights and README successfully uploaded in {elapsed:.1f}s!")
    print(f"Available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
