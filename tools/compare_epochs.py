#!/usr/bin/env python3
"""
Direct Qualitative & Quantitative Comparative Analysis: Epoch 1 vs Epoch 2 Checkpoints.
Downloads or loads both checkpoints from Hugging Face Hub, runs inference on benchmark samples,
and outputs a detailed side-by-side span and XML diff report.
"""

import os
import sys
import difflib
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import torch
from transformers import AutoTokenizer

from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping


DEFAULT_REPO = "daominhwysi/mmbert-small-vi-exam-seq-labeling"
EPOCH1_REV = "6a66c3434eafdde032997ed90d4ad97fc5b459d2"
EPOCH2_REV = "934b6a53bac15e17763d700ad3016f86db2c6ddb"


def load_model_for_revision(repo_id: str, revision: str, device: str = "cpu"):
    print(f"Loading checkpoint '{repo_id}' @ revision '{revision[:8]}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    tag_to_id, id_to_tag = load_label_mapping(repo_id, revision=revision)
    model = EnhancedBertForTokenClassification.from_pretrained(
        repo_id,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id,
        revision=revision
    ).to(device)
    model.eval()
    return model, tokenizer, id_to_tag


def run_comparison(
    sample_paths: List[Path],
    repo_id: str = DEFAULT_REPO,
    rev1: str = EPOCH1_REV,
    rev2: str = EPOCH2_REV,
    output_dir: Path = Path("artifacts/samples_predictions/epoch_comparison"),
    device: str = "cpu",
    batch_size: int = 4
):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    report_lines.append("# Side-by-Side Evaluation: Epoch 1 vs Epoch 2\n")
    report_lines.append(f"- **Model Repo**: `{repo_id}`")
    report_lines.append(f"- **Epoch 1 Revision**: `{rev1[:8]}` (Training Log: Val Loss 0.3580 | F1 86.41%)")
    report_lines.append(f"- **Epoch 2 Revision**: `{rev2[:8]}` (Training Log: Val Loss 0.4059 | F1 88.14%)\n")

    # Load Epoch 1
    m1, tok1, id2tag1 = load_model_for_revision(repo_id, rev1, device=device)
    # Predict for all samples with Epoch 1
    preds_e1 = {}
    for p in sample_paths:
        text = p.read_text(encoding="utf-8")
        print(f"[Epoch 1] Predicting {p.name} ({len(text)} chars)...")
        res1 = predict_text(text, m1, tok1, id2tag1, device=device, batch_size=batch_size)
        preds_e1[p.name] = res1
    del m1, tok1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load Epoch 2
    m2, tok2, id2tag2 = load_model_for_revision(repo_id, rev2, device=device)
    # Predict for all samples with Epoch 2
    preds_e2 = {}
    for p in sample_paths:
        text = p.read_text(encoding="utf-8")
        print(f"[Epoch 2] Predicting {p.name} ({len(text)} chars)...")
        res2 = predict_text(text, m2, tok2, id2tag2, device=device, batch_size=batch_size)
        preds_e2[p.name] = res2
    del m2, tok2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compare results
    for p in sample_paths:
        name = p.name
        res1 = preds_e1[name]
        res2 = preds_e2[name]
        xml1 = res1["xml_text"]
        xml2 = res2["xml_text"]
        spans1 = res1["spans"]
        spans2 = res2["spans"]

        # Save XML files
        (output_dir / "epoch1").mkdir(exist_ok=True)
        (output_dir / "epoch2").mkdir(exist_ok=True)
        (output_dir / "epoch1" / f"{p.stem}.xml").write_text(xml1, encoding="utf-8")
        (output_dir / "epoch2" / f"{p.stem}.xml").write_text(xml2, encoding="utf-8")

        counts1 = Counter(s["label"] for s in spans1)
        counts2 = Counter(s["label"] for s in spans2)
        all_labels = sorted(set(counts1.keys()) | set(counts2.keys()))

        print(f"\n=======================================================")
        print(f"Sample: {name}")
        print(f"=======================================================")
        print(f"{'Label':<20} | {'Epoch 1 (86.4%)':<16} | {'Epoch 2 (88.1%)':<16}")
        print(f"{'-'*20}-+-{'-'*16}-+-{'-'*16}")
        for lbl in all_labels:
            print(f"{lbl:<20} | {counts1.get(lbl, 0):<16} | {counts2.get(lbl, 0):<16}")
        print(f"{'-'*20}-+-{'-'*16}-+-{'-'*16}")
        print(f"{'TOTAL SPANS':<20} | {len(spans1):<16} | {len(spans2):<16}")

        # Compute diff in XML tags
        diff = list(difflib.unified_diff(
            xml1.splitlines(),
            xml2.splitlines(),
            fromfile=f"Epoch 1 ({name})",
            tofile=f"Epoch 2 ({name})",
            lineterm=""
        ))

        report_lines.append(f"## Sample: `{name}`\n")
        report_lines.append(f"| Label | Epoch 1 (86.41%) | Epoch 2 (88.14%) | Delta |")
        report_lines.append(f"| :--- | :---: | :---: | :---: |")
        for lbl in all_labels:
            c1 = counts1.get(lbl, 0)
            c2 = counts2.get(lbl, 0)
            d = c2 - c1
            d_str = f"+{d}" if d > 0 else str(d)
            report_lines.append(f"| `{lbl}` | {c1} | {c2} | {d_str} |")
        report_lines.append(f"| **Total Spans** | **{len(spans1)}** | **{len(spans2)}** | **{len(spans2) - len(spans1)}** |\n")

        if diff:
            report_lines.append("<details><summary>Click to view XML Tag Diff (first 50 diff lines)</summary>\n")
            report_lines.append("```diff")
            report_lines.extend(diff[:50])
            if len(diff) > 50:
                report_lines.append(f"... ({len(diff) - 50} more diff lines)")
            report_lines.append("```\n</details>\n")
        else:
            report_lines.append("*Both epochs produced 100% identical XML predictions on this document.*\n")

    report_path = output_dir / "comparison_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nDetailed comparative report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Epoch 1 vs Epoch 2 sequence labelling.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO)
    parser.add_argument("--rev1", type=str, default=EPOCH1_REV)
    parser.add_argument("--rev2", type=str, default=EPOCH2_REV)
    parser.add_argument("--samples", nargs="*", default=[
        "artifacts/samples/literature_exam_for_gifted_2.txt",
        "artifacts/samples/official_literature_exam_2025.txt"
    ])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    sample_paths = [Path(s) for s in args.samples if Path(s).exists()]
    if not sample_paths:
        print("No valid sample files found!")
        return

    run_comparison(
        sample_paths=sample_paths,
        repo_id=args.repo_id,
        rev1=args.rev1,
        rev2=args.rev2,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
