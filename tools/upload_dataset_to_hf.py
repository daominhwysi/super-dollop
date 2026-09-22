#!/usr/bin/env python3
"""
Upload compiled sequence labelling training & test datasets to Hugging Face Hub,
automatically updating and replacing the repository README.md with comprehensive v2 documentation.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from huggingface_hub import HfApi

WORKSPACE_DIR = Path(__file__).resolve().parent.parent


def generate_hf_readme(manifest: dict) -> str:
    """Generates comprehensive Markdown documentation for the v2 Hugging Face dataset."""
    train = manifest.get("train", {})
    test = manifest.get("test_validation", {})

    train_docs = train.get("total_documents", 0)
    real_pristine = train.get("real_pristine_documents", 0)
    real_aug = train.get("real_augmented_documents", 0)
    synth_docs = train.get("synthetic_documents", 0)
    train_tokens = train.get("total_tokens", 0)
    train_chunks = train.get("total_bio_chunks", 0)

    test_docs = test.get("total_documents", 0)
    test_tokens = test.get("total_tokens", 0)
    test_chunks = test.get("total_bio_chunks", 0)

    train_tags = train.get("tag_distribution", {})
    tag_rows = []
    for tag, count in train_tags.items():
        test_count = test.get("tag_distribution", {}).get(tag, 0)
        tag_rows.append(f"| `{tag}` | {count:,} | {test_count:,} |")
    tag_table = "\n".join(tag_rows) if tag_rows else "| *N/A* | - | - |"

    readme_content = f"""---
language:
- vi
license: apache-2.0
task_categories:
- token-classification
task_ids:
- named-entity-recognition
tags:
- vietnamese
- exam
- sequence-labeling
- educational
- synthetic
- ocr
size_categories:
- 10K<n<100K
pretty_name: Vietnamese Exam Sequence Labelling Dataset v2
---

# Vietnamese Exam Sequence Labelling Dataset (v2)

A comprehensive, production-grade dataset for **token-level and span-level sequence labelling** of Vietnamese educational examination documents (covering grades 8–12 across Mathematics, Physics, Chemistry, Biology, History, Geography, Literature, and English).

Generated and curated by the [Vietnamese Sequence Labelling v2](https://github.com/daominhwysi/vietnamese-sequence-labelling-v2) pipeline, combining real OCR-annotated examination documents (both scanned and digital PDFs) with synthetic curriculum exams and an isolated 4-tier Gold Benchmark test suite.

## What's New in v2

1. **Unified Multi-Scale Sliding Windows**: Pre-chunked into multiple context window sizes (`[512, 128]`, `[768, 192]`, `[1024, 256]`, `[2048, 512]`) for transformer encoders (mmBERT, PhoBERT, XLM-RoBERTa).
2. **Three Standard ML Formats**:
   - **BIO Token Chunks** (`train_bio_chunks.jsonl`, `test_bio_chunks.jsonl`) for token classification.
   - **Document Spans** (`train_spans.jsonl`, `test_spans.jsonl`) with character offsets for span-based extractors (LayoutLM, GLiNER, spaCy).
   - **Generative Pairs** (`train_generative.jsonl`, `test_generative.jsonl`) for LLM instruction fine-tuning (Qwen2.5, Llama 3).
3. **Strictly Isolated Gold Benchmark Test Set**:
   - 63 real examination documents (~455k tokens) selected via **Constrained Max-Min Facility Dispersion** on 13-D structural vectors across 4 difficulty tiers.
   - **Zero Data Leakage**: Guaranteed 0 document overlap between training and testing splits.
4. **Pedagogical Diversity & Layout Invariants**:
   - Full support for inline choices (`A. ... B. ... C. ... D. ...`), 2x2 grids, and multi-line options.
   - **Stemless Cloze Questions (`AGENTS.md`)**: Full architectural support for questions where `<question_label>` is followed directly by `<option_label>` without an explicit `<stem>`.
   - Markdown tables and LaTeX math formulas ($...$ and $$...$$) strictly preserved with zero character corruption.

---

## Dataset Splits & Statistics

| Partition | Documents | Tokens | Multi-Scale BIO Chunks | Window Scales | Formats Available |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | **{train_docs:,}**<br>({real_pristine} pristine real + {real_aug} augmented real + {synth_docs} synthetic) | **{train_tokens:,}** | **{train_chunks:,}** | `512`, `768`, `1024`, `2048` | `train_bio_chunks.jsonl`<br>`train_spans.jsonl`<br>`train_generative.jsonl` |
| **Test / Val Set (Gold Benchmark)** | **{test_docs:,}**<br>(100% pristine real across 4 tiers) | **{test_tokens:,}** | **{test_chunks:,}** | `512`, `768`, `1024`, `2048` | `test_bio_chunks.jsonl`<br>`test_spans.jsonl`<br>`test_generative.jsonl` |

---

## Tag Set & Distribution

| BIO Tag | Training Count | Test/Validation Count |
| :--- | :---: | :---: |
{tag_table}

### Entity Definitions
- `O`: Outside — unlabelled text or background layout.
- `B-QUESTION_LABEL` / `I-QUESTION_LABEL`: Question prefix markers (e.g. `Câu 1:`, `Question 2.`).
- `B-STEM` / `I-STEM`: Main question prompt or problem statement.
- `B-OPTION_LABEL` / `I-OPTION_LABEL`: Choice prefixes (e.g. `A.`, `B.`, `C.`, `D.`).
- `B-OPTION_TEXT` / `I-OPTION_TEXT`: Text content of each choice.
- `B-STIMULUS` / `I-STIMULUS`: Shared reading passage or stimulus block for passage-based questions.
- `B-SECTION` / `I-SECTION`: Section headers and instructions (e.g. `PHẦN I. TRẮC NGHIỆM`).
- `B-EXPLANATION` / `I-EXPLANATION`: Answer keys, detailed solutions, and scoring barems.
- `B-FIGURE` / `I-FIGURE`: Reference to illustrations, diagrams, and figures.

---

## Data Schema Examples

### Format 1: BIO Token Chunks (`train_bio_chunks.jsonl`, `test_bio_chunks.jsonl`)
```json
{{
  "chunk_id": "exam-185--697ed7686201_chunk_000",
  "doc_id": "exam-185--697ed7686201",
  "start_token_idx": 0,
  "end_token_idx": 512,
  "window_size": 512,
  "stride": 128,
  "tokens": ["Câu", "1", ":", "Một", "vật", "dao", "động", "điều", "hòa", "với", "phương", "trình", "..."],
  "ner_tags": ["B-QUESTION_LABEL", "I-QUESTION_LABEL", "I-QUESTION_LABEL", "B-STEM", "I-STEM", "I-STEM", "..."]
}}
```

### Format 2: Document Spans (`train_spans.jsonl`, `test_spans.jsonl`)
```json
{{
  "doc_id": "exam-185--697ed7686201",
  "source": "real",
  "text": "Câu 1: Một vật dao động điều hòa...\\nA. 2s\\nB. 4s",
  "spans": [
    {{"start": 0, "end": 7, "label": "question_label", "text": "Câu 1:"}},
    {{"start": 8, "end": 37, "label": "stem", "text": "Một vật dao động điều hòa..."}},
    {{"start": 38, "end": 43, "label": "option_label", "text": "A."}},
    {{"start": 44, "end": 46, "label": "option_text", "text": "2s"}}
  ]
}}
```

### Format 3: Generative Instruction Tuning (`train_generative.jsonl`, `test_generative.jsonl`)
```json
{{
  "doc_id": "exam-185--697ed7686201",
  "source": "real",
  "prompt": "Sequence label the following examination text using standard XML structural tags:\\n\\nCâu 1: Một vật dao động điều hòa... A. 2s B. 4s",
  "response": "<question_label>Câu 1:</question_label> <stem>Một vật dao động điều hòa...</stem>\\n<option_label>A.</option_label> <option_text>2s</option_text>\\n<option_label>B.</option_label> <option_text>4s</option_text>"
}}
```

---

## How to Load

```python
from datasets import load_dataset

# 1. Load token classification sliding-window chunks:
dataset = load_dataset(
    "daominhwysi/synthetic-seq-labelling-vi-exam-v2",
    data_files={{
        "train": "train_bio_chunks.jsonl",
        "test": "test_bio_chunks.jsonl"
    }}
)
print(dataset["train"][0])

# 2. Load document-level character spans:
spans_dataset = load_dataset(
    "daominhwysi/synthetic-seq-labelling-vi-exam-v2",
    data_files={{
        "train": "train_spans.jsonl",
        "test": "test_spans.jsonl"
    }}
)
print(spans_dataset["train"][0])
```

---

## Citation

```bibtex
@misc{{vi-seq-labelling-v2,
  title  = {{Vietnamese Exam Sequence Labelling Dataset v2}},
  author = {{daominhwysi}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/daominhwysi/synthetic-seq-labelling-vi-exam-v2}}
}}
```
"""
    return readme_content.strip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="daominhwysi/synthetic-seq-labelling-vi-exam-v2",
        help="Hugging Face dataset repository ID (default: daominhwysi/synthetic-seq-labelling-vi-exam-v2)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training_dataset",
        help="Directory containing dataset files to upload (default: data/training_dataset)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/training_dataset/dataset_manifest.json",
        help="Path to dataset manifest JSON to generate README from"
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip generating and uploading README.md"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Update v2 dataset with documentation, multi-scale BIO chunks, Spans, and Gold Benchmark Test set",
        help="Commit message for upload"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = WORKSPACE_DIR / args.data_dir
    manifest_path = WORKSPACE_DIR / args.manifest

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    # 1. Generate updated README.md if manifest is present
    if not args.skip_readme and manifest_path.exists():
        try:
            print(f"Generating updated README.md from '{manifest_path.name}'...")
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            readme_text = generate_hf_readme(manifest_data)
            readme_file = data_dir / "README.md"
            readme_file.write_text(readme_text, encoding="utf-8")
            print(f"Saved updated README.md ({len(readme_text):,} bytes) to {readme_file}")
        except Exception as e:
            print(f"Warning: Failed to generate README.md from manifest: {e}")

    # 2. Discover files to upload
    files_to_upload = sorted([f for f in data_dir.iterdir() if f.is_file()])
    if not files_to_upload:
        print(f"Error: No files found in '{data_dir}'.")
        sys.exit(1)

    print("\n============================================================")
    print(f"Hugging Face Dataset Upload")
    print(f"  Target Repo  : {args.repo_id}")
    print(f"  Source Dir   : {data_dir}")
    print(f"  Files ({len(files_to_upload)}):")
    total_bytes = 0
    for f in files_to_upload:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_bytes += f.stat().st_size
        print(f"    - {f.name:<25} ({size_mb:8.2f} MB)")
    print(f"  Total Size   : {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.2f} GB)")
    print("============================================================\n")

    api = HfApi()
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name', 'Unknown')} ({user_info.get('fullname', '')})")

    # 3. Upload each file sequentially with status
    start_time = time.time()
    for idx, f in enumerate(files_to_upload, 1):
        file_start = time.time()
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"[{idx}/{len(files_to_upload)}] Uploading {f.name} ({size_mb:.2f} MB)...")
        try:
            commit_info = api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Upload {f.name} (v2 dataset)"
            )
            elapsed = time.time() - file_start
            print(f"  --> Successfully uploaded {f.name} in {elapsed:.1f}s.")
        except Exception as e:
            print(f"  --> Error uploading {f.name}: {e}")
            raise e

    total_elapsed = time.time() - start_time
    print(f"\nAll files successfully uploaded in {total_elapsed:.1f}s!")
    print(f"Dataset URL: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
