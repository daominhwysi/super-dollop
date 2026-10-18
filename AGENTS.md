# Agent Specifications & Core Architecture Invariants

This document outlines critical architectural constraints, data contracts, and domain-specific rules that all autonomous agents (Parser, Reviewer, Editor, and Auditor) operating on this codebase must strictly observe.

---

## 1. Core Rule: Question Labels Do NOT Have to Be Followed by Stems

> [!IMPORTANT]
> **A `<question_label>` DOES NOT have to be followed by a `<stem>`.**
> 
> In educational and standardized examinations (e.g. English cloze tests, fill-in-the-blank passages, reading sections with numbered blanks `(1)`, `(2)`, `(3)...`), questions legitimately consist of only:
> ```xml
> <question_label>## Câu 631</question_label>
> <option_label>A.</option_label> <option_text>This chance did not happen</option_text>
> <option_label>B.</option_label> <option_text>By happening this chance</option_text>
> <option_label>C.</option_label> <option_text>This happen did not by chance</option_text>
> <option_label>D.</option_label> <option_text>This did not happen by chance</option_text>
> ```
> In these formats, the prompt/blank is embedded directly within the shared reading passage (or stimulus), and the question itself contains only choices.

### Agent Behavioral Invariants for Stems:
1. **Reviewer Agent (`AnnotationReviewerAgent` / LLM Semantic Reviewer)**:
   - **NEVER** flag `<question_label>` followed immediately by `<option_label>` as a `missing_stem` defect or error.
   - Cloze / fill-in items without explicit `<stem>` tags are fully valid sequence labeling outputs.
   - Do NOT deduct points or route documents to `NEEDS_REVISION` or `DISCARD` due to absent stems in questions that have valid choices.

2. **Editor Agent (`EditorAgent` / Surgical Repair)**:
   - **NEVER** attempt to synthesize, hallucinate, or insert artificial `<stem>` tags into cloze or fill-in questions where none exist in the raw source document.
   - Only edit stems if a stem existed in the raw text but was improperly wrapped or omitted.

3. **Parser Agent (`ParserAgentWorker` / `LongContextParserPipeline`)**:
   - When encountering question numbers that directly index options or blanks in a passage, emit `<question_label>` followed immediately by `<option_label>` without forcing an empty `<stem></stem>`.

4. **Deterministic Auditor (`DeterministicAuditor`)**:
   - The deterministic checks do **not** require stem presence.
   - Only empty `<stem></stem>` tags with whitespace are flagged; completely absent `<stem>` elements in choice questions are valid.

---

## 2. Deterministic Pipeline Invariant Check

A full audit of the project's deterministic (DET) components confirms:
* **`DeterministicParser` (`parse_chunk_deterministic`)**: Uses interval scoping `[q.end, owned_options[0].start]`. When options start immediately after the label, `_make_span` returns `None` and no stem span is emitted. Parsing proceeds without error.
* **`parse_spans_into_structured_questions`**: Checks `(current_q["stem"] or current_q["options"])`. Questions with options and empty stems are correctly structured into questions.
* **`serializer_validator.py`**: Looks for stems matching `q_lab.end <= s.start < next_q`. If none match, `stem_text = ""` and question serialization completes successfully.
* **`DeterministicAuditor.check_question_coverage_via_det`**: Matches questions based on question numbers and clean labels, correctly matching questions even when either or both have no stem.

---

## 3. Reviewer & Decision Logic Invariants

* **Gating on Major Defects**: A document with unaddressed `CRITICAL` or `MAJOR` issues (e.g. unannotated questions, corrupted stimuli) must never be bypassed to `PASS` purely because of a high composite score.
* **Deterministic Confirmation Accounting**: When deterministic issues are confirmed by the LLM reviewer as true positives, they must be factored into the penalty and routing pipeline.

---

## 4. Research Reports & Artifacts Storage Convention

> [!IMPORTANT]
> **All research reports, benchmark analyses, evaluation documents, and exploratory artifacts MUST be written to `artifacts/reports/`.**
> 
> Future agent sessions, subagents, and automated research workflows must strictly adhere to the following file layout rules:
> 1. Store all research reports, architectural assessments, and model evaluation documents in `artifacts/reports/<topic_slug>.md`.
> 2. Do NOT save exploratory research notes or benchmark evaluations in the repository root or arbitrary directories.
> 3. Maintain clean separation between research artifacts (`artifacts/reports/`), raw OCR logs (`artifacts/ocr_logs/`), and pipeline batch outputs (`data/sequence_labelling_annotated/`).

---

## 5. Hardware Pre-Flight Check Before Computational Workloads

> [!CAUTION]
> **ALWAYS check system hardware capabilities BEFORE launching tasks that consume substantial computational power.**
> 
> Before executing model fine-tuning, whole-dataset evaluation, sliding-window transformer inferences, or embeddings generation:
> 1. **Check GPU Availability**: Run a fast hardware inspection (e.g. `torch.cuda.is_available()`, device count, GPU memory).
> 2. **Never Run Unconstrained Heavy Transformer Loops on CPU**:
>    - Evaluating a 300M+ parameter model over 60+ documents or thousands of sliding-window chunks on host CPU will throttle CPU cores for 15+ minutes and lock up the workspace.
>    - If CUDA is absent or CPU-only:
>      - **Prohibited**: Running unconstrained full-dataset neural inference or training.
>      - **Required Alternative**: Use `--limit 1` / smoke tests, mocked passes, or fast deterministic evaluation (`DeterministicParser` / `--use_det` which finishes in milliseconds).
>    - Full model fine-tuning and massive benchmark inference runs are strictly reserved for GPU accelerators (T4 / V100 / A100 / H100).

---

## 6. Default Transformer Backbone: `aisingapore/SEA-LION-ModernBERT-300M`

> [!IMPORTANT]
> **The canonical default transformer backbone for sequence labelling is `aisingapore/SEA-LION-ModernBERT-300M`.**
> 
> (Upgraded from legacy `jhu-clsp/mmBERT-base`).
> 
> - **Rationale**:
>   - ModernBERT architecture with 8192 token context window, Unpad + FlashAttention, and Rotary Embeddings (RoPE).
>   - Significantly superior tokenization compression and preservation for Vietnamese examination delimiters (`## Câu`, choice markers `A.`, `B.`) compared to mmBERT.
>   - Sub-millisecond sequence labelling latency budget.
> - **Configuration Invariant**:
>   - All default configs (`configs/train_config.yaml`), training scripts (`model/train.py`), and model loaders must target `aisingapore/SEA-LION-ModernBERT-300M`.
>   - If an agent or script forgets, switch the backbone back to `aisingapore/SEA-LION-ModernBERT-300M`.

---

## 7. Project Architecture & Directory Layout

### 7.1 Architecture Overview

The system operates as an end-to-end pipeline designed to ingest unstructured Vietnamese examination documents (PDFs/images), produce 100% character-preserved XML annotations, audit and surgically repair defects, compile tokenized training splits, and train/serve sub-millisecond neural sequence labelling models:

```
[Raw Exam PDFs/Images]
         │
         ▼  (tools/ocr/batch_ocr_raw_dataset.py / sequence_labelling/annotator/pdf_converter.py)
[Standardized Page-Aware Markdown with OCR Metadata]
         │
         ▼  (sequence_labelling/parser/long_parser/greedy_chunker.py)
[Passage-Locked Chunks (Target ~4K, Max ~6.5K Tokens)]
         │
         ├──► [Deterministic Parser] (sequence_labelling/parser/deterministic_parser.py) ──► Fast Path (<0.5ms)
         │
         ▼  (sequence_labelling/parser/long_parser/parser_agent_worker.py)
[LLM Parser Worker / Anchored XML Parser] (Gemini 3.8 Flash High / GPT-5.6 Luna)
         │
         ▼  (sequence_labelling/parser/long_parser/sequence_reconciler.py & source_merger/)
[Full Reconciled & Merged Exam Annotations (merged.xml, merged.json)]
         │
         ▼  (tools/review/batch_review_annotated.py / sequence_labelling/annotator/reviewer.py)
[Deterministic Auditor + LLM Semantic Reviewer] ──► Score [0-100] & Routing (PASS / NEEDS_REVISION / DISCARD)
         │
         ▼  (tools/review/edit_review_set.py / sequence_labelling/annotator/editor.py)
[Surgical Editor Loop] (Isolated branch repair in data/sequence_labelling_annotated_branch/)
         │
         ▼  (tools/annotation/merge_branch_to_annotated.py)
[Canonical Verified Dataset (data/sequence_labelling_annotated/)]
         │
         ▼  (tools/dataset/build_training_dataset.py)
[Tokenized BIO Training Splits (Parquet / JSONL / Hugging Face Hub)]
         │
         ▼  (model/train.py & model/inference/)
[ModernBERT Token Classification Head + Constrained Viterbi Decoding]
```

---

### 7.2 Directory Structure & Subsystem Breakdown

```
vietnamese-sequence-labelling-v2/
├── AGENTS.md                                # Core architecture invariants, contracts, and system directory map
├── README.md                                # User and developer operational handbook
├── pyproject.toml                           # uv package manager specifications and tool dependencies
│
├── configs/                                 # Centralized pipeline configurations
│   ├── parser_config.yaml                   # Model endpoints, providers, chunking budgets, and agent prompts
│   └── train_config.yaml                    # Hyperparameters for ModernBERT training, batch size, and LR
│
├── sequence_labelling/                      # Core production sequence labelling library
│   ├── annotator/                           # OCR conversion, XML auditing, review, and surgical editing
│   │   ├── pdf_converter.py                 # Multi-engine PDF/image vision OCR with optional RF-DETR figures
│   │   ├── agy_pdf_ocr.py                   # Specialized Google Antigravity PDF vision OCR converter
│   │   ├── annotate_ocr.py                  # Core XML tag extraction, span parsers, and CoNLL export
│   │   ├── reviewer.py                      # AnnotationReviewerAgent, composite scoring, and routing engine
│   │   ├── editor.py                        # Surgical EditorAgent using precise search-and-replace XML repairs
│   │   ├── revision_loop.py                 # Multi-turn isolated reviewer-editor remediation controller
│   │   ├── xml_checker.py                   # Low-level structural XML validator and defect classifier
│   │   └── xml_cleaner.py                   # Deterministic XML sanitizer and repair normalizer
│   │
│   ├── parser/                              # Parsing engines and reconciliation algorithms
│   │   ├── deterministic_parser.py          # Pure regex/interval parser for zero text corruption (<0.5ms)
│   │   └── long_parser/                     # Long-context multi-chunk parser framework
│   │       ├── greedy_chunker.py            # Passage-locked page chunking avoiding cross-question splits
│   │       ├── parser_agent_worker.py       # Worker wrapping LLM calls for XML annotation extraction
│   │       ├── anchored_xml_llm_parser.py   # Self-closing stimulus anchor tag parsing and reconciliation
│   │       ├── sequence_reconciler.py       # Inter-chunk state stack and boundary seam reconciler
│   │       ├── linking_agent.py             # Graph resolver mapping orphan questions to stimuli
│   │       └── source_merger/               # Character-exact chunk stitching enforcing zero drift
│   │           ├── merger.py                # Source-driven sequence merger algorithm
│   │           └── serializer_validator.py  # Validation and serialization of final structured JSON/XML
│   │
│   ├── llm/                                 # LLM client abstractions and telemetry
│   │   ├── agy_adapter.py                   # Antigravity CLI subprocess bridge with session caching
│   │   ├── agy_rate_limiter.py              # Rolling usage limiter, budget ceiling, and 5h-cap recovery
│   │   ├── deepseek_client.py               # Direct OpenAI-compatible API client with exponential backoff
│   │   └── token_tracker.py                 # Per-session input/output token and cost ledger
│   │
│   ├── api.py                               # FastAPI background job service and OpenAI-compatible endpoints
│   └── config.py                            # Runtime environment and configuration resolver
│
├── model/                                   # Neural sequence labelling model training and inference
│   ├── module/                              # Neural architecture modules
│   │   ├── head.py                          # EnhancedBertForTokenClassification with classification head
│   │   └── dataset.py                       # PyTorch Dataset supporting sliding-window BIO chunking
│   │
│   ├── inference/                           # Inference runtime and post-processors
│   │   ├── predict.py                       # Single-document sliding-window prediction and XML builder
│   │   ├── predict_folder.py                # Batch directory inference evaluator
│   │   └── viterbi.py                       # C-accelerated Constrained Viterbi decoding enforcing BIO syntax
│   │
│   └── train.py                             # Distributed PyTorch training with FP16/BF16, evaluation, & F1
│
├── synthetic_exam_generator/                # Synthetic exam generation and data augmentation engine
│   ├── curriculum.py                        # Curriculum progression generator across difficulty tiers
│   ├── exam_compiler.py                     # Compiles raw templates and variables into coherent exams
│   ├── template_bank.py                     # High-school math question templates and distractors
│   ├── reconstructor.py                     # Ground-truth permutation, typo injection, and style variation
│   └── parser.py                            # Synthetic template AST parser
│
├── tools/                                   # Categorized operational CLI tools and batch workflows
│   ├── ocr/                                 # Document ingestion and Vision OCR batch tooling
│   │   ├── batch_ocr_raw_dataset.py         # Batch PDF OCR runner producing page-level Markdown
│   │   ├── run_agy_batch_ocr.py             # Multi-worker AGY OCR orchestrator with auto-cap recovery
│   │   ├── check_batch_status.py            # Real-time terminal status inspector for batch OCR runs
│   │   ├── convert_pdfs_to_markdown.py      # Digital PDF-to-Markdown converter with math preservation
│   │   ├── classify_raw_pdfs.py             # PDF classifier (scanned vs digital vs hybrid)
│   │   └── curate_500_diverse_pdfs.py       # Stratified diversity sampler and deduplicator for exams
│   │
│   ├── annotation/                          # Ground-truth sequence labelling annotation pipeline
│   │   ├── annotate_sequence_labelling_dataset.py # Canonical parser annotation pipeline runner
│   │   ├── run_agy_batch_annotation.py      # Resilient multi-worker AGY batch annotation pipeline
│   │   ├── check_annotation_batch_status.py # Real-time dashboard for batch annotation progress
│   │   ├── strip_annotation_tags.py         # Deterministic stripper validating raw text identity
│   │   ├── remerge_annotated_dataset.py     # Batch tool rebuilding merged XMLs from cached chunk outputs
│   │   └── merge_branch_to_annotated.py     # Safely merges approved branch candidates to canonical data
│   │
│   ├── review/                              # Quality auditing, surgical editing, and remediation
│   │   ├── batch_review_annotated.py        # Batch reviewer executing deterministic + LLM audit
│   │   ├── edit_review_set.py               # Surgical Editor/Reviewer revision loop on branch files
│   │   ├── reparse_review_set.py            # Reparses failing documents directly from raw Markdown
│   │   ├── reparse_and_review.py            # Unified end-to-end reparse, audit, and comparison pipeline
│   │   └── run_review_and_edit_500.sh       # Production orchestration shell script for review & repair
│   │
│   ├── evaluation/                          # Model & dataset benchmark evaluation suites
│   │   ├── evaluate_model.py                # Comprehensive 63 Gold Benchmark document evaluation
│   │   ├── evaluate_500_dataset.py          # Invariant, DET, and neural evaluation on 500-doc dataset
│   │   ├── evaluate_trained_model_500.py    # Evaluation of trained Hugging Face checkpoints
│   │   ├── eval_tokens_and_det_f1.py        # Token counts and Deterministic Parser F1 evaluation
│   │   ├── select_gold_benchmark.py         # Stratified token-budgeted gold benchmark set selector
│   │   └── eval_adaptive_scoring_experiment.py # Experimental diminishing-penalty scoring engine
│   │
│   ├── dataset/                             # Dataset compilation and Hugging Face Hub publishing
│   │   ├── build_training_dataset.py        # Compiles BIO sequence chunks into train/test Parquet/JSONL
│   │   ├── upload_dataset_to_hf.py          # Uploads versioned dataset splits and dataset cards to HF
│   │   └── upload_model_to_hf.py            # Uploads model checkpoints, weights, and model cards to HF
│   │
│   ├── monitoring/                          # Cost, rate limit, and telemetry observability
│   │   ├── token_monitor.py                 # Real-time token usage, latency, and cost terminal dashboard
│   │   └── safe_ccusage.py                  # Lock-safe SQLite snapshot bridge for Antigravity telemetry
│   │
│   ├── visualize/                           # Parsing and chunking diagnostic visualizers
│   │   ├── visualize_chunker.py             # Visualizes greedy oversize chunking and passage locks
│   │   ├── visualize_parser.py              # Visualizes parser worker chunk extraction
│   │   ├── visualize_merger.py              # Visualizes chunk reconciliation and seam merging
│   │   └── visualize_linker.py              # Visualizes stimulus-question graph resolution
│   │
│   └── experiments/                         # Research prototypes and experimental OCR architectures
│       ├── chunk_extractor.py               # PyMuPDF digital text chunk and formula extractor
│       ├── digital_chunk_ocr.py             # Hybrid vision/digital hybrid OCR pipeline
│       ├── dynamic_mosaic_packer.py         # 2D bin-packing mosaic compositor for visual crops
│       ├── hybrid_digital_ocr.py            # Dual-path digital/raster exam OCR engine
│       ├── run_chunk_injection_benchmark.py # Benchmark for digital chunk injection performance
│       └── run_experiment_and_benchmark.py  # End-to-end OCR experiment suite runner
│
├── tests/                                   # Automated test suite (237+ passing unit & integration tests)
│   ├── unit/                                # Unit tests for all core components
│   │   ├── ocr/                             # Tests for reviewer, editor, revision loop, checker, cleaner
│   │   ├── synthetic_exam_generator/        # Tests for curriculum, compiler, template bank, reconstructor
│   │   └── test_*.py                        # Tests for adapters, head, viterbi, strip_tags, etc.
│   └── test_merger/                         # Comprehensive source merger and chunk provenance tests
│
├── artifacts/                               # Generated reports, benchmarks, logs, and run outputs
│   ├── reports/                             # Research and evaluation reports (all markdown reports live here)
│   ├── ocr_logs/                            # Raw execution logs from batch OCR and parser jobs
│   └── token_usage/                         # Timestamped JSON token accounting logs
│
└── data/                                    # Local examination datasets (gitignored large assets)
    ├── sequence_labelling_input_data_raw/   # Raw PDF and image input documents
    ├── sequence_labelling_input_data/       # OCR Markdown files produced by batch OCR
    ├── sequence_labelling_annotated/        # Canonical annotated ground truth XML and structured JSON
    └── sequence_labelling_annotated_branch/ # Disposable staging branch for surgical editor candidate repairs
```


