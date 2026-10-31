# Agent Specifications & Core Architecture Invariants

This document outlines critical architectural constraints, data contracts, and domain-specific rules that all autonomous agents (Parser, Reviewer, Editor, and Auditor) and developers operating on this codebase must strictly observe.

---

## 1. Core Annotation & Sequence Labelling Invariants

### 1.1 Question Labels Do NOT Have to Be Followed by Stems

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

#### Agent Behavioral Invariants for Stems:
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

### 1.2 Markdown Heading Prefix Invariant (`## Câu 1.`)

> [!IMPORTANT]
> **Leading Markdown heading markers (`## `, `### `, `# `) immediately preceding a question label belong INSIDE `<question_label>`.**
> 
> Example:
> ```xml
> <!-- CORRECT -->
> <question_label>## Câu 1.</question_label>
> <stem>Cho hàm số f(x)...</stem>
> 
> <!-- FORBIDDEN: split into section and question label -->
> <section>## Câu 1.</section><question_label>Câu 1.</question_label>
> 
> <!-- FORBIDDEN: swallowed into preceding stem or option text -->
> <option_text>D. 4 ##</option_text> <question_label>Câu 2.</question_label>
> ```
> 
> - Question labels must capture their markdown header syntax wholly so that stripping tags reproduces the exact original document without leaving orphan `#` tokens.
> - `DeterministicParser` applies `_heading_start` interval scoping to guarantee `#` markers on the label line are attributed to `question_label` and not swallowed by preceding stems or options.

---

### 1.3 Sentence Arrangement Questions Invariant

> [!IMPORTANT]
> **In sentence and paragraph arrangement questions, sentence markers `a., b., c., d., e.` belong INSIDE `<stem>`, NOT `<option_label>`.**
> 
> Standardized examinations often contain sentence arrangement questions such as:
> ```text
> Câu 15. Sắp xếp các câu sau thành đoạn văn hoàn chỉnh:
> a. Mặt trời vừa mọc.
> b. Chúng tôi bắt đầu chuyến hành trình.
> c. Cảnh vật dần hiện ra rõ nét.
> A. a - b - c        B. b - a - c        C. c - a - b        D. a - c - b
> ```
> In these items, propositions `a., b., c.` are sentence components within the prompt text. The actual choices being selected are `A., B., C., D.`.
> 
> - Tag sentences `a., b., c.` inside `<stem>...</stem>`.
> - Tag choices `A.`, `B.`, `C.`, `D.` as `<option_label>` and their permutation text as `<option_text>`.
> - Do NOT misclassify `a.`, `b.`, `c.` as `<option_label>` in this question type.

---

### 1.4 Section-Bounded Stimulus Attribution

> [!IMPORTANT]
> **Intervening `<section>` dividers strictly close preceding `<stimulus>` scope.**
> 
> In structured exams with multiple parts (e.g., `Phần I: Trắc nghiệm nhiều lựa chọn`, `Phần II: Trắc nghiệm đúng/sai`, `Phần III: Trả lời ngắn`), a stimulus from an earlier section must **never** leak or be attributed to questions in subsequent sections.
> 
> - [`serializer_validator.py`](sequence_labelling/parser/long_parser/source_merger/serializer_validator.py) checks for any intervening `<section>` tags between a candidate stimulus end and a question start. If a section heading exists, the preceding stimulus association is terminated.

---

### 1.5 Architectural Anti-Pattern: Rejection of Post-Hoc Regex Mutators (`xml_cleaner.py`)

> [!CAUTION]
> **`xml_cleaner.py` and post-hoc regex string-mutating cleaners are classified as ARCHITECTURAL ANTI-PATTERNS.**
> 
> The core pipeline guarantees **100% Verbatim Source Fidelity**. Relying on heuristic regex scripts (`xml_cleaner.py`) to auto-close, auto-splice, or rebalance XML tags post-generation is strictly deprecated for the following reasons:
> 1. **Masking Root Errors**: Post-hoc regex cleaners treat superficial symptoms while hiding underlying prompt ambiguities and few-shot flaws.
> 2. **Corruption Risk**: Heuristic regex substitutions risk introducing unintended line breaks, corrupting mathematical LaTeX expressions (e.g. `$x < 0$` or `\begin{matrix}`), and invalidating character offset alignments.
> 3. **"First-Time Right" Principle**: All structural boundaries (stems before tables, cloze blanks, question prefixes) must be enforced at generation time via precise Prompt Specifications and mathematically verified Few-Shot Examples.
> 4. **Read-Only Validation**: Post-generation checks must be strictly non-mutating validators (such as `xml_checker.py`), while structural corrections must occur via LLM diff-patching (`EditorAgent`), never through heuristic regex mutation.

---

## 2. Deterministic Pipeline Invariant Check

A full audit of the project's deterministic (DET) components confirms:
* **`DeterministicParser` (`parse_chunk_deterministic`)**: Uses interval scoping `[q.end, owned_options[0].start]`. When options start immediately after the label, `_make_span` returns `None` and no stem span is emitted. Leading `#` markers are captured into the question label via `_heading_start`. Parsing proceeds without error in `<0.5ms`.
* **`parse_spans_into_structured_questions`**: Evaluates `(current_q["stem"] or current_q["options"])`. Questions with options and empty stems are correctly structured into standard question objects.
* **`serializer_validator.py`**: Identifies stems matching `q_lab.end <= s.start < next_q`. If none match, `stem_text = ""` and serialization completes with zero text distortion. Preceding stimulus lookup verifies no section boundaries intervene.
* **`DeterministicAuditor.check_question_coverage_via_det`**: Matches questions based on parsed question numbers and normalized labels, correctly matching questions even when either or both lack explicit stems.

---

## 3. Reviewer, Editor & Dual-Granularity Invariants

### 3.1 Gating on Major & Critical Defects
* **Gating on Major Defects**: A document with unaddressed `CRITICAL` or `MAJOR` issues (e.g. unannotated questions, corrupted stimuli, severed question boundaries) must never be bypassed to `PASS` purely because of a high composite score.
* **Deterministic Confirmation Accounting**: When deterministic issues are confirmed by the LLM reviewer as true positives, they must be factored into penalty calculations and routing decisions.

### 3.2 Dual-Granularity Auditing & Repair (`chunk` vs `document`)
* Default processing granularity is `--granularity chunk` (Chunk-as-Samples architecture).
* Chunk-level auditing evaluates individual ~4K-token units with surgical focus, preventing context degradation from long multi-page documents.
* Document-level auditing aggregates chunk scores and audits cross-chunk global invariants (stimulus multi-question rule, whole-exam question numbering continuity).

### 3.3 Unambiguous Surgical Patching
* `SearchReplacePatcher` in [`sequence_labelling/annotator/editor.py`](sequence_labelling/annotator/editor.py) enforces exact substring uniqueness:
  ```python
  exact_count = current_text.count(search_str)
  if exact_count > 1:
      # REJECT: Ambiguous match; editor must supply additional surrounding context
  ```
* This prevents misaligned edits when repeated phrases (e.g. identical choice labels `A. 1`) appear across multiple questions.

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
>      - **Required Alternative**: Use `--limit 1` / smoke tests, mocked passes, or fast deterministic evaluation (`DeterministicParser` / `--no-llm` which finishes in milliseconds).
>    - Full model fine-tuning and massive benchmark inference runs are strictly reserved for GPU accelerators (T4 / V100 / A100 / H100).

---

## 6. Default Transformer Backbone: `aisingapore/SEA-LION-ModernBERT-300M`

> [!IMPORTANT]
> **The canonical default transformer backbone for sequence labelling is `aisingapore/SEA-LION-ModernBERT-300M`.**
> 
> (Upgraded from legacy `jhu-clsp/mmBERT-base`).
> 
> - **Architecture & Rationale**:
>   - ModernBERT architecture with up to 8192 token context window, Unpad + FlashAttention/PyTorch Native SDPA, and Rotary Position Embeddings (RoPE).
>   - Significantly superior tokenization compression and preservation for Vietnamese examination delimiters (`## Câu`, choice markers `A.`, `B.`) compared to older multilingual models.
>   - Sub-millisecond sequence labelling latency budget.
> - **Enhanced Classification Head (`model/module/head.py`)**:
>   - `SigmoidWeightedLayerPooling`: 4-layer sigmoid-gated layer pooling replacing competing softmax.
>   - 1D Depthwise Convolutional Refinement (`kernel_size=7`) capturing local boundary n-grams.
>   - Tightened Multi-Sample Dropout (MSD): `[0.05, 0.10, 0.15, 0.20, 0.25]`.
>   - `per_branch_focal_loss: true`: Focal loss computed on individual dropout branches prior to ensemble averaging.
> - **Batching & Efficiency (`model/train.py`)**:
>   - `LengthGroupedBatchSampler`: Buckets multi-scale windows (`512`, `768`, `1024`, `2048`) by length (`// 64`) to eliminate wasted padding computation on GPU while preserving SGD randomness across epochs.
> - **Pretokenization & Parity (`tools/dataset/build_training_dataset.py`)**:
>   - `attach_pretokenization` maps character spans directly to raw-text subword BPE tokens with exact offset mappings, ensuring 100% token parity between training and inference runtime (`predict.py`).
> - **Configuration Invariant**:
>   - All default configs (`configs/train_config.yaml`), training scripts (`model/train.py`), and model loaders must target `aisingapore/SEA-LION-ModernBERT-300M`.

---

## 7. Project Architecture & Directory Layout

### 7.1 Architecture Overview

The system operates as an end-to-end pipeline designed to ingest unstructured Vietnamese examination documents (PDFs/images), produce 100% character-preserved XML annotations, audit and surgically repair defects, compile tokenized training splits, and train/serve sub-millisecond neural sequence labelling models:

```
[Raw Exam PDFs/Images]
         │
         ▼  (vsl ocr / tools/ocr/run_agy_batch_ocr.py)
[Standardized Page-Aware Markdown with OCR & Exam Metadata (Subject, Grade, Year)]
         │
         ├──────────────────────────────────────────────┐
         ▼                                              ▼  (vsl metadata)
[Stage 1: Document Cartographer] (vsl cartography)     [Dataset Metadata Catalog]
   │ Emits Top-Down Spatial Manifest: Sections,          (Subject, Grade, Year, Tiers)
   │ Stimuli [p_start, p_end], Questions Q(1)[1,1], Q(3)[1,2]
   │ Computes clean_cut_seams & terminal cuts (max_manifest_page)
         │
         ▼  (sequence_labelling/parser/long_parser/greedy_chunker.py)
[Stage 2: Cartography-Locked Chunks] (Target ~4K, Max ~6.5K Tokens, Snapped to Clean Seams)
         │
         ├──► [Deterministic Parser] (vsl audit --no-llm / deterministic_parser.py) ──► Fast Path (<0.5ms)
         │
         ▼  (vsl parse / sequence_labelling/parser/long_parser/parser_agent_worker.py)
[Stage 3: Parallel LLM Parser Workers] (Gemini 3.8 Flash High / GPT-6 Luna)
   │ Surgically annotates chunks with character-exact XML tags; emits self-closing <stimulus> anchors
         │
         ▼  (sequence_labelling/parser/long_parser/sequence_reconciler.py & source_merger/)
[Stage 4: Reconciled & Merged Exam Annotations (merged.xml, merged.json)]
   │ Enforces stimulus multi-question rule globally (≥2 questions = <stimulus>, 1 question = fold to stem)
   │ Strictly bounds preceding stimuli by intervening <section> dividers
         │
         ▼  (vsl audit / sequence_labelling/annotator/reviewer.py)
[Stage 5: Cartography-Audited Reviewer] ──► Score [0-100] & Routing (PASS / NEEDS_REVISION / DISCARD)
   │ Audits 100% recall against Cartographer Manifest: flags any missing Q or severed stimulus
         │
         ▼  (vsl repair / sequence_labelling/annotator/revision_loop.py)
[Stage 6: Surgical Editor Loop] (Isolated branch repair in data/sequence_labelling_annotated_branch/)
   │ Unambiguous Search/Replace patcher (exact_count == 1); max 2 rounds; plan-locked cache invalidation
         │
         ▼  (vsl branch merge / tools/annotation/merge_branch_to_annotated.py)
[Canonical Verified Dataset (data/sequence_labelling_annotated/)]
         │
         ▼  (vsl dataset build / tools/dataset/build_training_dataset.py)
[Tokenized BIO Splits (train_spans, train_bio_chunks, train_generative) / Hugging Face Hub]
   │ Direct Raw-Text Subword BPE Pretokenization; Paired Whitespace Augmentation; Hard Negatives
         │
         ▼  (seqlabel-train / model/train.py & seqlabel-predict / model/inference/predict.py)
[ModernBERT Token Classification Head + LengthGroupedBatchSampler + Constrained Viterbi Decoding]
```

---

### 7.2 Directory Structure & Subsystem Breakdown

```
vietnamese-sequence-labelling-v2/
├── AGENTS.md                                # Core architecture invariants, contracts, and system directory map
├── README.md                                # User and developer operational handbook
├── pyproject.toml                           # uv package manager specifications, dependencies, and CLI entrypoints
│
├── configs/                                 # Centralized pipeline configurations
│   ├── augmentation_config.yaml             # Paired real whitespace, synthetic layout, and hard-negative injection policies
│   ├── dagshub_config.yaml                  # Remote MLflow experiment tracking credentials & settings
│   ├── parser_config.yaml                   # Model endpoints, providers, chunking budgets, and agent prompts
│   └── train_config.yaml                    # Hyperparameters for ModernBERT training, batch size, and LR
│
├── sequence_labelling/                      # Core production sequence labelling library
│   ├── annotator/                           # OCR conversion, XML auditing, review, and surgical editing
│   │   ├── agy_pdf_ocr.py                   # Specialized Google Antigravity PDF vision OCR converter
│   │   ├── anchor_helper.py                 # Anchor extraction and fuzzy boundary resolver
│   │   ├── annotate_ocr.py                  # Core XML tag extraction, span parsers, and CoNLL export
│   │   ├── editor.py                        # Surgical EditorAgent using precise search-and-replace XML repairs
│   │   ├── figure_detector.py               # RF-DETR figure and diagram crop detector
│   │   ├── ocr_document.py                  # Document-level OCR text and layout representation
│   │   ├── pdf_converter.py                 # Multi-engine PDF/image vision OCR converter
│   │   ├── prompt_annotator.md              # Canonical LLM system prompt for sequence labelling
│   │   ├── reviewer.py                      # AnnotationReviewerAgent, composite scoring, and routing engine
│   │   ├── revision_loop.py                 # Multi-turn isolated reviewer-editor remediation controller
│   │   ├── xml_checker.py                   # Low-level structural XML validator and defect classifier
│   │   └── xml_cleaner.py                   # Deterministic XML sanitizer and repair normalizer
│   │
│   ├── cli.py                               # Unified `vsl` CLI with native TMUX orchestration & live monitoring
│   ├── config.py                            # Runtime environment and configuration resolver
│   ├── api.py                               # FastAPI background job service and OpenAI-compatible endpoints
│   │
│   ├── metadata/                            # Examination metadata parsing & extraction
│   │   ├── extractor.py                     # Heuristic & LLM subject, grade, year, and tier classifier
│   │   └── schema.py                        # Pydantic schemas for document and exam metadata
│   │
│   ├── parser/                              # Parsing engines and reconciliation algorithms
│   │   ├── deterministic_parser.py          # Pure regex/interval parser with heading support (<0.5ms)
│   │   ├── parser.py                        # Parser pipeline interface
│   │   ├── ocr_logger.py                    # JSON/terminal logging for parsing passes
│   │   └── long_parser/                     # Long-context multi-chunk parser framework
│   │       ├── anchored_xml_llm_parser.py   # Self-closing stimulus anchor tag parsing and reconciliation
│   │       ├── document_cartographer.py     # Top-down topological cartographer emitting page-span manifests
│   │       ├── greedy_chunker.py            # Cartography-locked page chunking avoiding cross-question splits
│   │       ├── linking_agent.py             # Graph resolver mapping orphan questions to stimuli
│   │       ├── parser_agent_worker.py       # Worker wrapping LLM calls for XML annotation extraction
│   │       ├── pipeline.py                  # Long-context parser pipeline orchestrator
│   │       ├── sequence_reconciler.py       # Inter-chunk state stack and boundary seam reconciler
│   │       ├── validator_agent.py           # Pre-merger semantic and structural chunk validator
│   │       └── source_merger/               # Character-exact chunk stitching enforcing zero drift
│   │           ├── merger.py                # Source-driven sequence merger algorithm
│   │           └── serializer_validator.py  # Section-bounded stimulus serializer & validation
│   │
│   └── llm/                                 # LLM client abstractions and telemetry
│       ├── agy_adapter.py                   # Antigravity CLI subprocess bridge with session caching
│       ├── agy_rate_limiter.py              # Rolling usage limiter, budget ceiling, and 5h-cap recovery
│       ├── deepseek_client.py               # OpenAI-compatible API client with exponential backoff & retry
│       ├── llm_logger.py                    # Structured logging for model inputs and completions
│       └── token_tracker.py                 # Per-session input/output token and cost ledger
│
├── model/                                   # Neural sequence labelling model training and inference
│   ├── module/                              # Neural architecture modules
│   │   ├── dataset.py                       # Sliding-window BIO dataset with pretokenization support
│   │   └── head.py                          # EnhancedBertForTokenClassification (Sigmoid pooling, Conv1D, MSD)
│   │
│   ├── inference/                           # Inference runtime and post-processors
│   │   ├── predict.py                       # Sliding-window inference with subword aggregation
│   │   ├── predict_folder.py                # Batch directory inference evaluator
│   │   └── viterbi.py                       # C-accelerated Constrained Viterbi decoding enforcing BIO syntax
│   │
│   ├── export.py                            # Model checkpoint and ONNX export utility
│   └── train.py                             # PyTorch training with LengthGroupedBatchSampler, FP16/BF16, & F1
│
├── synthetic_exam_generator/                # Synthetic exam generation and data augmentation engine
│   ├── adversarial_theory_generator.py      # Adversarial theory question synthesizer
│   ├── cli.py                               # CLI entrypoint for synthetic exam generation (`synthetic-exam-gen`)
│   ├── config.py                            # Synthetic generator configuration
│   ├── curriculum.py                        # Curriculum progression generator across difficulty tiers
│   ├── deepseek_client.py                   # LLM client for synthetic item generation
│   ├── exam_compiler.py                     # Compiles raw templates and variables into coherent exams
│   ├── generator.py                         # High-level generation coordinator
│   ├── hard_negative_injector.py            # Adversarial expository prose injector (background 'O' tokens)
│   ├── latex_utils.py                       # LaTeX formula masking and delimiter jitter utilities
│   ├── parser.py                            # Synthetic template AST parser
│   ├── reconstructor.py                     # Layout permutation, typo injection, and style variation
│   ├── template_bank.py                     # High-school math question templates and distractors
│   └── token_tracker.py                     # Token tracking for synthetic generator sessions
│
├── tools/                                   # Categorized operational CLI tools and batch workflows
│   ├── ocr/                                 # Document ingestion and Vision OCR batch tooling
│   │   ├── check_batch_status.py            # Real-time terminal status inspector for batch OCR runs
│   │   ├── classify_raw_pdfs.py             # PDF classifier (scanned vs digital vs hybrid)
│   │   ├── curate_500_diverse_pdfs.py       # Stratified diversity sampler and deduplicator for exams
│   │   └── run_agy_batch_ocr.py             # Multi-worker AGY OCR orchestrator with auto-cap recovery
│   │
│   ├── cartography/                         # Topological spatial mapping tools
│   │   └── generate_dataset_cartography.py  # Batch DocumentCartographer manifest runner
│   │
│   ├── metadata/                            # Exam metadata catalog generation
│   │   └── generate_dataset_metadata.py     # Batch metadata extractor runner
│   │
│   ├── annotation/                          # Ground-truth sequence labelling annotation pipeline
│   │   ├── annotate_sequence_labelling_dataset.py # Canonical parser annotation pipeline runner
│   │   ├── check_annotation_batch_status.py # Real-time dashboard for batch annotation progress
│   │   ├── check_cartography_reparse_status.py # Status checker for cartography-based reparse runs
│   │   ├── merge_branch_to_annotated.py     # Safely merges approved branch candidates to canonical data
│   │   ├── remerge_annotated_dataset.py     # Batch tool rebuilding merged XMLs from cached chunk outputs
│   │   ├── reparse_cartography_documents.py # Cartography-locked reparsing with chunk cache invalidation
│   │   ├── reparse_discarded_chunks.py      # Reparses specific discarded chunks
│   │   ├── run_agy_batch_annotation.py      # Resilient multi-worker AGY batch annotation pipeline
│   │   └── strip_annotation_tags.py         # Deterministic stripper validating raw text identity
│   │
│   ├── review/                              # Quality auditing, surgical editing, and remediation
│   │   ├── edit_review_set.py               # Surgical Editor/Reviewer revision loop on branch files
│   │   ├── reparse_and_review.py            # Unified end-to-end reparse, audit, and comparison pipeline
│   │   ├── reparse_review_set.py            # Reparses failing documents directly from raw Markdown
│   │   ├── run_reparse_review_edit_chunks.py # Chunk-level reparse, review, and edit pipeline
│   │   └── run_review_and_edit_500.sh       # Production orchestration shell script for review & repair
│   │
│   ├── audit/                               # Specialized audit tools
│   │   └── audit_chunk_seam_severing.py     # Verifies whether chunks severed active questions or stimuli
│   │
│   ├── evaluation/                          # Model & dataset benchmark evaluation suites
│   │   ├── eval_adaptive_scoring_experiment.py # Experimental diminishing-penalty scoring engine
│   │   ├── eval_tokens_and_det_f1.py        # Token counts and Deterministic Parser F1 evaluation
│   │   ├── evaluate_500_dataset.py          # Invariant, DET, and neural evaluation on 500-doc dataset
│   │   ├── evaluate_model.py                # Comprehensive 63 Gold Benchmark document evaluation
│   │   ├── evaluate_trained_model_500.py    # Evaluation of trained Hugging Face checkpoints
│   │   └── select_gold_benchmark.py         # Stratified token-budgeted gold benchmark set selector
│   │
│   ├── dataset/                             # Dataset compilation and Hugging Face Hub publishing
│   │   ├── build_training_dataset.py        # Compiles BIO sequence chunks with direct subword alignment
│   │   ├── upload_dataset_to_hf.py          # Uploads versioned multi-config dataset splits to HF
│   │   └── upload_model_to_hf.py            # Uploads model checkpoints, weights, and model cards to HF
│   │
│   ├── monitoring/                          # Cost, rate limit, and telemetry observability
│   │   ├── safe_ccusage.py                  # Lock-safe SQLite snapshot bridge for Antigravity telemetry
│   │   └── token_monitor.py                 # Real-time token usage, latency, and cost terminal dashboard
│   │
│   ├── visualize/                           # Parsing and chunking diagnostic visualizers
│   │   ├── visualize_chunker.py             # Visualizes greedy oversize chunking and passage locks
│   │   ├── visualize_linker.py              # Visualizes stimulus-question graph resolution
│   │   ├── visualize_merger.py              # Visualizes chunk reconciliation and seam merging
│   │   └── visualize_parser.py              # Visualizes parser worker chunk extraction
│   │
│   └── experiments/                         # Research prototypes and experimental OCR architectures
│       ├── chunk_extractor.py               # PyMuPDF digital text chunk and formula extractor
│       ├── digital_chunk_ocr.py             # Hybrid vision/digital hybrid OCR pipeline
│       ├── dynamic_mosaic_packer.py         # 2D bin-packing mosaic compositor for visual crops
│       ├── hybrid_digital_ocr.py            # Dual-path digital/raster exam OCR engine
│       ├── run_chunk_injection_benchmark.py # Benchmark for digital chunk injection performance
│       └── run_experiment_and_benchmark.py  # End-to-end OCR experiment suite runner
│
├── tests/                                   # Automated test suite (289+ passing unit & integration tests)
│   ├── unit/                                # Unit tests across all core subsystems
│   │   ├── ocr/                             # Tests for reviewer, editor, revision loop, checker, cleaner
│   │   ├── synthetic_exam_generator/        # Tests for curriculum, compiler, template bank, reconstructor
│   │   ├── test_cli.py                      # Tests for unified `vsl` CLI
│   │   ├── test_pipeline_regressions.py     # Regression tests for heading markers, cloze items, etc.
│   │   └── test_*.py                        # Tests for adapters, head, viterbi, strip_tags, etc.
│   ├── test_merger/                         # Comprehensive source merger and chunk provenance tests
│   ├── regex_question_parser/               # Deterministic regex parsing benchmarks
│   └── ocr_benchmarks/                      # Accuracy benchmarks for OCR converters
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
    ├── sequence_labelling_annotated_branch/ # Disposable staging branch for surgical editor candidate repairs
    ├── document_cartography/                # Topological Cartographer manifests
    ├── dataset_training_splits/             # Compiled tokenized BIO and span splits
    └── synthetic_exams/                     # Generated curriculum and augmented synthetic exams
```

---

## 8. Document Cartography & Topological Boundary Invariants

> [!IMPORTANT]
> **Stage 1 Document Cartography Invariant**:
> 
> 1. **Clean-Cut Seams Enforced**: Chunker cuts must never split an active question (`Q(3) [1, 2]`) or a stimulus group (`[Stimulus ... Last Question]`) across chunk boundaries. All chunk divisions must snap strictly to `clean_cut_seams` determined by `DocumentCartographer`.
> 2. **Top-Down Sequential Stream**: The Cartographer scans the document top-to-bottom in physical reading order, outputting objects with exact page spans `[start_page, end_page]`.
> 3. **Terminal Exam Boundary Cut (`max_manifest_page`)**:
>    - When `max_manifest_page` is reached, `greedy_chunker.py` forces a clean terminal cut to isolate the exam content from trailing answer grids, solution keys, or appendices.
> 4. **Global Multi-Question Stimulus Resolution**:
>    - Chunk workers in Stage 3 emit candidate `<stimulus id="...">` tags without guessing about future unseen pages.
>    - The global Source Merger (Stage 4) verifies question counts across the entire document:
>      - Count $\ge 2$: Preserved as `<stimulus id="...">`.
>      - Count $== 1$: Safely collapsed into that single question's `<stem>`.
> 5. **Deterministic Recall Auditing**: The Reviewer (Stage 5) uses the Cartographer's manifest as an authoritative pre-scan ground truth to verify 100% question recall. Any question present in the manifest but missing from the parsed XML is flagged immediately.
> 6. **Plan-Locked Chunk Cache Invalidation**:
>    - In [`reparse_cartography_documents.py`](tools/annotation/reparse_cartography_documents.py), cached chunk outputs are only reused if their `page_ids` and `raw_chunk_text` match the active Cartographer plan exactly. If chunk boundaries change, the cache is automatically invalidated and reparsed.

---

## 9. Isolated Staging Invariant: All Edits and Candidate Reparses Route to Branch

> [!IMPORTANT]
> **All surgical edits, defect remediation loops, and candidate re-annotations MUST ALWAYS be routed to `data/sequence_labelling_annotated_branch/`.**
>
> Under NO circumstances should any autonomous agent, tool, or reparse script overwrite canonical `data/sequence_labelling_annotated/` in-place without an isolated branch staging and verification step.
>
> ### Core Invariants:
> 1. **Isolated Branch Staging**:
>    - All `EditorAgent` surgical repairs, multi-round reviewer loops, and chunk/document re-annotations MUST output directly to `data/sequence_labelling_annotated_branch/<doc_id>/`.
>    - Canonical `data/sequence_labelling_annotated/` remains an immutable production baseline during repair and reparse operations.
> 2. **Audited Merge Protocol**:
>    - Repaired or re-annotated files can only be merged into canonical `data/sequence_labelling_annotated/` using the verified merge gate:
>      ```bash
>      uv run vsl branch merge --branch-dir data/sequence_labelling_annotated_branch --target-dir data/sequence_labelling_annotated
>      ```
>    - The merge gate enforces:
>      - Strict character fidelity verification (`strip_tags(candidate) == strip_tags(original)`).
>      - Complete XML syntax validation (`is_valid_xml`).
>      - Minimum score threshold verification (`min_score=60.0` or higher).
>      - Update of audit ledgers and progress trackers (`review_report.json`).

---

## 10. Anti-Sprawl & Tooling Reuse Invariant: Prioritize Unified `vsl` CLI

> [!CAUTION]
> **NEVER create a new script without verifying existing tools first.**
> 
> A major recurring failure mode is that agents create new standalone scripts (`run_*_tmux.sh`, `reparse_*_shard*.py`, `test_*_batch.py`) instead of reusing or extending existing tools, placing temporary scripts at the same directory level as canonical production tools.
>
> ### Strict Rules for All Agents:
> 1. **Check the Unified `vsl` CLI First**:
>    - Before writing ANY script to run OCR, parsing, review, repair, or evaluation, run `uv run vsl --help`.
>    - Standard commands (`vsl ocr`, `vsl parse`, `vsl audit`, `vsl repair`, `vsl branch merge`, `vsl dataset build`, `vsl evaluate`, `vsl status`, `vsl tmux`) cover the entire lifecycle with native multi-threading and background session handling.
> 2. **Parameterize Instead of Duplicating**:
>    - NEVER create a new script simply to change a model name, filter by doc ID, alter concurrency, or change shard count.
>    - Use existing CLI flags: `--concurrency`, `--model`, `--provider`, `--filter-decision`, `--pattern`, `--manifest`, `--limit`, `--doc-id`.
> 3. **Native Detached TMUX Execution**:
>    - To run long-running batch jobs in the background, append `--tmux [SESSION_NAME]` to any `vsl` command (e.g. `uv run vsl repair --tmux vsl-repair-batch`).
>    - Manage background runs using `vsl tmux list`, `vsl tmux attach`, `vsl tmux logs`, or `vsl tmux kill`.
>    - NEVER create hardcoded `.sh` wrapper scripts for tmux.
> 4. **Mandatory Scratch Sandbox for Ad-Hoc Scripts**:
>    - If a temporary probe, one-off diagnostic, or ad-hoc calculation script is strictly necessary, it MUST be saved in:
>      `<appDataDir>/brain/<conversation-id>/scratch/` or `.scratch/` (gitignored).
>    - It is **STRICTLY PROHIBITED** to place temporary scripts directly into `tools/`, `scripts/`, or the repository root.
