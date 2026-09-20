# Vietnamese Sequence Labelling v2

Vietnamese examination-document processing tools for OCR, sequence-labelling annotation, structural merging, deterministic auditing, LLM review, and surgical XML repair.

The project is designed for school exams, cloze tests, reading passages, multiple-choice questions, answer keys, figures, tables, and long multi-page stimuli. It preserves source text while adding structural tags such as `<question_label>`, `<stem>`, `<option_label>`, `<option_text>`, and `<stimulus>`.

## What is in the repository?

The main processing path is:

```text
PDF / image → OCR Markdown → chunking/parser/reconciliation
           → merged.xml + merged.json → deterministic audit
           → semantic review → PASS / NEEDS_REVISION / DISCARD
           → editor/reviewer repair loop for NEEDS_REVISION
```

Important components:

- `sequence_labelling/annotator/pdf_converter.py`: PDF/image OCR with optional RF-DETR figure detection.
- `sequence_labelling/annotator/annotate_ocr.py`: OCR/annotation-oriented text processing.
- `sequence_labelling/parser/long_parser/`: passage-locked chunking, parser workers, reconciliation, anchoring, and serialization.
- `sequence_labelling/annotator/reviewer.py`: deterministic auditing, LLM semantic review, scoring, and routing.
- `sequence_labelling/annotator/editor.py`: exact Search/Replace XML repair.
- `sequence_labelling/annotator/revision_loop.py`: isolated editor/reviewer rounds.
- `review_studio/`: browser-based audit and editing interface.
- `sequence_labelling/api.py`: OCR/job and OpenAI-compatible worker API.

## Requirements and setup

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Credentials for the providers/models you use
- Optional: `jinja2` for Review Studio; it is imported by the Studio but is not currently declared in `pyproject.toml`

Install the project environment:

```bash
uv sync
```

If you will run Review Studio:

```bash
uv pip install jinja2
```

Create a local `.env` file for provider credentials. Do not commit it.

```dotenv
OPENAI_API_KEY=...
# Optional provider-specific credentials:
# XAH_API_KEY=...
# LLM_API_KEY=...
# NVIDIA_API_KEY=...
# DEEPSEEK_API_KEY=...
# CMD_API_KEY=...
```

Provider endpoints, default models, token budgets, reviewer thresholds, and figure detection are configured in [`config.yaml`](config.yaml). The main data locations can be overridden with:

```bash
export SEQUENCE_LABEL_DATA_DIR=/path/to/data
export SEQUENCE_LABEL_MODEL_DIR=/path/to/models
export SEQUENCE_LABEL_ARTIFACTS_DIR=/path/to/artifacts
```

Inspect resolved defaults with:

```bash
uv run sequence-label paths
```

## Data layout

Runtime datasets are intentionally local and ignored by Git:

```text
data/
├── sequence_labelling_input_data_raw/   # source PDFs/images
├── sequence_labelling_input_data/       # OCR Markdown
├── sequence_labelling_annotated/        # canonical merged XML/JSON
├── sequence_labelling_annotated_branch/ # editor candidates/checkpoints
└── ...

artifacts/
└── ocr_logs/                            # generated reports/logs
```

An annotated document commonly contains:

```text
<document>/
├── merged.xml
├── merged.json
├── audit_report.json
└── chunks/
```

`merged.xml` is the source-preserving sequence-labelled representation. `merged.json` contains structured questions and stimuli derived from the XML. The merger validates that removing annotation tags reproduces the canonical source text.

## End-to-end batch workflow

### 1. OCR PDFs or images

```bash
uv run python tools/batch_ocr_raw_dataset.py \
  --raw-dir data/sequence_labelling_input_data_raw \
  --out-dir data/sequence_labelling_input_data \
  --provider codex \
  --concurrency 1
```

Use `--overwrite` only when intentionally regenerating existing Markdown. The OCR stage supports PDF and common image formats and emits page-aware Markdown with figure metadata where configured.

### 2. Annotate OCR Markdown

```bash
uv run python tools/annotate_sequence_labelling_dataset.py \
  --input_dir data/sequence_labelling_input_data \
  --output_dir data/sequence_labelling_annotated \
  --provider agy \
  --model gemini-3.8-flash \
  --concurrency 4
```

The tool skips documents with existing `merged.json` unless `--overwrite` is supplied. Use `--pattern`, `--limit`, and smaller token budgets for controlled experiments.

### 3. Audit the annotated dataset

```bash
uv run python tools/batch_review_annotated.py \
  --annotated-dir data/sequence_labelling_annotated \
  --raw-dir data/sequence_labelling_input_data \
  --provider codex \
  --model gpt-5.6-luna \
  --thinking high \
  --output-report artifacts/review_report.md
```

The reviewer writes a Markdown report and a JSON report beside the requested output. Existing per-document `audit_report.json` files are reused unless `--overwrite` is supplied.

For a fast deterministic-only audit:

```bash
uv run python tools/batch_review_annotated.py \
  --annotated-dir data/sequence_labelling_annotated \
  --raw-dir data/sequence_labelling_input_data \
  --no-llm
```

### 4. Repair `NEEDS_REVISION` documents

The editor/reviewer loop never overwrites the canonical annotated directory. It writes candidates and checkpoint artifacts to the branch directory.

```bash
uv run python tools/edit_review_set.py \
  --report data/review_report.json \
  --filter needs_revision \
  --concurrency 4 \
  --reuse-prerun-logs \
  --auto-save \
  --branch-dir data/sequence_labelling_annotated_branch \
  --out-report artifacts/revision_resolution_report.json
```

Safe continuation is the default:

- compatible branch checkpoints are resumed;
- documents checkpointed as `PASS` are skipped;
- unresolved candidates are reviewed from their latest branch XML;
- `--no-resume` starts from the original annotated XML;
- `--retry-unresolved` retries prior unresolved checkpoints;
- `--dry-run` performs no file writes;
- `--force-large` includes documents above the default 300 KB XML limit.

Each resumed document can contain:

```text
<branch>/<document>/
├── merged.xml
├── merged.json
├── revision_state.json
└── revisions/
    ├── initial_review.json
    └── round_001.json
```

The aggregate report is checkpointed as documents finish. To keep a long run alive after disconnecting from a shell:

```bash
tmux new-session -d -s sequence-review \
  'uv run python tools/edit_review_set.py --report data/review_report.json --filter needs_revision --concurrency 4 --reuse-prerun-logs --auto-save --branch-dir data/sequence_labelling_annotated_branch --out-report artifacts/revision_resolution_report.json 2>&1 | tee artifacts/full_revision_loop.log'

tmux attach -t sequence-review
```

### 5. Reparse failed documents

```bash
uv run python tools/reparse_review_set.py \
  --report data/review_report.json \
  --filter all_failures \
  --raw-dir data/sequence_labelling_input_data \
  --output-dir data/sequence_labelling_annotated \
  --provider agy \
  --model gemini-3.8-flash
```

For a combined reparse-and-review workflow:

```bash
uv run python tools/reparse_and_review.py \
  --report data/review_report.json \
  --filter all_failures \
  --stage all \
  --out-report artifacts/ocr_logs/reparse_and_review_report.json
```

Use `--stage reparse-only`, `--stage review-only`, `--doc-id`, `--limit`, or `--dry-run` to narrow the operation.

### 6. Re-merge annotation chunks

```bash
uv run python tools/remerge_annotated_dataset.py \
  --input data/sequence_labelling_annotated \
  --concurrency 8
```

This rebuilds merged files from retained chunk outputs and is useful after chunk-level repairs or parser changes.

## Annotation contract

Recognized structural tags include:

```xml
<section>...</section>
<stimulus id="stim_1" start_anchor="..." end_anchor="..." />
<question_label>Câu 1</question_label>
<stem>...</stem>
<option_label>A</option_label>
<option_text>...</option_text>
<explanation>...</explanation>
<figure id="fig_1" description="..." />
```

The source text must remain unchanged after annotation tags are removed. The merger and serializer enforce this invariant and reject out-of-bounds or unbalanced spans.

### Questions without explicit stems are valid

A question label does not always need a following `<stem>`. Cloze, fill-in-the-blank, and passage-indexed questions may validly begin with options:

```xml
<question_label>Câu 631</question_label>
<option_label>A.</option_label><option_text>...</option_text>
```

The parser, reviewer, editor, and deterministic auditor must preserve this form. The editor must not invent an empty or synthetic stem when the raw source contains no explicit question prompt.

### Review routing

The reviewer combines deterministic checks with semantic review. Deterministic findings are not automatically actionable: the semantic reviewer confirms true positives before they are sent to the editor. Critical and major defects remain routing gates even when the numeric score is high.

The editor uses exact Search/Replace blocks:

```text
<<<<<<< SEARCH
old XML text
=======
new XML text
>>>>>>> REPLACE
```

The reviewer is stateless across rounds and receives only the raw source plus the current candidate XML. Editor conversation history is retained only by the editor for that document’s repair rounds.

## HTTP services

### OCR and worker API

Start the worker:

```bash
PYTHONPATH=. uv run uvicorn sequence_labelling.api:app \
  --host 0.0.0.0 \
  --port 8100
```

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/jobs` | Submit a PDF/image upload or `raw_text` job |
| `GET` | `/v1/jobs` | List in-memory jobs |
| `GET` | `/v1/jobs/{job_id}` | Get job status and result |
| `DELETE` | `/v1/jobs/{job_id}` | Remove an in-memory job |
| `POST` | `/v1/answer-keys/map` | Extract and map answer keys to supplied questions |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat proxy |

Example raw-text job:

```bash
curl -X POST http://localhost:8100/v1/jobs \
  -F 'raw_text=Câu 1. Chọn đáp án đúng. A. Một B. Hai'
```

Jobs are held in process memory. They are not a durable queue and disappear when the worker restarts.

### Review Studio

Start the interactive reviewer/editor backend:

```bash
PYTHONPATH=. uv run uvicorn review_studio.review_studio:app \
  --host 0.0.0.0 \
  --port 8200
```

Open `http://localhost:8200/`. The Studio supports document browsing, deterministic audits, auto-fix preview, XML editing, and saving updated `merged.xml`/`merged.json` files. Configure alternate locations with:

```bash
export SEQUENCE_LABEL_DATA_DIR=/path/to/data
export SEQUENCE_LABEL_REVIEW_REPORT=/path/to/review_report.json
```

The Studio save endpoint writes to the selected document path, so use it against a branch or disposable copy when preserving canonical annotations is required.

## Configuration and providers

`config.yaml` defines provider base URLs and default model assignments for OCR, parser/annotation, linker, answer-key mapping, reviewer, editor, figure detection, and chunking.

The provider adapter in `sequence_labelling/llm/deepseek_client.py` routes requests to configured OpenAI-compatible services, Codex, or the local `agy` adapter. Keep API keys in `.env` or the process environment; never place secrets in `config.yaml` or committed reports.

## Testing

Focused editor/reviewer tests:

```bash
UV_CACHE_DIR=/tmp/vsl-uv-cache uv run pytest \
  tests/unit/ocr/test_revision_loop.py \
  tests/unit/ocr/test_editor_agent.py -q
```

Configured suite:

```bash
UV_CACHE_DIR=/tmp/vsl-uv-cache uv run pytest -q
```

The tests cover OCR helpers, XML cleaning/checking, deterministic auditing, parser/merger invariants, review payloads, editor patching, revision loops, and provider adapters.

## Troubleshooting

### `uv` cannot create its cache

Point the cache at a writable location:

```bash
UV_CACHE_DIR=/tmp/vsl-uv-cache uv run ...
```

### Review Studio fails with a Jinja2 import error

Install the currently undeclared optional dependency:

```bash
uv pip install jinja2
```

### A document is skipped as oversized

The editor/reviewer loop defaults to 300 KB per XML document. Inspect the document first, then use `--force-large` when the larger request is intentional.

### A revision loop is interrupted

Rerun the same command. Compatible `revision_state.json` checkpoints and branch candidates are used automatically. Inspect `revision_resolution_report.json` and each document’s `revisions/` directory before retrying unresolved documents.

### A provider request fails

Check the selected provider in `config.yaml`, the matching environment variable, the model name, and the provider base URL. Start with `--limit 1` or `--doc-id ...` before launching a large batch.

## Repository rules

- Treat `data/`, model files, artifacts, and logs as runtime state, not source-controlled code.
- Preserve the canonical annotated directory during automated repair; use a branch directory for candidates.
- Do not infer missing stems for valid cloze or passage-indexed questions.
- Do not bypass major or critical review defects because a composite score is high.
- Keep source-fidelity and structural validation enabled when changing parser, merger, reviewer, or editor behavior.

## License

See [`LICENSE`](LICENSE).
