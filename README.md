# Vietnamese Sequence Labelling v2

Standalone OCR, annotation, parsing, review, dataset, and visualization project used by Azozo.

## Run

```bash
PYTHONPATH=. uv run uvicorn sequence_labelling.api:app --host 0.0.0.0 --port 8100
```

The worker exposes `/v1/jobs` for exam OCR/annotation and `/v1/answer-keys/map` for answer-key extraction. Dataset and review tools run locally against `data/`; generated output belongs in `artifacts/`.

## Reviewer → Editor loop

The revision loop uses a confirmation gate:

1. The reviewer runs deterministic checks and sends every candidate finding to the semantic reviewer with a stable `issue_id`.
2. The semantic reviewer marks each finding `is_true_positive: true` or `false`. Only true positives, plus newly identified semantic errors, are copied to `ReviewReport.confirmed_issues`.
3. The editor receives `confirmed_issues`, the current annotated XML, and the raw OCR source. It edits the XML by returning surgical `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` blocks; the patcher applies those blocks to the current XML and the cleaner normalizes the result.
4. The reviewer receives only the raw source and patched XML, never the editor conversation, and confirms the next round's errors independently.

The editor may also return an `ISSUE_ASSESSMENTS` JSON block to mark a supplied confirmed issue as a possible false positive. That claim is recorded and removed from later editor prompts, but it does not by itself turn an unresolved audit into `PASS`; a reviewer or human must adjudicate the disagreement.

The large `data/` and `models/` directories are local runtime assets and are intentionally ignored by Git.
