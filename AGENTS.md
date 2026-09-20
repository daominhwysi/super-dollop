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
