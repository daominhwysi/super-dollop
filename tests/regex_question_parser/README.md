# Regex question parser experiment

This experiment treats regex as a high-recall candidate generator. It does not
attempt to infer sections, stimuli, or whether a numbered sentence is
semantically an exam question.

## Algorithm

1. Find line-start question labels such as `12.`, `12)`, `Question 12:`,
   `Câu 12.`, and Markdown variants.
2. Bound each candidate by the next detected label while retaining source
   character offsets.
3. Extract choices written as `(A)`, `A.`, `A)`, or `A:`, including choices
   placed on the same line.
4. Score structural evidence: explicit question prefixes, question marks,
   interrogatives, prompt verbs, choices, blanks, and sequential numbering.
5. Return candidates above the threshold as `questions` and preserve everything
   else as `rejected`.

The default score threshold is conservative enough to reject ordinary numbered
instructions. For an LLM validation stage, lower `min_score` to `1` or `2` and
send both lists with `raw_text`, offsets, signals, and score.

## Run against OCR text

```bash
uv run python tests/regex_question_parser/question_parser.py input.txt
```

Use `--min-score 2` for a higher-recall candidate set:

```bash
uv run python tests/regex_question_parser/question_parser.py \
  --min-score 2 \
  input.txt
```

The parser intentionally leaves section and stimulus detection for a later
document-level pass. Removing rejected blocks before that pass would discard
headings, directions, and passages needed to establish question context.
