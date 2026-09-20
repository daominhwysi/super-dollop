"""High-recall regex question candidate extraction experiment.

This module deliberately produces candidates rather than claiming to understand
the document. A later semantic validator can inspect both accepted and rejected
candidates, together with their source spans and scoring signals.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


QUESTION_START_RE = re.compile(
    r"""
    ^[ \t]*
    (?:>\s*)?                         # Markdown quote / OCR decoration
    (?:\*{1,2}\s*)?                   # Optional Markdown emphasis
    (?:(?P<prefix>questions?|q|câu|cau)\s*)?
    (?P<number>\d{1,4})
    \s*(?:[.)]|:|-)\s*
    (?:\*{1,2}\s*)?
    (?P<first_line>[^\n]*)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

CHOICE_RE = re.compile(
    r"""
    (?<!\S)
    (?:
        \((?P<parenthesized>[A-Ha-h])\)
        |
        (?P<plain>[A-Ha-h])\s*[.):]
    )
    \s+
    """,
    re.VERBOSE,
)

INTERROGATIVE_RE = re.compile(
    r"^(?:what|when|where|which|who|whom|whose|why|how|is|are|do|does|did|"
    r"can|could|would|should|will|hãy|tai sao|tại sao|đâu|nào)\b",
    re.IGNORECASE,
)

PROMPT_VERB_RE = re.compile(
    r"^(?:choose|select|identify|explain|describe|calculate|compute|prove|"
    r"show|find|determine|complete|compare|discuss|mark|chọn|xác định|"
    r"giải|tính|chứng minh|trình bày)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Choice:
    label: str
    text: str


@dataclass
class QuestionCandidate:
    number: int
    label: str
    stem: str
    choices: list[Choice]
    raw_text: str
    start_char: int
    end_char: int
    score: int = 1
    signals: list[str] = field(default_factory=lambda: ["numbered-block"])
    accepted: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ParseResult:
    questions: list[QuestionCandidate]
    rejected: list[QuestionCandidate]

    def to_dict(self) -> dict[str, object]:
        return {
            "questions": [candidate.to_dict() for candidate in self.questions],
            "rejected": [candidate.to_dict() for candidate in self.rejected],
        }


def _clean_fragment(value: str) -> str:
    lines = []
    for line in value.splitlines():
        cleaned = re.sub(r"^[ \t]*>\s?", "", line)
        lines.append(cleaned.strip())
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()


def _extract_stem_and_choices(content: str) -> tuple[str, list[Choice]]:
    matches = list(CHOICE_RE.finditer(content))
    if not matches:
        return _clean_fragment(content), []

    stem = _clean_fragment(content[: matches[0].start()])
    choices = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        label = match.group("parenthesized") or match.group("plain")
        choices.append(
            Choice(
                label=label.upper(),
                text=_clean_fragment(content[match.end() : end]),
            )
        )
    return stem, choices


def _score_candidate(candidate: QuestionCandidate, explicit_prefix: bool) -> None:
    stem = candidate.stem.strip()

    if explicit_prefix:
        candidate.score += 2
        candidate.signals.append("explicit-question-prefix")
    if "?" in stem:
        candidate.score += 2
        candidate.signals.append("question-mark")
    if INTERROGATIVE_RE.match(stem):
        candidate.score += 1
        candidate.signals.append("interrogative")
    if PROMPT_VERB_RE.match(stem):
        candidate.score += 1
        candidate.signals.append("question-prompt-verb")
    if len(candidate.choices) >= 2:
        candidate.score += 2
        candidate.signals.append("multiple-choices")
    if len(candidate.choices) >= 3:
        candidate.score += 1
        candidate.signals.append("three-or-more-choices")
    if re.search(r"(?:_{3,}|-{3,}|\.{3,})", stem):
        candidate.score += 1
        candidate.signals.append("blank")


def _add_sequence_signals(candidates: Sequence[QuestionCandidate]) -> None:
    for index, candidate in enumerate(candidates):
        previous_is_sequential = (
            index > 0 and candidates[index - 1].number + 1 == candidate.number
        )
        next_is_sequential = (
            index + 1 < len(candidates)
            and candidate.number + 1 == candidates[index + 1].number
        )
        if previous_is_sequential or next_is_sequential:
            candidate.score += 1
            candidate.signals.append("sequential-number")


def parse_question_candidates(text: str, min_score: int = 3) -> ParseResult:
    """Extract and score numbered question candidates from arbitrary exam text.

    ``min_score`` controls the regex-only filter. Use a lower value to send more
    ambiguous numbered blocks to an LLM validator.
    """

    starts = list(QUESTION_START_RE.finditer(text))
    candidates = []

    for index, match in enumerate(starts):
        end_char = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        continuation = text[match.end() : end_char]
        content = f"{match.group('first_line')}\n{continuation}"
        stem, choices = _extract_stem_and_choices(content)
        prefix = match.group("prefix")
        label = f"{prefix} {match.group('number')}" if prefix else match.group("number")
        candidate = QuestionCandidate(
            number=int(match.group("number")),
            label=label.strip(),
            stem=stem,
            choices=choices,
            raw_text=text[match.start() : end_char].strip(),
            start_char=match.start(),
            end_char=end_char,
        )
        _score_candidate(candidate, explicit_prefix=prefix is not None)
        candidates.append(candidate)

    _add_sequence_signals(candidates)
    for candidate in candidates:
        candidate.accepted = candidate.score >= min_score

    return ParseResult(
        questions=[candidate for candidate in candidates if candidate.accepted],
        rejected=[candidate for candidate in candidates if not candidate.accepted],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 OCR text file")
    parser.add_argument("--min-score", type=int, default=3)
    args = parser.parse_args()

    result = parse_question_candidates(
        args.input.read_text(encoding="utf-8"),
        min_score=args.min_score,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
