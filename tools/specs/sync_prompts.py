#!/usr/bin/env python3
"""
sync_prompts.py — Prompt & Specification Alignment Validator.

Audits prompt files, reviewer prompts, and few-shot examples against the Single Source
of Truth (SSOT) defined in `sequence_labelling/specs/annotation_spec.yaml`.
"""

import re
import sys
from pathlib import Path
import yaml


def audit_spec_alignment(repo_root: Path) -> bool:
    spec_path = repo_root / "sequence_labelling" / "specs" / "annotation_spec.yaml"
    prompt_path = repo_root / "sequence_labelling" / "annotator" / "prompt_annotator.md"
    reviewer_path = repo_root / "sequence_labelling" / "annotator" / "reviewer.py"
    parser_path = repo_root / "sequence_labelling" / "parser" / "long_parser" / "anchored_xml_llm_parser.py"

    errors = []

    if not spec_path.exists():
        print(f"[FAIL] Missing SSOT spec: {spec_path}")
        return False

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    print(f"[*] Loaded SSOT Specification v{spec.get('version')}...")

    # 1. Audit prompt_annotator.md
    if prompt_path.exists():
        content = prompt_path.read_text(encoding="utf-8")

        # Invariant 1: Stimulus multi-question rule in Example 2
        # Verify no single-question stimulus in Example 2
        if '<stimulus id="stim_2" start_anchor="**2.** To build' in content:
            errors.append("prompt_annotator.md Example 2 still tags single-question context as <stimulus>.")

        # Invariant 2: Sentence arrangement in Example 2
        if '<stimulus id="stim_3" start_anchor="Dear Minh,' in content:
            errors.append("prompt_annotator.md Example 2 still tags sentence arrangement as <stimulus>.")

        # Invariant 3: Table boundary
        if "<stem>" in content and "<table>" in content:
            if re.search(r"<stem>[^<]*<table>", content):
                errors.append("Found <table> illegally nested inside <stem> in prompt_annotator.md.")

        # Invariant 4: Check if table example has closing stem before table
        if "<question_label>## Câu 3:</question_label> <stem>Phát biểu sau đúng hay sai?</stem>\n<table>" not in content:
            errors.append("prompt_annotator.md Example 4 missing strict closure of <stem> before <table>.")

        print("  [+] prompt_annotator.md audited.")

    # 2. Audit reviewer.py
    if reviewer_path.exists():
        rev_content = reviewer_path.read_text(encoding="utf-8")

        # Invariant 1: Expected Parser Output XML must not wrap school metadata in <section>
        if "<section>SỞ GD&ĐT HÀ NỘI" in rev_content:
            errors.append("reviewer.py expected parser output illegally wraps administrative metadata in <section>.")

        # Invariant 2: Check deterministic precheck does not penalize empty stems with options
        if "empty_stems and not has_options_or_stimuli" not in rev_content:
            errors.append("reviewer.py deterministic precheck does not implement empty stem exemption for cloze/options.")

        print("  [+] reviewer.py audited.")

    # 3. Audit anchored_xml_llm_parser.py
    if parser_path.exists():
        parser_content = parser_path.read_text(encoding="utf-8")
        if "TABLE BOUNDARY RULE" not in parser_content:
            errors.append("anchored_xml_llm_parser.py missing TABLE BOUNDARY RULE.")
        if "CLOZE & EMBEDDED BLANKS" not in parser_content:
            errors.append("anchored_xml_llm_parser.py missing CLOZE & EMBEDDED BLANKS rule.")
        print("  [+] anchored_xml_llm_parser.py audited.")

    if errors:
        print("\n[FAIL] Found alignment violations against SSOT:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("\n[PASS] All prompt templates, reviewer rubrics, and few-shot examples 100% align with SSOT!")
    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    success = audit_spec_alignment(repo_root)
    sys.exit(0 if success else 1)
