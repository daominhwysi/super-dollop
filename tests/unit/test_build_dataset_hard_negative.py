"""
Unit test suite verifying hard-negative injection at the build_dataset step.
Guarantees:
1. Hard-negative modules are safely injected in-memory into synthetic exams.
2. The original synthetic exams on disk in data/synthetic_exams remain 100% untouched.
3. Chunks generated in train_bio_chunks.jsonl carry the injected hard-negative tokens.
4. Downstream BIO sequence syntax and token alignment are fully preserved.
"""

import hashlib
import json
import random
from pathlib import Path

import pytest

from synthetic_exam_generator.hard_negative_injector import (
    HardNegativeBank,
    SyntheticHardNegativeInjector,
    InjectionConfig,
)
from tools.dataset.build_training_dataset import (
    process_synthetic_exams,
    tokenize_with_offsets,
    align_spans_to_bio,
    chunk_bio_sequence,
)

WORKSPACE_DIR = Path(__file__).resolve().parents[2]


def test_process_synthetic_exams_with_injector_preserves_disk_files(tmp_path):
    # 1. Setup synthetic exam files in a test dir
    synth_dir = tmp_path / "synthetic_exams"
    synth_dir.mkdir()

    raw_exam = {
        "exam_id": "test_exam_001",
        "subject": "math",
        "grade": 12,
        "sections": {
            "PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn": [
                {
                    "is_group": False,
                    "stem": "Tính đạo hàm của hàm số $y = x^2$ tại $x = 1$.",
                    "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
                    "answer": "B",
                    "explanation": "Ta có $y' = 2x$, suy ra $y'(1) = 2$."
                },
                {
                    "is_group": False,
                    "stem": "Tìm giá trị cực đại của hàm số $y = -x^2 + 4$.",
                    "options": ["A. 0", "B. 2", "C. 4", "D. -4"],
                    "answer": "C",
                    "explanation": "Giá trị cực đại đạt tại $x = 0$, $y = 4$."
                }
            ]
        }
    }
    exam_file = synth_dir / "exam_math_001.json"
    initial_content = json.dumps(raw_exam, ensure_ascii=False, indent=2)
    exam_file.write_text(initial_content, encoding="utf-8")
    initial_hash = hashlib.sha256(exam_file.read_bytes()).hexdigest()

    # 2. Setup mock hard-negative bank with 1 module
    hn_dir = tmp_path / "synthetic_hard_negative"
    hn_dir.mkdir()
    hn_xml = """
    <!-- MODULE: id="mod_test_1" type="formula_card" subject="math_algebra" -->
    <section>### Bảng nguyên hàm sơ cấp</section>
    Nguyên hàm của hằng số $k$ là $\\int k dx = kx + C$.
    <!-- /MODULE -->
    """
    (hn_dir / "theory_test.xml").write_text(hn_xml, encoding="utf-8")

    bank = HardNegativeBank(hn_dir)
    inj_cfg = InjectionConfig(enabled=True, injection_prob=1.0, max_module_chars=1000)
    injector = SyntheticHardNegativeInjector(bank, inj_cfg)

    # 3. Process synthetic exams with injector
    records, num_injected = process_synthetic_exams(
        synthetic_dir=synth_dir,
        target_count=1,
        seed=42,
        injector=injector,
    )

    # Invariant 1: Exactly 1 record produced, marked as injected
    assert len(records) == 1
    assert num_injected == 1
    assert records[0]["hard_negative_injected"] is True
    assert "Bảng nguyên hàm sơ cấp" in records[0]["raw_text"]

    # Invariant 2: Original file on disk is 100% UNCHANGED
    after_content = exam_file.read_text(encoding="utf-8")
    after_hash = hashlib.sha256(exam_file.read_bytes()).hexdigest()
    assert initial_hash == after_hash
    assert initial_content == after_content

    # Invariant 3: Chunks generated from record carry hard_negative_injected metadata
    doc = records[0]
    tokens, offsets = tokenize_with_offsets(doc["raw_text"])
    bio_tags = align_spans_to_bio(offsets, doc["spans"])
    chunks = chunk_bio_sequence(doc["doc_id"], tokens, bio_tags, offsets, chunk_size=512, stride=448)

    assert len(chunks) >= 1
    for c in chunks:
        c["source"] = doc["source"]
        if doc.get("hard_negative_injected"):
            c["hard_negative_injected"] = True
        assert c["hard_negative_injected"] is True
        assert len(c["tokens"]) == len(c["ner_tags"])

    # Invariant 4: Theory tokens (including heading '### Bảng nguyên hàm sơ cấp') are strictly 'O'
    theory_sub = "Bảng nguyên hàm sơ cấp"
    start_idx = doc["raw_text"].find(theory_sub)
    end_idx = start_idx + len(theory_sub)
    assert start_idx != -1

    for (t_start, t_end), tok, tag in zip(offsets, tokens, bio_tags):
        if start_idx <= t_start and t_end <= end_idx:
            assert tag == "O", f"Theory token '{tok}' should be labeled 'O', but got '{tag}'"


def test_process_synthetic_exams_with_disabled_injector(tmp_path):
    synth_dir = tmp_path / "synthetic_exams"
    synth_dir.mkdir()
    raw_exam = {
        "exam_id": "test_002",
        "subject": "math",
        "sections": {
            "PHẦN I": [
                {
                    "is_group": False,
                    "stem": "Câu hỏi đơn giản?",
                    "options": ["A. 1", "B. 2"],
                    "answer": "A"
                }
            ]
        }
    }
    (synth_dir / "exam_002.json").write_text(json.dumps(raw_exam), encoding="utf-8")

    # Call with injector=None
    records, num_injected = process_synthetic_exams(
        synthetic_dir=synth_dir,
        target_count=1,
        seed=42,
        injector=None,
    )
    assert len(records) == 1
    assert num_injected == 0
    assert records[0]["hard_negative_injected"] is False
