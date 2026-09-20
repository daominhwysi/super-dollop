"""
Unit tests for Editor Agent and Search/Replace Patcher Engine.
"""

import json
from typing import Any, List
import pytest

from sequence_labelling.annotator.editor import (
    EditorAgent,
    EditorResult,
    SearchReplaceBlock,
    SearchReplacePatcher,
)
from sequence_labelling.annotator.reviewer import AuditIssue, IssueSeverity


# ---------------------------------------------------------------------------
# SearchReplacePatcher Unit Tests
# ---------------------------------------------------------------------------

def test_patcher_parse_single_block():
    llm_output = """Here is the surgical fix:
<<<<<<< SEARCH
<question_label>**Câu 1.**</question_label> <stem>Cho hàm số $y=f(x)$. a) Đồng biến. b) Nghịch biến.</stem>
=======
<question_label>**Câu 1.**</question_label> <stem>Cho hàm số $y=f(x)$.</stem>
<option_label>a)</option_label> <option_text>Đồng biến.</option_text>
<option_label>b)</option_label> <option_text>Nghịch biến.</option_text>
>>>>>>> REPLACE
<|END|>"""
    blocks = SearchReplacePatcher.parse_blocks(llm_output)
    assert len(blocks) == 1
    assert "<stem>Cho hàm số $y=f(x)$. a) Đồng biến. b) Nghịch biến.</stem>" in blocks[0].search_text
    assert "<option_label>a)</option_label>" in blocks[0].replace_text


def test_patcher_parse_multiple_blocks():
    llm_output = """```xml
<<<<<<< SEARCH
<question_label>**Câu 1.**</question_label> <stem>Old stem 1</stem>
=======
<question_label>**Câu 1.**</question_label> <stem>New stem 1</stem>
>>>>>>> REPLACE

Some commentary in between

<<<<<<< SEARCH
<question_label>**Câu 2.**</question_label> <stem>Old stem 2</stem>
=======
<question_label>**Câu 2.**</question_label> <stem>New stem 2</stem>
>>>>>>> REPLACE
```"""
    blocks = SearchReplacePatcher.parse_blocks(llm_output)
    assert len(blocks) == 2
    assert "Old stem 1" in blocks[0].search_text
    assert "New stem 1" in blocks[0].replace_text
    assert "Old stem 2" in blocks[1].search_text
    assert "New stem 2" in blocks[1].replace_text


def test_patcher_apply_exact():
    original = (
        "<section>HEADER</section>\n"
        "<question_label>**Câu 1.**</question_label> <stem>Broken stem text</stem>\n"
        "<option_label>A.</option_label> <option_text>Option A</option_text>\n"
    )
    block = SearchReplaceBlock(
        search_text="<stem>Broken stem text</stem>",
        replace_text="<stem>Repaired stem text</stem>",
    )
    patched, applied, failed = SearchReplacePatcher.apply_blocks(original, [block])
    assert applied == 1
    assert len(failed) == 0
    assert "<stem>Repaired stem text</stem>" in patched
    assert "Broken stem text" not in patched


def test_patcher_apply_line_trimmed():
    original = (
        "<section>HEADER</section>   \n"
        "<question_label>**Câu 1.**</question_label>  \n"
        "<stem>Line 1  \nLine 2  </stem>\n"
    )
    # Search block has no trailing spaces
    block = SearchReplaceBlock(
        search_text="<stem>Line 1\nLine 2</stem>",
        replace_text="<stem>Repaired Line 1\nRepaired Line 2</stem>",
    )
    patched, applied, failed = SearchReplacePatcher.apply_blocks(original, [block])
    assert applied == 1
    assert len(failed) == 0
    assert "Repaired Line 1" in patched


def test_patcher_apply_fuzzy_whitespace():
    original = "<stem>Một  vật   dao   động  điều hoà.</stem>"
    block = SearchReplaceBlock(
        search_text="<stem>Một vật dao động điều hoà.</stem>",
        replace_text="<stem>Một chất điểm dao động điều hoà.</stem>",
    )
    patched, applied, failed = SearchReplacePatcher.apply_blocks(original, [block])
    assert applied == 1
    assert "Một chất điểm dao động điều hoà." in patched


def test_patcher_failed_block_handling():
    original = "<stem>Actual text</stem>"
    block = SearchReplaceBlock(
        search_text="<stem>Non-existent text</stem>",
        replace_text="<stem>Replacement</stem>",
    )
    patched, applied, failed = SearchReplacePatcher.apply_blocks(original, [block])
    assert applied == 0
    assert len(failed) == 1
    assert "SEARCH block not found" in failed[0]
    assert patched == original


# ---------------------------------------------------------------------------
# EditorAgent Functional Tests
# ---------------------------------------------------------------------------

def test_editor_agent_deterministic_preclean():
    """Verifies that pure syntax errors (unclosed stems, math inequalities) are resolved deterministically."""
    raw_ocr = "Câu 1. Biểu thức x < 5 và y > 10. A. 1 B. 2"
    bad_xml = "<question_label>Câu 1.</question_label> <stem>Biểu thức x < 5 và y > 10. <option_label>A.</option_label> <option_text>1</option_text> <option_label>B.</option_label> <option_text>2</option_text>"

    editor = EditorAgent()
    res = editor.repair_document(
        annotated_xml=bad_xml,
        raw_ocr_text=raw_ocr,
        doc_id="test_det_clean",
    )
    assert res.success is True
    assert res.deterministic_only is True
    assert "</stem>" in res.repaired_xml
    assert res.final_decision == "PASS"


def test_editor_agent_sub_question_segmentation_mock():
    """Verifies end-to-end mock repair for absorbed True/False sub-questions."""
    raw_ocr = (
        "Câu 1: Cho hàm số y = f(x).\n"
        "a) Hàm số đồng biến trên khoảng (0; 2).\n"
        "b) Hàm số có 3 điểm cực trị."
    )
    absorbed_xml = (
        "<question_label>Câu 1:</question_label> "
        "<stem>Cho hàm số y = f(x).\n"
        "a) Hàm số đồng biến trên khoảng (0; 2).\n"
        "b) Hàm số có 3 điểm cực trị.</stem>"
    )

    issues = [
        AuditIssue(
            category="option_structure",
            severity=IssueSeverity.MAJOR,
            message="Sub-questions a) and b) absorbed into stem without option_label tags.",
            context_snippet="a) Hàm số đồng biến...",
        )
    ]

    mock_diff_response = """<<<<<<< SEARCH
<stem>Cho hàm số y = f(x).
a) Hàm số đồng biến trên khoảng (0; 2).
b) Hàm số có 3 điểm cực trị.</stem>
=======
<stem>Cho hàm số y = f(x).</stem>
<option_label>a)</option_label> <option_text>Hàm số đồng biến trên khoảng (0; 2).</option_text>
<option_label>b)</option_label> <option_text>Hàm số có 3 điểm cực trị.</option_text>
>>>>>>> REPLACE"""

    def mock_completion(messages: List[Any], **kwargs: Any) -> str:
        return mock_diff_response

    editor = EditorAgent()
    res = editor.repair_document(
        annotated_xml=absorbed_xml,
        raw_ocr_text=raw_ocr,
        issues=issues,
        doc_id="test_sub_q",
        completion_fn=mock_completion,
    )

    assert "<option_label>a)</option_label>" in res.repaired_xml
    assert "<option_text>Hàm số đồng biến" in res.repaired_xml
    assert "<option_label>b)</option_label>" in res.repaired_xml


def test_editor_agent_stimulus_anchor_repair_mock():
    """Verifies end-to-end mock repair for broken stimulus anchors."""
    raw_ocr = (
        "Dựa vào thông tin sau đây để trả lời các câu từ 1 đến 2: Đoạn văn mẫu thí nghiệm hóa học.\n"
        "Câu 1. Hiện tượng là gì? A. Kết tủa B. Khí\n"
        "Câu 2. Chất sinh ra là gì? A. CO2 B. H2O"
    )
    bad_anchor_xml = (
        '<stimulus id="stim_1" start_anchor="Không_tồn_tại" end_anchor="Không_khớp" />\n'
        "<question_label>Câu 1.</question_label> <stem>Hiện tượng là gì?</stem> <option_label>A.</option_label> <option_text>Kết tủa</option_text> <option_label>B.</option_label> <option_text>Khí</option_text>\n"
        "<question_label>Câu 2.</question_label> <stem>Chất sinh ra là gì?</stem> <option_label>A.</option_label> <option_text>CO2</option_text> <option_label>B.</option_label> <option_text>H2O</option_text>"
    )

    issues = [
        AuditIssue(
            category="stimulus",
            severity=IssueSeverity.MAJOR,
            message="Stimulus #1 start_anchor not found in raw source document text.",
        )
    ]

    mock_diff_response = """<<<<<<< SEARCH
<stimulus id="stim_1" start_anchor="Không_tồn_tại" end_anchor="Không_khớp" />
=======
<stimulus id="stim_1" start_anchor="Dựa vào thông tin sau đây" end_anchor="thí nghiệm hóa học." />
>>>>>>> REPLACE"""

    def mock_completion(messages: List[Any], **kwargs: Any) -> str:
        return mock_diff_response

    editor = EditorAgent()
    res = editor.repair_document(
        annotated_xml=bad_anchor_xml,
        raw_ocr_text=raw_ocr,
        issues=issues,
        doc_id="test_stimulus_fix",
        completion_fn=mock_completion,
    )

    assert 'start_anchor="Dựa vào thông tin sau đây"' in res.repaired_xml
    assert 'end_anchor="thí nghiệm hóa học."' in res.repaired_xml


def test_fuzzy_replace_large_doc_does_not_hang():
    """Verify that SearchReplacePatcher fuzzy matching does not hang on large 500KB documents."""
    import time
    from sequence_labelling.annotator.editor import SearchReplaceBlock, SearchReplacePatcher

    big_doc = ("<question_label>## Câu 91</question_label> <stem>Nội dung câu 91</stem>\n" * 5000)
    big_doc += "<question_label>## Câu 92</question_label>\n<stem>Các từ “du mục” và “di cư” trong bài thơ?</stem>\n"
    big_doc += ("<question_label>## Câu 93</question_label> <stem>Nội dung câu 93</stem>\n" * 5000)

    block_match = SearchReplaceBlock(
        search_text='<question_label>## Câu 92</question_label> <stem>Các từ “du mục” và “di cư” trong bài thơ?</stem>',
        replace_text='<question_label>## Câu 92</question_label> <stem>REPLACED</stem>',
    )
    block_mismatch = SearchReplaceBlock(
        search_text='<question_label>## Câu 999</question_label> <stem>MISMATCH NEVER IN DOC</stem>',
        replace_text='NEVER',
    )

    t0 = time.time()
    patched, count, fails = SearchReplacePatcher.apply_blocks(big_doc, [block_match, block_mismatch])
    elapsed = time.time() - t0

    assert count == 1
    assert "REPLACED" in patched
    assert len(fails) == 1
    # Must complete in under 1 second (previously hung for 5+ hours)
    assert elapsed < 1.0, f"Search took too long: {elapsed:.2f}s"


def test_edit_review_set_branch_dir_saving(tmp_path):
    """Verify that process_single_revision writes strictly to branch_dir and never mutates original files."""
    from unittest.mock import MagicMock
    from pathlib import Path
    from sequence_labelling.annotator.revision_loop import RevisionLoopResult
    from tools.edit_review_set import process_single_revision

    annotated_dir = tmp_path / "sequence_labelling_annotated"
    annotated_dir.mkdir(parents=True)
    exam_dir = annotated_dir / "exam_001"
    exam_dir.mkdir(parents=True)
    orig_xml_path = exam_dir / "merged.xml"
    orig_json_path = exam_dir / "merged.json"

    orig_xml_content = "<original>exam 1</original>"
    orig_xml_path.write_text(orig_xml_content, encoding="utf-8")
    orig_json_path.write_text('{"document_id": "exam_001"}', encoding="utf-8")

    raw_dir = tmp_path / "sequence_labelling_input_data"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "exam_001.md"
    raw_ocr_text = "Câu 1. Nội dung câu 1.\nA. Đáp án A\nB. Đáp án B"
    raw_path.write_text(raw_ocr_text, encoding="utf-8")

    branch_dir = tmp_path / "sequence_labelling_annotated_branch"

    mock_loop = MagicMock()
    repaired_xml = (
        "<section>ĐỀ THI</section>\n"
        "<question_label>Câu 1.</question_label> <stem>Nội dung câu 1.</stem>\n"
        "<option_label>A.</option_label> <option_text>Đáp án A</option_text>\n"
        "<option_label>B.</option_label> <option_text>Đáp án B</option_text>"
    )
    mock_loop.run.return_value = RevisionLoopResult(
        doc_id="exam_001",
        success=True,
        initial_score=40.0,
        final_score=95.0,
        initial_decision="NEEDS_REVISION",
        final_decision="PASS",
        rounds_completed=1,
        applied_patches_count=1,
        repaired_xml=repaired_xml,
    )

    target = {
        "doc_id": "exam_001",
        "source_decision": "NEEDS_REVISION",
        "xml_path": orig_xml_path,
        "raw_path": raw_path,
        "rel_path": Path("exam_001.md"),
        "rel_xml_path": Path("exam_001/merged.xml"),
        "initial_score": 40.0,
        "issues": [],
    }

    # 1. Run with auto_save=True targeting branch_dir
    res = process_single_revision(
        target=target,
        revision_loop=mock_loop,
        auto_save=True,
        dry_run=False,
        branch_dir=branch_dir,
        annotated_dir=annotated_dir,
    )

    # Verify original files in annotated_dir were NEVER modified
    assert orig_xml_path.read_text(encoding="utf-8") == orig_xml_content
    assert orig_json_path.read_text(encoding="utf-8") == '{"document_id": "exam_001"}'

    # Verify branch files were created properly
    branch_xml_file = branch_dir / "exam_001" / "merged.xml"
    branch_json_file = branch_dir / "exam_001" / "merged.json"
    assert branch_xml_file.exists()
    assert branch_xml_file.read_text(encoding="utf-8") == repaired_xml
    assert branch_json_file.exists()
    assert res["saved_to_branch"] is True
    assert res["branch_xml_path"] == str(branch_xml_file)

    # 2. Strict safety invariant check: if branch_dir is set to annotated_dir, RuntimeError is raised
    with pytest.raises(RuntimeError, match="Direct overwriting of original files is strictly prohibited"):
        process_single_revision(
            target=target,
            revision_loop=mock_loop,
            auto_save=True,
            dry_run=False,
            branch_dir=annotated_dir,
            annotated_dir=annotated_dir,
        )


def test_edit_review_set_reuse_prerun_logs(tmp_path):
    """Verify that process_single_revision passes initial_report and cached_llm_result when reuse_prerun_logs is True."""
    from unittest.mock import MagicMock
    from pathlib import Path
    from sequence_labelling.annotator.revision_loop import RevisionLoopResult
    from sequence_labelling.annotator.reviewer import ReviewReport, ReviewDecision
    from tools.edit_review_set import process_single_revision

    mock_loop = MagicMock()
    mock_loop.run.return_value = RevisionLoopResult(
        doc_id="exam_002",
        success=True,
        initial_score=50.0,
        final_score=90.0,
    )

    xml_path = tmp_path / "exam_002.xml"
    xml_path.write_text("<test>doc</test>", encoding="utf-8")
    raw_path = tmp_path / "exam_002.md"
    raw_path.write_text("test source", encoding="utf-8")

    dummy_report = ReviewReport(
        doc_id="exam_002",
        overall_score=50.0,
        deterministic_score=60.0,
        decision=ReviewDecision.NEEDS_REVISION,
        is_malfunctioned=False,
        issues=[],
        confirmed_issues=[],
    )
    dummy_cached_llm = {"score": 50.0, "parser_error_confirmations": []}

    target = {
        "doc_id": "exam_002",
        "source_decision": "NEEDS_REVISION",
        "xml_path": xml_path,
        "raw_path": raw_path,
        "rel_path": Path("exam_002.md"),
        "rel_xml_path": Path("exam_002.xml"),
        "initial_score": 50.0,
        "issues": [],
        "initial_report": dummy_report,
        "cached_llm_result": dummy_cached_llm,
    }

    # 1. When reuse_prerun_logs=True, initial_report & cached_llm_result are forwarded
    process_single_revision(
        target=target,
        revision_loop=mock_loop,
        auto_save=False,
        dry_run=True,
        branch_dir=tmp_path / "branch",
        annotated_dir=tmp_path,
        reuse_prerun_logs=True,
    )
    mock_loop.run.assert_called_with(
        annotated_xml="<test>doc</test>",
        raw_ocr_text="test source",
        doc_id="exam_002",
        initial_report=dummy_report,
        cached_llm_result=dummy_cached_llm,
    )

    # 2. When reuse_prerun_logs=False, initial_report & cached_llm_result are None
    mock_loop.reset_mock()
    process_single_revision(
        target=target,
        revision_loop=mock_loop,
        auto_save=False,
        dry_run=True,
        branch_dir=tmp_path / "branch",
        annotated_dir=tmp_path,
        reuse_prerun_logs=False,
    )
    mock_loop.run.assert_called_with(
        annotated_xml="<test>doc</test>",
        raw_ocr_text="test source",
        doc_id="exam_002",
        initial_report=None,
        cached_llm_result=None,
    )


def test_edit_review_set_saves_unresolved_candidate_and_checkpoint(tmp_path):
    from pathlib import Path
    from unittest.mock import MagicMock
    from sequence_labelling.annotator.revision_loop import RevisionLoopResult
    from tools.edit_review_set import process_single_revision

    annotated_dir = tmp_path / "annotated"
    exam_dir = annotated_dir / "exam_003"
    exam_dir.mkdir(parents=True)
    original_xml = exam_dir / "merged.xml"
    original_xml.write_text("<stem>Original</stem>", encoding="utf-8")
    raw_path = tmp_path / "exam_003.md"
    raw_path.write_text("Original", encoding="utf-8")
    branch_dir = tmp_path / "branch"

    loop = MagicMock()
    loop.run.return_value = RevisionLoopResult(
        doc_id="exam_003",
        success=False,
        initial_score=60,
        final_score=78,
        initial_decision="NEEDS_REVISION",
        final_decision="NEEDS_REVISION",
        rounds_completed=1,
        applied_patches_count=1,
        repaired_xml="<stem>Candidate</stem>",
    )
    target = {
        "doc_id": "exam_003",
        "source_decision": "NEEDS_REVISION",
        "xml_path": original_xml,
        "raw_path": raw_path,
        "rel_path": Path("exam_003.md"),
        "rel_xml_path": Path("exam_003/merged.xml"),
        "initial_score": 60,
        "issues": [],
    }

    result = process_single_revision(
        target=target,
        revision_loop=loop,
        auto_save=True,
        dry_run=False,
        branch_dir=branch_dir,
        annotated_dir=annotated_dir,
    )

    assert result["success"] is False
    assert result["saved_to_branch"] is True
    assert (branch_dir / "exam_003" / "merged.xml").read_text(encoding="utf-8") == "<stem>Candidate</stem>"
    state_path = branch_dir / "exam_003" / "revision_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "NEEDS_REVISION"
    assert state["result"]["final_score"] == 78


def test_edit_review_set_resumes_compatible_branch_candidate(tmp_path):
    from pathlib import Path
    from tools.edit_review_set import apply_branch_resume, _sha256_file

    original = tmp_path / "original.xml"
    raw = tmp_path / "original.md"
    original.write_text("<stem>Original</stem>", encoding="utf-8")
    raw.write_text("source", encoding="utf-8")
    branch = tmp_path / "branch"
    branch_doc = branch / "exam_004"
    branch_doc.mkdir(parents=True)
    candidate = branch_doc / "merged.xml"
    candidate.write_text("<stem>Candidate</stem>", encoding="utf-8")
    state = {
        "schema_version": 1,
        "status": "NEEDS_REVISION",
        "original_xml_sha256": _sha256_file(original),
        "raw_sha256": _sha256_file(raw),
    }
    (branch_doc / "revision_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    target = {
        "doc_id": "exam_004",
        "source_decision": "NEEDS_REVISION",
        "xml_path": original,
        "raw_path": raw,
        "rel_path": Path("exam_004.md"),
        "rel_xml_path": Path("exam_004/merged.xml"),
    }

    resumed = apply_branch_resume([target], branch, resume=True)

    assert resumed[0]["resume_from_branch"] is True
    assert resumed[0]["xml_path"] == candidate.resolve()
