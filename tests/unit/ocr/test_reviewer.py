import pytest
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from sequence_labelling.annotator.reviewer import (
    AnnotationReviewerAgent,
    DeterministicAuditor,
    DeepSeekReviewer,
    ReviewDecision,
    IssueSeverity,
    ReviewReport,
    BatchReviewSummary,
    compute_grade,
    compute_llm_penalties,
    PARSER_ERROR_PENALTIES,
    is_spurious_end_sentinel_issue,
)


@pytest.fixture
def valid_xml():
    return """<section># ĐỀ THI KHẢO SÁT CHẤT LƯỢNG</section>

<question_label>**Câu 1.**</question_label> <stem>Trong không gian $Oxyz$, cho điểm $A(1;2;3)$. Tọa độ hình chiếu của $A$ lên mặt phẳng $(Oxy)$ là</stem>
- <option_label>A.</option_label> <option_text>$(1;2;0)$.</option_text>
- <option_label>B.</option_label> <option_text>$(1;0;3)$.</option_text>
- <option_label>C.</option_label> <option_text>$(0;2;3)$.</option_text>
- <option_label>D.</option_label> <option_text>$(0;0;3)$.</option_text>

<stimulus id="stim_1" start_anchor="Dựa vào thông tin sau" end_anchor="trả lời câu 2 và 3." />

<question_label>**Câu 2.**</question_label> <stem>Tính giá trị của biểu thức $P$.</stem>
- <option_label>A.</option_label> <option_text>10.</option_text>
- <option_label>B.</option_label> <option_text>20.</option_text>

<question_label>**Câu 3.**</question_label> <stem>Tính giá trị của biểu thức $Q$.</stem>
- <option_label>A.</option_label> <option_text>30.</option_text>
- <option_label>B.</option_label> <option_text>40.</option_text>
"""


@pytest.fixture
def broken_xml_unclosed():
    return """<section># ĐỀ THI KHẢO SÁT</section>
<question_label>**Câu 1.**</question_label> <stem>Tính tích phân $\\int_0^1 x dx$.
- <option_label>A.</option_label> <option_text>1/2.</option_text>
"""


@pytest.fixture
def broken_xml_prohibited_tags():
    return """<pages>
<page>
<page_metadata>
{"p": 1}
</page_metadata>
<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Biểu thức nào sau đây đúng?</stem>
<option_label>A.</option_label> <option_text>Đúng</option_text>
</page>
</pages>
"""


@pytest.fixture
def broken_xml_no_questions():
    return """<section># ĐỀ THI THỬ THPT QUỐC GIA</section>
Đây là phần giới thiệu hướng dẫn làm bài thi. Thí sinh đọc kỹ đề trước khi làm.
"""


def test_grade_computation():
    assert compute_grade(95.0) == "A"
    assert compute_grade(85.0) == "B"
    assert compute_grade(75.0) == "C"
    assert compute_grade(65.0) == "D"
    assert compute_grade(45.0) == "F"


def test_deterministic_auditor_valid_xml(valid_xml):
    issues, score = DeterministicAuditor.check_xml_syntax(valid_xml)
    assert score == 100.0
    assert len(issues) == 0

    proh_issues, proh_score = DeterministicAuditor.check_prohibited_tags(valid_xml)
    assert proh_score == 100.0
    assert len(proh_issues) == 0

    q_issues, q_score, metrics = DeterministicAuditor.check_question_and_option_structure(valid_xml)
    assert q_score == 100.0
    assert metrics["questions_count"] == 3
    assert metrics["option_labels_count"] == 8
    assert metrics["option_texts_count"] == 8


def test_deterministic_auditor_unclosed_tag(broken_xml_unclosed):
    issues, score = DeterministicAuditor.check_xml_syntax(broken_xml_unclosed)
    assert score == 90.0
    assert any(iss.severity == IssueSeverity.MAJOR for iss in issues)
    assert any("Unclosed tag '<stem>'" in iss.message for iss in issues)
    assert not any(iss.severity == IssueSeverity.CRITICAL for iss in issues)


def test_deterministic_auditor_mismatched_tag():
    mismatched_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi</option_text>
- <option_label>A.</option_label> <option_text>A</option_text>
"""
    issues, score = DeterministicAuditor.check_xml_syntax(mismatched_xml)
    assert score == 85.0
    assert any(iss.severity == IssueSeverity.MAJOR for iss in issues)
    assert any("Mismatched closing tag" in iss.message for iss in issues)
    assert not any(iss.severity == IssueSeverity.CRITICAL for iss in issues)

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(mismatched_xml, use_llm=False)
    assert report.decision == ReviewDecision.NEEDS_REVISION
    assert not report.is_malfunctioned


def test_deterministic_auditor_mid_tag_eof_truncation_is_critical():
    truncated_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung</stem>
<opt"""
    issues, score = DeterministicAuditor.check_xml_syntax(truncated_xml)
    assert score <= 60.0
    assert any(iss.severity == IssueSeverity.CRITICAL for iss in issues)
    assert any("truncated mid-tag" in iss.message for iss in issues)

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(truncated_xml, use_llm=False)
    assert report.decision == ReviewDecision.DISCARD
    assert report.is_malfunctioned is True


def test_trailing_unclosed_explanation_moves_to_needs_revision():
    # 50-question document where last question has unclosed <explanation>
    blocks = []
    for i in range(1, 50):
        blocks.append(
            f"<question_label>**Câu {i}.**</question_label> <stem>Câu hỏi số {i}.</stem>\n"
            f"- <option_label>A.</option_label> <option_text>Opt A</option_text>\n"
            f"- <option_label>B.</option_label> <option_text>Opt B</option_text>\n"
            f"<explanation>Lời giải câu {i}.</explanation>"
        )
    blocks.append(
        "<question_label>**Câu 50.**</question_label> <stem>Câu hỏi số 50.</stem>\n"
        "- <option_label>A.</option_label> <option_text>Opt A</option_text>\n"
        "<explanation>Lời giải câu 50 bị thiếu thẻ đóng"
    )
    doc_xml = "\n\n".join(blocks)

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(doc_xml, use_llm=False)
    assert report.decision == ReviewDecision.NEEDS_REVISION
    assert not report.is_malfunctioned
    assert report.overall_score >= 85.0
    assert len(report.discard_reasons) == 0


def test_unexpected_closing_tag_moves_to_needs_revision():
    unexpected_xml = """<section># ĐỀ THI</section>
</option_text>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi</stem>
- <option_label>A.</option_label> <option_text>A</option_text>
"""
    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(unexpected_xml, use_llm=False)
    assert report.decision == ReviewDecision.NEEDS_REVISION
    assert not report.is_malfunctioned


@patch("sequence_labelling.annotator.reviewer.chat")
def test_review_document_with_mocked_deepseek(mock_chat, valid_xml):
    mock_llm_response = json.dumps({
        "score": 95.0,
        "decision": "PASS",
        "is_malfunctioned": False,
        "discard_reasons": [],
        "rubric_scores": {
            "xml_well_formedness": 100.0,
            "schema_conformance": 100.0,
            "verbatim_fidelity": 95.0,
            "sequence_continuity": 100.0,
            "question_option_completeness": 95.0,
            "stimulus_accuracy": 90.0,
        },
        "issues": [
            {
                "category": "stimulus",
                "severity": "MINOR",
                "message": "Stimulus covers 2 questions appropriately.",
                "context_snippet": None
            }
        ],
        "summary": "High-quality exam sequence labelling with clean XML."
    })
    mock_chat.return_value = mock_llm_response

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True)
    assert report.decision == ReviewDecision.PASS
    assert report.overall_score >= 90.0
    assert report.grade == "A"
    assert "High-quality" in report.summary


@patch("sequence_labelling.annotator.reviewer.chat")
def test_llm_confirms_deterministic_errors_without_repeating_them(mock_chat, broken_xml_unclosed):
    deterministic_issues, _ = DeterministicAuditor.check_xml_syntax(broken_xml_unclosed)
    assert deterministic_issues
    parser_issue = deterministic_issues[0]

    mock_chat.return_value = json.dumps({
        "parser_error_confirmations": [
            {
                "issue_id": 1,
                "is_true_positive": True,
                "reason": "The stem remains open at end of the XML document.",
            }
        ],
        "issues": [
            {
                "category": parser_issue.category,
                "severity": parser_issue.severity.value,
                "message": parser_issue.message,
                "context_snippet": parser_issue.context_snippet,
            },
            {
                "category": "missing_stem",
                "severity": "MINOR",
                "message": "Additional semantic issue not reported by the deterministic pre-check.",
                "context_snippet": None,
            },
        ],
        "summary": "Confirmed the deterministic finding and found one additional issue.",
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(broken_xml_unclosed, use_llm=True)

    matching_parser = [
        issue for issue in report.issues
        if issue.category == parser_issue.category and issue.message == parser_issue.message
    ]
    additional_llm = [issue for issue in report.issues if issue.category == "missing_stem"]

    assert len(matching_parser) == 1
    assert len(additional_llm) == 1
    assert report.llm_confirmations == [
        {
            "issue_id": 1,
            "is_true_positive": True,
            "reason": "The stem remains open at end of the XML document.",
            "category": parser_issue.category,
            "severity": parser_issue.severity.value,
            "message": parser_issue.message,
        }
    ]

    prompt_sent = mock_chat.call_args.kwargs["prompt"]
    assert "Deterministic Parser Findings Requiring Confirmation" in prompt_sent
    assert '"issue_id": 1' in prompt_sent


@patch("sequence_labelling.annotator.reviewer.chat")
def test_false_positive_deterministic_finding_is_not_editor_actionable(
    mock_chat, broken_xml_unclosed
):
    mock_chat.return_value = json.dumps({
        "parser_error_confirmations": [
            {
                "issue_id": 1,
                "is_true_positive": False,
                "reason": "The source document intentionally ends at this boundary.",
            }
        ],
        "issues": [],
        "summary": "The deterministic finding is a false positive.",
    })

    agent = AnnotationReviewerAgent(min_score=0)
    report = agent.review_document(broken_xml_unclosed, use_llm=True)

    assert report.confirmation_status == "complete"
    assert report.confirmed_issues == []
    assert report.llm_confirmations[0]["is_true_positive"] is False


@patch("sequence_labelling.annotator.reviewer.chat")
def test_review_semantic_includes_raw_ocr_source(mock_chat, valid_xml):
    mock_chat.return_value = json.dumps({
        "score": 92.0,
        "decision": "PASS",
        "is_malfunctioned": False,
        "discard_reasons": [],
        "rubric_scores": {
            "xml_well_formedness": 100.0,
            "schema_conformance": 100.0,
            "verbatim_fidelity": 90.0,
            "sequence_continuity": 100.0,
            "question_option_completeness": 100.0,
            "stimulus_accuracy": 90.0,
        },
        "issues": [],
        "summary": "Omitted lecture text does not affect exam completeness."
    })

    raw_ocr = "### BÀI GIẢNG LÝ THUYẾT DÀI 100 TRANG...\n\n" + valid_xml
    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, raw_ocr_text=raw_ocr, use_llm=True)

    # Verify that raw_ocr_text was included in prompt to chat
    call_args = mock_chat.call_args
    prompt_sent = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
    assert "Original Raw OCR Source Text" in prompt_sent
    assert "BÀI GIẢNG LÝ THUYẾT" in prompt_sent
    assert report.decision == ReviewDecision.PASS

def test_discard_document_and_quarantine(tmp_path, broken_xml_no_questions):
    # Setup test workspace
    input_exam_dir = tmp_path / "sequence_labelling_annotated" / "Math" / "exam_999"
    input_exam_dir.mkdir(parents=True)
    xml_file = input_exam_dir / "merged.xml"
    xml_file.write_text(broken_xml_no_questions, encoding="utf-8")

    discard_dir = tmp_path / "sequence_labelling_discarded"

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_file(xml_file, use_llm=False)
    assert report.decision == ReviewDecision.DISCARD

    # Perform discard quarantine
    discard_res = agent.discard_document(
        xml_path=xml_file,
        report=report,
        discard_dir=discard_dir,
        dry_run=False,
    )

    assert discard_res["success"]
    assert not input_exam_dir.exists(), "Source directory should have been moved"

    quarantined_dir = discard_dir / "Math" / "exam_999"
    assert quarantined_dir.exists(), "Target quarantined directory must exist"
    assert (quarantined_dir / "merged.xml").exists()
    assert (quarantined_dir / "audit_report.json").exists()

    audit_data = json.loads((quarantined_dir / "audit_report.json").read_text(encoding="utf-8"))
    assert audit_data["decision"] == "DISCARD"
    assert audit_data["is_malfunctioned"] is True


def test_batch_review_flow(tmp_path, valid_xml, broken_xml_unclosed, broken_xml_no_questions):
    annot_base = tmp_path / "annotated"
    exam1 = annot_base / "exam_1"
    exam2 = annot_base / "exam_2"
    exam3 = annot_base / "exam_3"
    exam1.mkdir(parents=True)
    exam2.mkdir(parents=True)
    exam3.mkdir(parents=True)

    (exam1 / "merged.xml").write_text(valid_xml, encoding="utf-8")
    (exam2 / "merged.xml").write_text(broken_xml_unclosed, encoding="utf-8")
    (exam3 / "merged.xml").write_text(broken_xml_no_questions, encoding="utf-8")

    discard_base = tmp_path / "discarded"

    agent = AnnotationReviewerAgent(min_score=75)
    summary = agent.batch_review(
        annotated_dir=annot_base,
        discard_dir=discard_base,
        auto_discard=True,
        use_llm=False,
        concurrency=3,
    )

    assert summary.total_documents == 3
    assert summary.passed_count == 1
    assert summary.needs_revision_count == 1
    assert summary.discarded_count == 1
    assert len(summary.discarded_paths) == 1

    # Check export markdown report
    report_file = tmp_path / "report.md"
    md_content = agent.export_markdown_report(summary, report_file)
    assert "# 📋 Annotation Quality Audit" in md_content
    assert "Passed" in md_content
    assert report_file.exists()


def test_clean_raw_ocr_text_strips_metadata():
    raw_with_meta = """<pages>
<page>
# ĐỀ THI TOÁN
<page_metadata>
{ "p": 1, "seq": [["Q_START", "1"]] }
</page_metadata>
</page>
<page>
Câu 1. Tính giá trị.
<page_metadata>
{ "p": 2 }
</page>
</pages>"""
    cleaned = DeterministicAuditor.clean_raw_ocr_text(raw_with_meta)
    assert "<page_metadata>" not in cleaned
    assert "</page_metadata>" not in cleaned
    assert "<pages>" not in cleaned
    assert "<page>" not in cleaned
    assert "# ĐỀ THI TOÁN" in cleaned
    assert "Câu 1. Tính giá trị." in cleaned


def test_sample_xml_safely_preserves_tag_boundaries():
    blocks = [
        f"<question_label>**Câu {i}.**</question_label>\n<stem>Nội dung câu hỏi số {i} với độ dài văn bản nhất định.</stem>\n<explanation>Lời giải cho câu {i}.</explanation>"
        for i in range(1, 30)
    ]
    xml_doc = "\n\n".join(blocks)
    assert len(xml_doc) > 2000

    sampled = DeepSeekReviewer._sample_xml_safely(xml_doc, max_chars=1000)
    assert "AUDITOR_SAMPLING_WINDOW" in sampled
    assert not sampled.endswith("</")
    assert not sampled.startswith(">")
    assert "<question_label>**Câu 1.**</question_label>" in sampled


def test_stimulus_wrapping_system_tags_auto_reject():
    # Paired stimulus wrapping stem and question_label
    bad_stim_xml = """<section># ĐỀ THI TOÁN</section>
<stimulus id="stim_1">
<question_label>**Câu 1.**</question_label>
<stem>Nội dung câu hỏi bị stimulus bao bọc sai quy tắc.</stem>
- <option_label>A.</option_label> <option_text>1</option_text>
- <option_label>B.</option_label> <option_text>2</option_text>
</stimulus>
"""
    issues, score = DeterministicAuditor.check_stimulus_wrapping_system_tags(bad_stim_xml)
    assert score == 0.0
    assert any(iss.severity == IssueSeverity.CRITICAL for iss in issues)
    assert any("Stimulus tag illegally wraps system tag" in iss.message for iss in issues)

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(bad_stim_xml, use_llm=False)
    assert report.decision == ReviewDecision.DISCARD
    assert report.is_malfunctioned is True
    assert any("STIMULUS_NESTING" in r for r in report.discard_reasons)


def test_isolated_error_in_large_document_passes():
    # 30-question document with 29 perfect questions and 1 isolated question having sub-items in stem
    blocks = []
    for i in range(1, 30):
        blocks.append(
            f"<question_label>**Câu {i}.**</question_label> <stem>Câu hỏi số {i} tiêu chuẩn.</stem>\n"
            f"- <option_label>A.</option_label> <option_text>Đáp án A</option_text>\n"
            f"- <option_label>B.</option_label> <option_text>Đáp án B</option_text>"
        )
    # 30th question has a single un-tagged sub-item in stem
    blocks.append(
        "<question_label>**Câu 30.**</question_label> <stem>Câu hỏi có ý phụ:\n- a) ý thứ nhất</stem>\n"
        "- <option_label>A.</option_label> <option_text>Đáp án A</option_text>\n"
        "- <option_label>B.</option_label> <option_text>Đáp án B</option_text>"
    )
    large_xml = "\n\n".join(blocks)

    issues, q_score, metrics = DeterministicAuditor.check_question_and_option_structure(large_xml)
    assert metrics["questions_count"] == 30
    assert q_score >= 90.0
    # 1 error in 30 questions (3.3%) should be MINOR, not MAJOR
    subitem_issues = [iss for iss in issues if "sub-questions" in iss.message]
    assert len(subitem_issues) == 1
    assert subitem_issues[0].severity == IssueSeverity.MINOR

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(large_xml, use_llm=False)
    assert report.decision == ReviewDecision.PASS
    assert report.overall_score >= 85.0
    assert not report.is_malfunctioned


def test_systemic_major_errors_across_document():
    # 10 questions where 4 have un-tagged sub-items in stem (40% error rate -> systemic MAJOR)
    blocks = []
    for i in range(1, 11):
        if i <= 4:
            blocks.append(
                f"<question_label>**Câu {i}.**</question_label> <stem>Đề bài {i}:\n- a) Ý a\n- b) Ý b</stem>\n"
                f"- <option_label>A.</option_label> <option_text>Opt A</option_text>"
            )
        else:
            blocks.append(
                f"<question_label>**Câu {i}.**</question_label> <stem>Đề bài {i}</stem>\n"
                f"- <option_label>A.</option_label> <option_text>Opt A</option_text>"
            )
    systemic_xml = "\n\n".join(blocks)
    issues, q_score, metrics = DeterministicAuditor.check_question_and_option_structure(systemic_xml)
    subitem_issues = [iss for iss in issues if "sub-questions" in iss.message]
    assert len(subitem_issues) == 1
    assert subitem_issues[0].severity == IssueSeverity.MAJOR


def test_table_html_structure_valid():
    table_xml = """<section># ĐỀ THI HÓA HỌC</section>
<question_label>## Câu 1:</question_label> <stem>Phát biểu sau đúng hay sai?</stem>
<table>
<tr>
<th>Phát biểu</th>
<th>Đúng</th>
<th>Sai</th>
</tr>
<tr>
<td><option_text>Chất chỉ thị màu là chất có màu biến đổi phụ thuộc pH.</option_text></td>
<td>○</td>
<td>○</td>
</tr>
<tr>
<td><option_text>So với thymolphthalein, methyl da cam chuyển màu ở pH cao hơn.</option_text></td>
<td>○</td>
<td>○</td>
</tr>
</table>
"""
    issues, score = DeterministicAuditor.check_xml_syntax(table_xml)
    assert score == 100.0
    assert len(issues) == 0

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(table_xml, use_llm=False)
    assert report.decision == ReviewDecision.PASS
    assert report.overall_score >= 85.0


def test_unclosed_non_system_tags_ignored_no_penalty():
    table_xml_unclosed = """<section># ĐỀ THI HÓA HỌC</section>
<question_label>## Câu 1:</question_label> <stem>Phát biểu sau đúng hay sai?
<table>
<tr><th>Phát biểu<th>Đúng
<tr><td>Chất chỉ thị<td>○</tr>
<tr><td>Methyl da cam<td>○
</table>
</stem>
- <option_label>A.</option_label> <option_text>Đúng với <b>chữ in đậm chưa đóng</option_text>
"""
    issues, score = DeterministicAuditor.check_xml_syntax(table_xml_unclosed)
    assert score == 100.0
    assert len(issues) == 0


def test_figures_out_of_scope_no_penalties():
    figure_xml = """<section># ĐỀ THI VẬT LÝ</section>
<question_label>**Câu 1.**</question_label> <stem>Cho mạch điện như hình vẽ: <figure id="fig_1" description="Mạch điện RLC nối tiếp" bbox="100,200,300,400" />. Tính cường độ dòng điện.</stem>
- <option_label>A.</option_label> <option_text>1 A</option_text>
- <option_label>B.</option_label> <option_text>2 A</option_text>
"""
    issues, score = DeterministicAuditor.check_xml_syntax(figure_xml)
    assert score == 100.0
    assert len(issues) == 0

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(figure_xml, use_llm=False)
    assert report.decision == ReviewDecision.PASS
    assert report.overall_score >= 90.0


def test_discover_review_targets_merged_vs_chunk_rule(tmp_path):
    # Setup standard exam folder with merged.xml and chunks
    exam1 = tmp_path / "exam_1"
    exam1.mkdir()
    (exam1 / "merged.xml").write_text("<stem>Standard exam</stem>", encoding="utf-8")
    chunks1 = exam1 / "chunks"
    chunks1.mkdir()
    (chunks1 / "chunk_0.xml").write_text("<stem>Standard exam chunk 0</stem>", encoding="utf-8")
    (chunks1 / "chunk_1.xml").write_text("<stem>Standard exam chunk 1</stem>", encoding="utf-8")

    # Setup giant exam folder exceeding 500k tokens
    exam2 = tmp_path / "exam_2"
    exam2.mkdir()
    # Write ~2MB content to exceed 500k tokens
    giant_content = "<stem>" + ("giant text word " * 120_000) + "</stem>"
    (exam2 / "merged.xml").write_text(giant_content, encoding="utf-8")
    chunks2 = exam2 / "chunks"
    chunks2.mkdir()
    (chunks2 / "chunk_0.xml").write_text("<stem>Giant chunk 0</stem>", encoding="utf-8")
    (chunks2 / "chunk_1.xml").write_text("<stem>Giant chunk 1</stem>", encoding="utf-8")

    targets = AnnotationReviewerAgent.discover_review_targets(tmp_path, max_merged_tokens=500_000)
    target_names = [t.name for t in targets]

    # exam_1 is under 500k -> merged.xml selected, chunks ignored
    assert exam1 / "merged.xml" in targets
    assert exam1 / "chunks" / "chunk_0.xml" not in targets
    assert exam1 / "chunks" / "chunk_1.xml" not in targets

    # exam_2 is over 500k -> fallback to chunks, merged.xml ignored
    assert exam2 / "merged.xml" not in targets
    assert exam2 / "chunks" / "chunk_0.xml" in targets
    assert exam2 / "chunks" / "chunk_1.xml" in targets

    # Total targets = 1 (from exam1) + 2 (from exam2) = 3
    assert len(targets) == 3


def test_batch_review_scan_processed_and_resume(tmp_path):
    exam1 = tmp_path / "exam_1"
    exam1.mkdir()
    (exam1 / "merged.xml").write_text("<section># EXAM 1</section><question_label>1.</question_label><stem>Stem 1</stem><option_label>A.</option_label><option_text>Opt 1</option_text>", encoding="utf-8")
    
    exam2 = tmp_path / "exam_2"
    exam2.mkdir()
    (exam2 / "merged.xml").write_text("<section># EXAM 2</section><question_label>1.</question_label><stem>Stem 2</stem><option_label>A.</option_label><option_text>Opt 2</option_text>", encoding="utf-8")

    agent = AnnotationReviewerAgent(min_score=75)
    
    # 1. Initial review: both are processed
    summary1 = agent.batch_review(tmp_path, use_llm=False, overwrite=False)
    assert summary1.total_documents == 2
    assert summary1.passed_count == 2
    assert (exam1 / "audit_report.json").exists()
    assert (exam2 / "audit_report.json").exists()

    # 2. Second review with overwrite=False (resume mode)
    # Mock review_document to ensure it is NOT called for cached files
    with patch.object(agent, "review_document") as mock_rev:
        summary2 = agent.batch_review(tmp_path, use_llm=False, overwrite=False)
        assert summary2.total_documents == 2
        assert summary2.passed_count == 2
        mock_rev.assert_not_called()

    # 3. Third review with overwrite=True
    with patch.object(agent, "review_document", wraps=agent.review_document) as mock_rev:
        summary3 = agent.batch_review(tmp_path, use_llm=False, overwrite=True)
        assert summary3.total_documents == 2
        assert mock_rev.call_count == 2


def test_batch_review_filter_decision(tmp_path):
    exam1 = tmp_path / "exam_1"
    exam1.mkdir()
    (exam1 / "merged.xml").write_text("<section># EXAM 1</section><question_label>1.</question_label><stem>Stem 1</stem><option_label>A.</option_label><option_text>Opt 1</option_text>", encoding="utf-8")
    
    exam2 = tmp_path / "exam_2"
    exam2.mkdir()
    # Empty doc -> DISCARD
    (exam2 / "merged.xml").write_text("", encoding="utf-8")

    agent = AnnotationReviewerAgent(min_score=75)
    summary1 = agent.batch_review(tmp_path, use_llm=False, overwrite=False)
    assert summary1.passed_count == 1
    assert summary1.discarded_count == 1

    # Filter only DISCARD: only exam_2 is re-evaluated, exam_1 is cached
    with patch.object(agent, "review_document", wraps=agent.review_document) as mock_rev:
        summary2 = agent.batch_review(tmp_path, use_llm=False, overwrite=False, filter_decision="DISCARD")
        assert summary2.total_documents == 2

def test_stimulus_self_closing_not_flagged_as_nested():
    xml = """<section># EXAM</section>
<stimulus id="stim_1" start_anchor="Bình moka" end_anchor="trải nghiệm" />
<question_label>**Câu 1.**</question_label>
<stem>Câu hỏi 1?</stem>
<option_label>A.</option_label> <option_text>Lựa chọn A</option_text>
</stimulus>
"""
    issues, score = DeterministicAuditor.check_stimulus_wrapping_system_tags(xml)
    assert not any(iss.category == "stimulus_nesting" for iss in issues)


def test_deepseek_reviewer_sanitizes_latex_escapes_and_trailing_commas():
    reviewer = DeepSeekReviewer(model="gpt-5.6-luna", provider="codex")
    mock_response = """```json
{
  "score": 88.0,
  "decision": "PASS",
  "is_malfunctioned": false,
  "discard_reasons": [],
  "rubric_scores": {
    "xml_well_formedness": 100.0,
    "schema_conformance": 90.0,
    "verbatim_fidelity": 95.0,
    "sequence_continuity": 90.0,
    "question_option_completeness": 95.0,
    "stimulus_accuracy": 85.0,
  },
  "issues": [
    {
      "category": "verbatim_fidelity",
      "severity": "MINOR",
      "message": "Formula contains \\alpha + \\beta = \\gamma and \\underline{text}",
      "context_snippet": "\\frac{a}{b}",
    },
  ],
  "summary": "Valid exam with math expressions."
}
```"""
    with patch("sequence_labelling.annotator.reviewer.chat", return_value=mock_response):
        result = reviewer.review_semantic(
            xml_content="<section>test</section>",
            deterministic_metrics={},
        )
        assert result is not None
        assert result["score"] == 88.0
        assert result["decision"] == "PASS"
        assert "\\alpha" in result["issues"][0]["message"]


def test_det_question_coverage_detects_unannotated_question():
    raw_ocr = """# ĐỀ THI TOÁN HỌC
Câu 1. Cho hàm số y = f(x) liên tục trên R. Tính f(1).
A. 1
B. 2

Câu 2. Tìm giá trị lớn nhất của hàm số g(x) trên đoạn [0; 2].
A. 3
B. 4

Câu 3. Giải phương trình logarit cơ số 2 của x bằng 3.
A. 8
B. 9
"""
    # Parser annotated only Câu 1 and Câu 3, leaving Câu 2 unannotated
    annotated_xml = """<section># ĐỀ THI TOÁN HỌC</section>
<question_label>Câu 1.</question_label> <stem>Cho hàm số y = f(x) liên tục trên R. Tính f(1).</stem>
- <option_label>A.</option_label> <option_text>1</option_text>
- <option_label>B.</option_label> <option_text>2</option_text>

<question_label>Câu 3.</question_label> <stem>Giải phương trình logarit cơ số 2 của x bằng 3.</stem>
- <option_label>A.</option_label> <option_text>8</option_text>
- <option_label>B.</option_label> <option_text>9</option_text>
"""
    issues, score, metrics = DeterministicAuditor.check_question_coverage_via_det(
        annotated_xml, raw_ocr
    )
    assert metrics["det_total_questions"] == 3
    assert metrics["matched_questions_count"] == 2
    assert metrics["unannotated_questions_count"] == 1
    assert score == pytest.approx(66.7, rel=1e-2)

    unannot_issues = [iss for iss in issues if iss.category == "unannotated_question"]
    assert len(unannot_issues) == 1
    assert "Câu 2" in unannot_issues[0].message
    assert "unannotated by the parser" in unannot_issues[0].message


def test_reference_table_alien_format_zero_questions_passes():
    raw_reference_table = """# BẢNG TRA CỨU HẰNG SỐ VẬT LÝ NGUYÊN TỬ
| Hằng số | Ký hiệu | Giá trị |
| :--- | :--- | :--- |
| Tốc độ ánh sáng | c | 3.10^8 m/s |
| Hằng số Planck | h | 6.626.10^-34 J.s |
| Điện tích nguyên tố | e | 1.602.10^-19 C |
"""
    annotated_xml = """<section># BẢNG TRA CỨU HẰNG SỐ VẬT LÝ NGUYÊN TỬ</section>
<table>
<tr><th>Hằng số</th><th>Ký hiệu</th><th>Giá trị</th></tr>
<tr><td>Tốc độ ánh sáng</td><td>c</td><td>3.10^8 m/s</td></tr>
<tr><td>Hằng số Planck</td><td>h</td><td>6.626.10^-34 J.s</td></tr>
<tr><td>Điện tích nguyên tố</td><td>e</td><td>1.602.10^-19 C</td></tr>
</table>
"""
    # Verify structure auditor awards 100.0 and INFO severity when source has 0 questions
    issues, q_score, metrics = DeterministicAuditor.check_question_and_option_structure(
        annotated_xml, raw_ocr_text=raw_reference_table
    )
    assert q_score == 100.0
    assert metrics["questions_count"] == 0
    assert any(iss.category == "question_structure" and iss.severity == IssueSeverity.INFO for iss in issues)
    assert not any(iss.severity == IssueSeverity.CRITICAL for iss in issues)

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(annotated_xml, raw_ocr_text=raw_reference_table, use_llm=False)
    assert report.decision == ReviewDecision.PASS
    assert not report.is_malfunctioned
    assert report.overall_score >= 85.0


def test_alien_format_essay_zero_options_passes():
    raw_essay = """# ĐỀ THI TỰ LUẬN TOÁN CAO CẤP
Bài 1. Cho ma trận A vuông cấp 3 khả nghịch. Hãy chứng minh rằng det(A^-1) = 1/det(A).

Bài 2. Tính tích phân đường loại 2 của trường vectơ F dọc theo đường cong C từ O(0,0) đến B(1,1).
"""
    annotated_xml = """<section># ĐỀ THI TỰ LUẬN TOÁN CAO CẤP</section>
<question_label>Bài 1.</question_label> <stem>Cho ma trận A vuông cấp 3 khả nghịch. Hãy chứng minh rằng det(A^-1) = 1/det(A).</stem>

<question_label>Bài 2.</question_label> <stem>Tính tích phân đường loại 2 của trường vectơ F dọc theo đường cong C từ O(0,0) đến B(1,1).</stem>
"""
    issues, q_score, metrics = DeterministicAuditor.check_question_and_option_structure(
        annotated_xml, raw_ocr_text=raw_essay
    )
    assert q_score == 100.0
    assert metrics["questions_count"] == 2
    assert metrics["option_labels_count"] == 0
    assert len(issues) == 0

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(annotated_xml, raw_ocr_text=raw_essay, use_llm=False)
    assert report.decision == ReviewDecision.PASS
    assert report.overall_score >= 90.0


def test_verbatim_fidelity_with_pruned_lecture_notes():
    # 20 pages (~15,000 chars) of lecture notes with 1 exam question at the end
    lecture_notes = ("Lý thuyết chương 1 giới thiệu tổng quan về các phương pháp giải toán giải tích cổ điển. " * 150)
    raw_text = lecture_notes + "\n\nCâu 1. Cho hàm số f(x) = x^2. Tính đạo hàm f'(1).\nA. 2\nB. 1\n"

    annotated_xml = """<question_label>Câu 1.</question_label> <stem>Cho hàm số f(x) = x^2. Tính đạo hàm f'(1).</stem>
- <option_label>A.</option_label> <option_text>2</option_text>
- <option_label>B.</option_label> <option_text>1</option_text>
"""
    issues, verb_score, metrics = DeterministicAuditor.check_verbatim_alignment(
        annotated_xml, raw_text
    )
    assert verb_score == 100.0
    assert metrics["verbatim_span_fidelity"] == 100.0
    assert metrics["retention_ratio"] < 0.10
    # The low retention is marked as INFO, not CRITICAL or MAJOR
    assert not any(iss.severity in [IssueSeverity.CRITICAL, IssueSeverity.MAJOR] for iss in issues)

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(annotated_xml, raw_ocr_text=raw_text, use_llm=False)
    assert report.decision == ReviewDecision.PASS
    assert report.overall_score >= 85.0


def test_verbatim_fidelity_detects_hallucination():
    raw_text = """Câu 1. Cho hình lập phương ABCD.A'B'C'D' có cạnh bằng a. Tính thể tích khối chóp A.A'B'C'.
A. a^3 / 6
B. a^3 / 3
"""
    # Hallucinated question with completely fabricated stem not in source
    hallucinated_xml = """<question_label>Câu 1.</question_label> <stem>Một vật dao động điều hòa với biên độ A = 5cm và chu kỳ T = 2s. Xác định gia tốc cực đại.</stem>
- <option_label>A.</option_label> <option_text>a^3 / 6</option_text>
- <option_label>B.</option_label> <option_text>a^3 / 3</option_text>
"""
    issues, verb_score, metrics = DeterministicAuditor.check_verbatim_alignment(
        hallucinated_xml, raw_text
    )
    assert verb_score < 70.0
    hallucinated_issues = [iss for iss in issues if iss.category == "verbatim_fidelity" and "modified or hallucinated" in iss.message]
    assert len(hallucinated_issues) >= 1


def test_verbatim_fidelity_ignores_figure_and_table_tags():
    """
    Ensures that <figure .../> and HTML <table>/<tr>/<td> in the source text
    or stem do not cause false positive hallucination flags.
    """
    raw_text = (
        "## Câu 1\n\n"
        "Cho hàm số $y=f(x)$ và có đồ thị như hình bên dưới.\n\n"
        '<figure id="fig_9" description="Graph of y=f(x)" bbox="462,172,794,416" />\n\n'
        "Các khẳng định nào sau đây đúng, khẳng định nào sai?\n\n"
        "<table>\n"
        "<tr><th>Phát biểu</th><th>Đúng</th><th>Sai</th></tr>\n"
        "<tr><td>Hàm số đồng biến trên (0; 2)</td><td>○</td><td>○</td></tr>\n"
        "</table>"
    )

    xml_content = (
        "<question_label>## Câu 1</question_label>\n\n"
        "<stem>Cho hàm số $y=f(x)$ và có đồ thị như hình bên dưới.\n\n"
        '<figure id="fig_9" description="Graph of y=f(x)" bbox="462,172,794,416" />\n\n'
        "Các khẳng định nào sau đây đúng, khẳng định nào sai?\n\n"
        "<table>\n"
        "<tr><th>Phát biểu</th><th>Đúng</th><th>Sai</th></tr>\n"
        "<tr><td><option_text>Hàm số đồng biến trên (0; 2)</option_text></td><td>○</td><td>○</td></tr>\n"
        "</table></stem>"
    )

    issues, verb_score, metrics = DeterministicAuditor.check_verbatim_alignment(
        xml_content, raw_text
    )
    assert verb_score >= 95.0
    hallucinated_issues = [
        iss for iss in issues if iss.category == "verbatim_fidelity" and "modified or hallucinated" in iss.message
    ]
    assert len(hallucinated_issues) == 0



def test_reviewer_system_prompt_contains_parser_example_and_rules():
    prompt = DeepSeekReviewer.SYSTEM_PROMPT

    # 1. Verify full parser example is present
    assert "How the Sequence Labelling Parser Works" in prompt
    assert "Complete Real-World Parser Example:" in prompt
    assert "Raw OCR Input Text:" in prompt
    assert "Expected Parser Output XML:" in prompt
    assert "<stimulus id=\"stim_1\" start_anchor=" in prompt
    assert "<section>SỞ GD&ĐT HÀ NỘI" in prompt
    assert "<question_label>Câu 1:</question_label>" in prompt
    assert "<option_label>a)</option_label>" in prompt
    assert "<explanation>HƯỚNG DẪN GIẢI CHI TIẾT:" in prompt
    assert "<|END|>" in prompt

    # 2. Verify system vs non-system xml tag rule
    assert "SYSTEM XML TAGS (The ONLY tags you should audit" in prompt
    assert "NON-SYSTEM XML TAGS & FORMATTING (COMPLETELY IGNORE IF BROKEN OR UNCLOSED)" in prompt
    assert "ignore broken, unclosed, or mismatched NON-SYSTEM tags" in prompt.lower() or "completely ignore broken, unclosed, or mismatched non-system tags" in prompt.lower()

    # 3. Verify no AI scoring instruction
    assert "CRITICAL INSTRUCTION ON SCORING" in prompt
    assert "You do NOT calculate or provide any numerical scores" in prompt
    assert "The audit algorithm will automatically compute document scores and assign penalties" in prompt

    # 4. Verify taxonomy of parser error categories
    assert "unannotated_question" in prompt
    assert "redundant_section" not in prompt
    assert "stimulus_missing_citation" in prompt
    assert "stimulus_nesting" in prompt
    assert "single_question_stimulus" in prompt
    assert "broken_system_tag" in prompt
    assert "missing_question_label" in prompt
    assert "missing_stem" in prompt
    assert "missing_options" in prompt
    assert "absorbed_subquestions" in prompt
    assert "verbatim_mutation" not in prompt
    assert "VERBATIM SOURCE FIDELITY" in prompt
    assert "mislabelled_element" in prompt


@patch("sequence_labelling.annotator.reviewer.chat")
def test_algorithmic_penalty_calculation_without_ai_score(mock_chat, valid_xml):
    # LLM returns ONLY issues and summary, without any score or rubric_scores
    mock_chat.return_value = json.dumps({
        "issues": [
            {
                "category": "unannotated_question",
                "severity": "MAJOR",
                "message": "Question 'Câu 4' was left unannotated by the parser.",
                "context_snippet": None
            },
            {
                "category": "mislabelled_element",
                "severity": "MINOR",
                "message": "Section tag on line 10 wraps stem.",
                "context_snippet": "<section>Extra</section>"
            },
            {
                "category": "stimulus_missing_citation",
                "severity": "MINOR",
                "message": "Stimulus stim_1 end anchor does not include the citation part.",
                "context_snippet": "start_anchor=\"A\" end_anchor=\"B\""
            }
        ],
        "summary": "Detected 3 parser defects."
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True)

    # Algorithm assigns penalties:
    # unannotated_question (MAJOR) = 20.0
    # mislabelled_element (MINOR) = 5.0
    # stimulus_missing_citation (MINOR) = 5.0
    # Total LLM penalty = 30.0 -> llm_score = 70.0
    assert report.llm_score == 70.0
    # det_score is 100.0, overall = 100 * 0.4 + 70 * 0.6 = 82.0
    assert report.overall_score == 82.0
    assert len(report.issues) >= 3
    # Check rubric deductions
    assert report.rubric_scores.det_question_coverage <= 80.0
    assert report.rubric_scores.schema_conformance <= 95.0
    assert report.rubric_scores.stimulus_accuracy <= 95.0


@patch("sequence_labelling.annotator.reviewer.chat")
def test_broken_system_tag_major_routes_to_needs_revision(mock_chat, valid_xml):
    mock_chat.return_value = json.dumps({
        "issues": [
            {
                "category": "broken_system_tag",
                "severity": "MAJOR",
                "message": "Unclosed system tag <stem> at question 1.",
                "context_snippet": "<stem>test"
            }
        ],
        "summary": "Broken system tag detected."
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True)
    # MAJOR broken system tag routes to NEEDS_REVISION
    assert report.decision == ReviewDecision.NEEDS_REVISION
    assert not report.is_malfunctioned


@patch("sequence_labelling.annotator.reviewer.chat")
def test_stimulus_nesting_llm_issue_triggers_discard(mock_chat, valid_xml):
    mock_chat.return_value = json.dumps({
        "issues": [
            {
                "category": "stimulus_nesting",
                "severity": "CRITICAL",
                "message": "Stimulus wraps question elements.",
                "context_snippet": "<stimulus><question_label>1.</question_label></stimulus>"
            }
        ],
        "summary": "Fatal stimulus nesting."
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True)
    assert report.decision == ReviewDecision.DISCARD
    assert report.is_malfunctioned is True
    assert any("STIMULUS_NESTING" in r for r in report.discard_reasons)


def test_math_inequalities_and_non_system_tags_ignored_in_syntax_check():
    """
    Ensures math inequalities like '<1', malformed/broken tables, and arbitrary non-system
    tags are completely ignored and do not trigger schema_conformance or unknown XML tag errors.
    """
    xml_with_math_and_tables = (
        "<question_label>Câu 1.</question_label>\n"
        "<stem>Cho hàm số f(x):\n"
        "$$\n"
        "\\begin{cases}\n"
        "x^2, & x <1\n"
        "\\end{cases}\n"
        "$$\n"
        "thỏa mãn $\\int_0^2f(x)\\,dx=13$. Tính $T=a+b-ab$.\n"
        "<table>\n"
        "<tr><td>Cell unclosed\n"
        "<custom_widget>Weird tag</custom_widget>\n"
        "</stem>\n"
        "<option_label>A.</option_label> <option_text>10</option_text>\n"
        "<option_label>B.</option_label> <option_text>20</option_text>"
    )

    issues, score = DeterministicAuditor.check_xml_syntax(xml_with_math_and_tables)
    # Must NOT contain unknown tag '<1>' or schema_conformance errors
    for iss in issues:
        assert "<1>" not in iss.message
        assert "Unknown/unsupported XML tag" not in iss.message
    assert score == 100.0


def test_is_spurious_end_sentinel_issue_cases():
    """
    Verifies detection of spurious <|END|> sentinel issues vs genuine issues.
    """
    # Spurious cases (must return True)
    assert is_spurious_end_sentinel_issue(
        "truncated_output",
        "The annotated document ends without the required <|END|> sentinel."
    )
    assert is_spurious_end_sentinel_issue(
        "truncated_output",
        "Missing <|END|> sentinel at EOF."
    )
    assert is_spurious_end_sentinel_issue(
        "truncated_output",
        "Document ends without <|END|> delimiter."
    )
    assert is_spurious_end_sentinel_issue(
        "truncated_output",
        "The document lacks the required <|END|> terminal sentinel."
    )
    assert is_spurious_end_sentinel_issue(
        "xml_syntax",
        "End sentinel <|END|> was omitted at termination."
    )

    # Genuine issues (must return False)
    assert not is_spurious_end_sentinel_issue(
        "truncated_output",
        "Document was truncated mid-tag '<opt' at file end."
    )
    assert not is_spurious_end_sentinel_issue(
        "truncated_output",
        "The document cuts off mid-sentence at question 3 without finishing options."
    )
    assert not is_spurious_end_sentinel_issue(
        "xml_syntax",
        "Output truncated mid-tag at file termination: '<stem>'"
    )
    assert not is_spurious_end_sentinel_issue(
        "unannotated_question",
        "Question 5 was left unannotated by the parser."
    )


@patch("sequence_labelling.annotator.reviewer.chat")
def test_reviewer_drops_spurious_end_sentinel_issue_and_passes(mock_chat, valid_xml):
    """
    Verifies that the reviewer agent ignores the spurious 'ends without the required <|END|> sentinel'
    issue, does not deduct penalties, and correctly PASSES valid documents.
    """
    mock_chat.return_value = json.dumps({
        "issues": [
            {
                "category": "truncated_output",
                "severity": "MAJOR",
                "message": "The annotated document ends without the required <|END|> sentinel.",
                "line_number": None,
                "context_snippet": "<option_label>c.</option_label> <option_text>Hàm số $g(x)=f(1-x)$ nghịch biến trên khoảng ______</option_text>",
            }
        ],
        "summary": "Document lacks required <|END|> sentinel.",
        "is_malfunctioned": False,
        "discard_reasons": [
            "[TRUNCATED_OUTPUT] The annotated document ends without the required <|END|> sentinel."
        ],
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True)

    # Spurious issue and discard reason must be dropped
    assert len(report.issues) == 0
    assert report.llm_score == 100.0
    assert report.overall_score == 100.0
    assert report.decision == ReviewDecision.PASS
    assert not report.is_malfunctioned
    assert len(report.discard_reasons) == 0


@patch("sequence_labelling.annotator.reviewer.chat")
def test_reviewer_preserves_genuine_truncation_issue(mock_chat, valid_xml):
    """
    Verifies that genuine truncation issues (cut off mid-sentence, etc.) are NOT dropped
    and properly penalized.
    """
    mock_chat.return_value = json.dumps({
        "issues": [
            {
                "category": "truncated_output",
                "severity": "MAJOR",
                "message": "The document cuts off mid-sentence before completing question 3.",
                "line_number": None,
                "context_snippet": "<stem>Cho hàm số f(x) có...",
            }
        ],
        "summary": "Prematurely truncated generation.",
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True)

    # Genuine truncation issue must be retained
    assert len(report.issues) == 1
    assert report.issues[0].category == "truncated_output"
    assert report.llm_score == 75.0
    assert report.overall_score == 85.0


def test_reviewer_cached_llm_result_with_spurious_end_sentinel(valid_xml):
    """
    Verifies that when reviewing with a cached LLM result that contained the spurious
    <|END|> issue from a previous run, the reviewer cleans it up and resets the score to 100.
    """
    cached_llm = {
        "score": 75.0,
        "rubric_scores": {"xml_well_formedness": 75.0},
        "issues": [
            {
                "category": "truncated_output",
                "severity": "MAJOR",
                "message": "The annotated document ends without the required <|END|> sentinel.",
                "context_snippet": "<option_label>c.</option_label>",
            }
        ],
        "summary": "Document missing <|END|> sentinel.",
        "is_malfunctioned": False,
        "discard_reasons": [],
    }

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True, cached_llm_result=cached_llm)

    assert len(report.issues) == 0
    assert report.llm_score == 100.0
    assert report.overall_score == 100.0
    assert report.decision == ReviewDecision.PASS


@patch("sequence_labelling.annotator.reviewer.chat")
def test_major_issues_block_pass_even_with_high_score(mock_chat, valid_xml):
    """
    Verifies that documents with confirmed MAJOR defects cannot bypass to PASS
    even if the composite score is >= 85.0.
    """
    mock_chat.return_value = json.dumps({
        "issues": [
            {
                "category": "unannotated_question",
                "severity": "MAJOR",
                "message": "Question 5 was left unannotated by the parser.",
                "context_snippet": "Câu 5: Cho hình phẳng...",
            }
        ],
        "summary": "One question dropped.",
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(valid_xml, use_llm=True)

    # unannotated_question (MAJOR) penalty = 20.0 -> llm_score = 80.0
    # det_score = 100.0 -> overall = 100*0.4 + 80*0.6 = 88.0 >= 85.0
    assert report.overall_score == 88.0
    # Crucial invariant: MAJOR defect MUST route to NEEDS_REVISION, never PASS
    assert report.decision == ReviewDecision.NEEDS_REVISION
    assert not report.is_malfunctioned


@patch("sequence_labelling.annotator.reviewer.chat")
def test_confirmed_deterministic_issues_penalized_in_llm_score(mock_chat, broken_xml_unclosed):
    """
    Verifies that when deterministic findings are confirmed by LLM without repeating them
    in `issues`, the penalties are properly applied to llm_score via confirmed_issues.
    """
    mock_chat.return_value = json.dumps({
        "parser_error_confirmations": [
            {
                "issue_id": 1,
                "is_true_positive": True,
                "reason": "Unclosed stem tag confirmed.",
            }
        ],
        "issues": [],
        "summary": "Confirmed deterministic unclosed tag.",
    })

    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(broken_xml_unclosed, use_llm=True)

    # Deterministic broken_system_tag (MAJOR) penalty = 15.0
    # LLM score should reflect this confirmed issue (100 - 15 = 85.0), not remain at 100.0
    assert report.llm_score == 85.0
    assert len(report.confirmed_issues) == 1
    assert report.decision == ReviewDecision.NEEDS_REVISION


def test_cloze_question_labels_without_stems_valid():
    """
    Verifies that cloze / fill-in items where <question_label> is followed directly
    by choices (without <stem>) are 100% valid sequence labeling outputs.
    """
    cloze_xml = """<stimulus id="stim_1" start_anchor="Read the following" end_anchor="fill in the blanks" />
<section># READING CLOZE TEST</section>
<question_label>Câu 631.</question_label>
<option_label>A.</option_label> <option_text>This chance did not happen</option_text>
<option_label>B.</option_label> <option_text>By happening this chance</option_text>
<option_label>C.</option_label> <option_text>This happen did not by chance</option_text>
<option_label>D.</option_label> <option_text>This did not happen by chance</option_text>

<question_label>Câu 632.</question_label>
<option_label>A.</option_label> <option_text>however</option_text>
<option_label>B.</option_label> <option_text>although</option_text>
<option_label>C.</option_label> <option_text>despite</option_text>
<option_label>D.</option_label> <option_text>therefore</option_text>
"""
    agent = AnnotationReviewerAgent(min_score=75)
    report = agent.review_document(cloze_xml, use_llm=False)

    # Must have 0 structure issues, 100% score, and PASS
    assert report.decision == ReviewDecision.PASS
    assert report.overall_score == 100.0
    assert not any(iss.category == "missing_stem" for iss in report.issues)
    assert not any("empty <stem>" in iss.message for iss in report.issues)

