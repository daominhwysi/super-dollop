import pytest
from pathlib import Path
from sequence_labelling.annotator.xml_checker import (
    XMLChecker,
    XMLTagIssue,
    XMLValidationResult,
)


@pytest.fixture
def valid_exam_xml():
    return """<section># PHẦN I. CÂU HỎI TRẮC NGHIỆM</section>

<question_label>**Câu 1.**</question_label> <stem>Trong không gian $Oxyz$, cho điểm $A(1;2;3)$. Tọa độ hình chiếu của $A$ lên mặt phẳng $(Oxy)$ là</stem>
- <option_label>A.</option_label> <option_text>$(1;2;0)$.</option_text>
- <option_label>B.</option_label> <option_text>$(1;0;3)$.</option_text>
- <option_label>C.</option_label> <option_text>$(0;2;3)$.</option_text>
- <option_label>D.</option_label> <option_text>$(0;0;3)$.</option_text>
<explanation>Tọa độ hình chiếu của điểm $A$ lên mặt phẳng $(Oxy)$ là $(1;2;0)$.</explanation>

<stimulus id="stim_1" start_anchor="Dựa vào đoạn tư liệu sau" end_anchor="trả lời các câu hỏi 2 và 3." />

<question_label>**Câu 2.**</question_label> <stem>Dựa vào bảng số liệu sau:
<table>
  <tr><th>Năm</th><th>Sản lượng</th></tr>
  <tr><td>2020</td><td><option_text>100 tấn</option_text></td></tr>
</table>
Nhận xét nào sau đây là <b>chính xác</b>?</stem>
- <option_label>a)</option_label> <option_text>Sản lượng tăng đều.</option_text>
- <option_label>b)</option_label> <option_text>Sản lượng giảm mạnh.</option_text>

<figure id="fig_1" description="Biểu đồ hình cột" bbox="10,20,300,400" />
<|END|>"""


def test_xml_checker_valid_exam(valid_exam_xml):
    res = XMLChecker.check(valid_exam_xml)
    assert res.is_valid is True
    assert len(res.issues) == 0
    assert res.has_mismatched_tags is False
    assert res.has_unclosed_tags is False
    assert res.has_unexpected_closing_tags is False
    assert res.has_truncated_tag is False
    assert res.tag_counts["question_label"] == 2
    assert res.tag_counts["stem"] == 2
    assert res.tag_counts["option_label"] == 6
    assert res.tag_counts["option_text"] == 7
    assert res.tag_counts["explanation"] == 1
    assert res.tag_counts["stimulus"] == 1
    assert res.tag_counts["figure"] == 1


def test_xml_checker_mismatched_closing_tag():
    bad_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi ở đây</option_text>
- <option_label>A.</option_label> <option_text>Lựa chọn A</option_text>
"""
    res = XMLChecker.check(bad_xml)
    assert res.is_valid is False
    assert res.has_mismatched_tags is True
    mismatched = [i for i in res.issues if i.issue_type == "mismatched_closing_tag"]
    assert len(mismatched) == 1
    issue = mismatched[0]
    assert issue.tag_name == "option_text"
    assert issue.expected_tag == "stem"
    assert issue.opened_at_line == 2
    assert "Mismatched closing tag '</option_text>'" in issue.message


def test_xml_checker_unclosed_tag():
    bad_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi chưa được đóng thẻ
- <option_label>A.</option_label> <option_text>Lựa chọn A</option_text>
"""
    res = XMLChecker.check(bad_xml)
    assert res.is_valid is False
    assert res.has_unclosed_tags is True
    unclosed = [i for i in res.issues if i.issue_type == "unclosed_tag"]
    assert len(unclosed) == 1
    assert unclosed[0].tag_name == "stem"
    assert unclosed[0].line_number == 2


def test_xml_checker_unexpected_closing_tag():
    bad_xml = """<section># ĐỀ THI</section>
</stem>
<question_label>**Câu 1.**</question_label> <stem>Nội dung</stem>
"""
    res = XMLChecker.check(bad_xml)
    assert res.is_valid is False
    assert res.has_unexpected_closing_tags is True
    unexp = [i for i in res.issues if i.issue_type == "unexpected_closing_tag"]
    assert len(unexp) == 1
    assert unexp[0].tag_name == "stem"
    assert unexp[0].line_number == 2


def test_xml_checker_truncated_tag_at_eof():
    truncated_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi</stem>
- <option_label>A.</option_label> <option_text>Đáp án A</option_text>
<opt"""
    res = XMLChecker.check(truncated_xml)
    assert res.is_valid is False
    assert res.has_truncated_tag is True
    trunc = [i for i in res.issues if i.issue_type == "truncated_tag"]
    assert len(trunc) == 1
    assert trunc[0].severity == "CRITICAL"
    assert "truncated mid-tag" in trunc[0].message


def test_xml_checker_prohibited_tags():
    proh_xml = """<pages>
<page>
<page_metadata>
{"p": 1, "head": "CLEAN"}
</page_metadata>
<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung</stem>
</page>
</pages>
"""
    res = XMLChecker.check(proh_xml)
    assert res.is_valid is False
    assert res.has_prohibited_tags is True
    proh_issues = [i for i in res.issues if i.issue_type == "prohibited_tag"]
    assert len(proh_issues) >= 3  # pages, page, page_metadata


def test_xml_checker_empty_document():
    res = XMLChecker.check("")
    assert res.is_valid is False
    assert any(i.issue_type == "empty_document" for i in res.issues)

    res_no_tags = XMLChecker.check("Đây là văn bản thuần không có thẻ XML nào cả.")
    assert res_no_tags.is_valid is False
    assert any(i.issue_type == "empty_document" for i in res_no_tags.issues)


def test_xml_checker_file_validation(tmp_path: Path):
    file_p = tmp_path / "test_exam.xml"
    file_p.write_text("""<section># TOÁN</section>
<question_label>**Câu 1.**</question_label> <stem>Tính $1+1$.</stem>
- <option_label>A.</option_label> <option_text>2</option_text>
""", encoding="utf-8")

    res = XMLChecker.validate_file(file_p)
    assert res.is_valid is True
    assert "XML Validation PASSED" in res.summary()

    # Broken file
    bad_p = tmp_path / "bad_exam.xml"
    bad_p.write_text("""<section># TOÁN</section>
<question_label>**Câu 1.**</question_label> <stem>Tính $1+1$.</option_text>
""", encoding="utf-8")

    bad_res = XMLChecker.validate_file(bad_p)
    assert bad_res.is_valid is False
    assert "Contains Mismatched Closing Tag" in bad_res.summary()


def test_xml_checker_ignore_unclosed_non_system_tags_in_table():
    xml_with_unclosed_table = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Bảng sau:
<table>
  <tr><th>Tên<th>Điểm
  <tr><td>Toán<td>10</tr>
  <tr><td>Văn<td>9
</table>
Chọn đáp án đúng.</stem>
- <option_label>A.</option_label> <option_text>Đúng</option_text>
- <option_label>B.</option_label> <option_text>Sai</option_text>
"""
    res = XMLChecker.check(xml_with_unclosed_table)
    assert res.is_valid is True
    assert res.has_unclosed_tags is False
    assert res.has_mismatched_tags is False
    assert len(res.issues) == 0


def test_xml_checker_ignore_unclosed_non_system_inline_tags():
    xml_with_unclosed_formatting = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Cho tam giác $ABC$ có <p>đoạn 1 <p>đoạn 2 với <b>chữ in đậm chưa đóng</stem>
- <option_label>A.</option_label> <option_text>Lựa chọn <i>nghiêng chưa đóng</option_text>
- <option_label>B.</option_label> <option_text>Lựa chọn <span>span chưa đóng</option_text>
<explanation>Lời giải chi tiết <div>div chưa đóng</explanation>
"""
    res = XMLChecker.check(xml_with_unclosed_formatting)
    assert res.is_valid is True
    assert res.has_unclosed_tags is False
    assert res.has_mismatched_tags is False
    assert len(res.issues) == 0


def test_xml_checker_ignore_unclosed_non_system_tag_at_eof():
    xml_with_dangling_html_at_eof = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi.</stem>
- <option_label>A.</option_label> <option_text>Đáp án A.</option_text>
<b>
"""
    res = XMLChecker.check(xml_with_dangling_html_at_eof)
    assert res.is_valid is True
    assert res.has_unclosed_tags is False
    assert len(res.issues) == 0


def test_xml_checker_strictly_enforces_system_tags():
    bad_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi chưa đóng thẻ stem
- <option_label>A.</option_label> <option_text>Lựa chọn A</option_text>
"""
    res = XMLChecker.check(bad_xml)
    assert res.is_valid is False
    assert res.has_unclosed_tags is True
    assert any(i.tag_name == "stem" and i.issue_type == "unclosed_tag" for i in res.issues)

