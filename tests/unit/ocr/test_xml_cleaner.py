import pytest
from pathlib import Path
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.annotator.xml_checker import XMLChecker


def test_clean_math_inequalities():
    broken_xml = """<section># TOÁN</section>
<question_label>**Câu 1.**</question_label> <stem>Tìm $m$ để phương trình có nghiệm khi $x < 0$ và $\\Delta < 24$.</stem>
- <option_label>A.</option_label> <option_text>$m < 1$.</option_text>
- <option_label>B.</option_label> <option_text>$m < 0$.</option_text>
"""
    # Simulate broken XML where $x < 0$ became $x <0>
    broken_with_tags = broken_xml.replace("$x < 0$", "$x <0>$").replace("$\\Delta < 24$", "$\\Delta <24>$")
    
    # Pre-check fails
    val_before = XMLChecker.check(broken_with_tags)
    assert val_before.is_valid is False

    # Clean
    res = XMLCleaner.clean(broken_with_tags)
    assert res.is_valid_after is True
    assert "<0>" not in res.cleaned_xml
    assert "<24>" not in res.cleaned_xml


def test_clean_table_structures():
    broken_table_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Bảng sau:
<table>
  <tr><th>Tên</th><th>Điểm
  <tr><td>Toán<td>10</tr>
  <tr><td>Văn<td>9
</table>
Chọn câu đúng.</stem>
- <option_label>A.</option_label> <option_text>Đúng</option_text>
"""
    res = XMLCleaner.clean(broken_table_xml)
    assert res.is_valid_after is True
    assert "</td>" in res.cleaned_xml
    assert "</th>" in res.cleaned_xml
    assert "</tr>" in res.cleaned_xml
    assert "</table>" in res.cleaned_xml


def test_clean_prohibited_and_presentation_tags():
    raw_xml = """<pages>
<page>
<page_metadata>
{"p": 1}
</page_metadata>
<section># ĐỀ THI</section>
<center><b>TRƯỜNG THPT CHUYÊN</b></center>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu hỏi.</stem>
- <option_label>A.</option_label> <option_text>A</option_text>
<footer>Trang 1/4</footer>
</page>
</pages>
"""
    res = XMLCleaner.clean(raw_xml)
    assert res.is_valid_after is True
    assert "<page_metadata>" not in res.cleaned_xml
    assert "<pages>" not in res.cleaned_xml
    assert "<center>" not in res.cleaned_xml
    assert "<footer>" not in res.cleaned_xml
    assert "TRƯỜNG THPT CHUYÊN" in res.cleaned_xml


def test_clean_boundary_unclosed_tags():
    unclosed_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Câu hỏi 1 không có thẻ đóng stem
<question_label>**Câu 2.**</question_label> <stem>Câu hỏi 2.</stem>
- <option_label>A.</option_label> <option_text>A</option_text>
"""
    res = XMLCleaner.clean(unclosed_xml)
    assert res.is_valid_after is True
    assert "</stem>" in res.cleaned_xml


def test_clean_mismatched_paired_tag():
    mismatched_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Nội dung câu 1</option_text>
- <option_label>A.</option_label> <option_text>A</option_text>
"""
    res = XMLCleaner.clean(mismatched_xml)
    assert res.is_valid_after is True
    assert "</stem>" in res.cleaned_xml


def test_clean_file_in_place(tmp_path: Path):
    file_p = tmp_path / "exam_test.xml"
    file_p.write_text("""<page_metadata>{"p":1}</page_metadata>
<section># TOÁN</section>
<question_label>**Câu 1.**</question_label> <stem>Khi $x <0>$</stem>
- <option_label>A.</option_label> <option_text>A</option_text>
""", encoding="utf-8")

    res = XMLCleaner.clean_file(file_p, write_in_place=True)
    assert res.is_valid_after is True

    # Verify file on disk was updated
    content_after = file_p.read_text(encoding="utf-8")
    assert "<page_metadata>" not in content_after
    assert "<0>" not in content_after


def test_clean_unclosed_stem_before_option_label():
    broken_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Câu hỏi không đóng thẻ stem
- <option_label>A.</option_label> <option_text>Đáp án A</option_text>
- <option_label>B.</option_label> <option_text>Đáp án B</option_text>
"""
    res = XMLCleaner.clean(broken_xml)
    assert res.is_valid_after is True
    assert "</stem>" in res.cleaned_xml


def test_clean_unclosed_option_text_before_next_option():
    broken_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem>Câu hỏi 1.</stem>
- <option_label>A.</option_label> <option_text>Đáp án A không đóng
- <option_label>B.</option_label> <option_text>Đáp án B</option_text>
"""
    res = XMLCleaner.clean(broken_xml)
    assert res.is_valid_after is True
    assert "</option_text>" in res.cleaned_xml


def test_clean_unsupported_tags_t_and_aside():
    broken_xml = """<section># ĐỀ THI</section>
<question_label>**Câu 1.**</question_label> <stem><t>Nội dung câu 1</t> <aside>Ghi chú</aside></stem>
- <option_label>A.</option_label> <option_text>Đáp án A</option_text>
"""
    res = XMLCleaner.clean(broken_xml)
    assert res.is_valid_after is True
    assert "<t>" not in res.cleaned_xml
    assert "<aside>" not in res.cleaned_xml
    assert "Nội dung câu 1" in res.cleaned_xml


def test_clean_negative_and_float_math_inequalities():
    broken_xml = """<section># TOÁN</section>
<question_label>**Câu 1.**</question_label> <stem>Khi $x <-5>$ hoặc $y <0.5>$</stem>
- <option_label>A.</option_label> <option_text>Đúng</option_text>
"""
    res = XMLCleaner.clean(broken_xml)
    assert res.is_valid_after is True
    assert "<-5>" not in res.cleaned_xml
    assert "<0.5>" not in res.cleaned_xml

