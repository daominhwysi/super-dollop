import pytest
from sequence_labelling.annotator.anchor_helper import AnchorHelper
from sequence_labelling.annotator.reviewer import DeterministicAuditor

def test_auto_anchor_paired_stimulus():
    xml = """<section># EXAM</section>
<stimulus>
Dựa vào đoạn thông tin sau đây về lịch sử để trả lời các câu hỏi:
Đất nước Việt Nam trải qua hàng ngàn năm dựng nước và giữ nước oanh liệt.
</stimulus>
<question_label>**Câu 1.**</question_label> <stem>Câu hỏi 1?</stem>
<option_label>A.</option_label> <option_text>Lựa chọn A</option_text>
"""
    raw = """# EXAM
Dựa vào đoạn thông tin sau đây về lịch sử để trả lời các câu hỏi:
Đất nước Việt Nam trải qua hàng ngàn năm dựng nước và giữ nước oanh liệt.

**Câu 1.** Câu hỏi 1?
A. Lựa chọn A
"""
    fixed_xml, fixes = AnchorHelper.auto_anchor_stimuli(xml, raw)
    assert len(fixes) > 0
    assert "</stimulus>" not in fixed_xml
    assert '<stimulus id="stim_1"' in fixed_xml
    assert 'start_anchor="Dựa vào đoạn thông tin"' in fixed_xml
    assert 'end_anchor="và giữ nước oanh liệt."' in fixed_xml

    # Verify deterministic audit passes without stimulus issues
    issues, score = DeterministicAuditor.check_stimulus_anchors(fixed_xml, raw, raw)
    assert score == 100.0
    assert len(issues) == 0

def test_clean_orphaned_closing_stimulus():
    xml = """<section># EXAM</section>
<stimulus id="stim_1" start_anchor="Dựa vào đoạn thông tin" end_anchor="giữ nước oanh liệt." />
<question_label>**Câu 1.**</question_label> <stem>Câu hỏi 1?</stem>
<option_label>A.</option_label> <option_text>Lựa chọn A</option_text>
</stimulus>
"""
    fixed_xml, fixes = AnchorHelper.auto_anchor_stimuli(xml)
    assert "</stimulus>" not in fixed_xml
    assert any("orphaned" in f for f in fixes)
