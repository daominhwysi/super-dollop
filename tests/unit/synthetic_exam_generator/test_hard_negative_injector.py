"""
Unit test suite for synthetic_exam_generator/hard_negative_injector.py.
"""

import pytest
import random
from pathlib import Path
from synthetic_exam_generator.hard_negative_injector import (
    HardNegativeBank,
    HardNegativeModule,
    SyntheticHardNegativeInjector,
    InjectionConfig,
)

WORKSPACE_DIR = Path(__file__).resolve().parents[3]


@pytest.fixture
def sample_module():
    return HardNegativeModule(
        module_id="test_m1",
        module_type="definition",
        subject="math_algebra",
        raw_text="### Định nghĩa đạo hàm\nĐạo hàm của hàm số tại một điểm là giới hạn tỉ số gia số.",
        spans=[],
    )


def test_parse_module_xml_sanitization():
    raw_xml = """
    <!-- MODULE: id="m1" type="definition" subject="math" -->
    <!-- NESTED MALFORMED COMMENT -->
    <section>### Định nghĩa hàm số</section>
    Cho hàm số $y = f(x)$ xác định với $x < 0$.<br>
    Đây là dòng tiếp theo.
    <!-- /MODULE -->
    """
    raw_text, spans = HardNegativeBank._parse_module_xml(raw_xml)
    assert "NESTED" not in raw_text
    assert "<br>" not in raw_text
    assert "$x < 0$" in raw_text
    assert "<section>" not in raw_text
    assert "</section>" not in raw_text
    assert "### Định nghĩa hàm số" in raw_text
    # Invariant: Pedagogical theory has ZERO entity spans (100% background 'O')
    assert len(spans) == 0


def test_safe_points_blocks_stimulus_split():
    # Construct exam with a stimulus and two child questions
    spans = [
        {"start": 0, "end": 20, "label": "section"},
        {"start": 30, "end": 100, "label": "stimulus"},
        {"start": 110, "end": 120, "label": "question_label"},
        {"start": 121, "end": 150, "label": "stem"},
        {"start": 151, "end": 160, "label": "option_label"},
        {"start": 161, "end": 180, "label": "option_text"},
        {"start": 190, "end": 200, "label": "question_label"},
        {"start": 201, "end": 230, "label": "stem"},
        {"start": 231, "end": 240, "label": "option_label"},
        {"start": 241, "end": 260, "label": "option_text"},
        {"start": 280, "end": 300, "label": "section"},
    ]
    raw_text = "x" * 350
    injector = SyntheticHardNegativeInjector(None, InjectionConfig())
    safe_pts = injector._discover_safe_points(raw_text, spans)

    # Verify that no safe point was placed between Q1 (180) and Q2 (190) because stimulus was active!
    inter_q_pts = [pos for pos, st in safe_pts if st == "inter_question"]
    assert len(inter_q_pts) == 0, f"Inter-question point erroneously allowed inside stimulus: {inter_q_pts}"

    # Verify inter-section point before section 2 (280) was discovered
    inter_sec_pts = [pos for pos, st in safe_pts if st == "inter_section"]
    assert 280 in inter_sec_pts


def test_injection_span_shifting(sample_module):
    # Dummy exam: Section + 1 standalone question
    raw_text = "Section Title\n\nCâu 1. Stem text\nA. Opt 1\nB. Opt 2\n"
    spans = [
        {"start": 0, "end": 13, "label": "section", "text": "Section Title"},
        {"start": 15, "end": 21, "label": "question_label", "text": "Câu 1."},
        {"start": 22, "end": 31, "label": "stem", "text": "Stem text"},
        {"start": 32, "end": 34, "label": "option_label", "text": "A."},
        {"start": 35, "end": 40, "label": "option_text", "text": "Opt 1"},
        {"start": 41, "end": 43, "label": "option_label", "text": "B."},
        {"start": 44, "end": 49, "label": "option_text", "text": "Opt 2"},
    ]

    # Pre-exam injection (insert at pos 0)
    bank_mock = None
    injector = SyntheticHardNegativeInjector(bank_mock, InjectionConfig(enabled=True, injection_prob=1.0))
    
    # Manually test shift
    from synthetic_exam_generator.hard_negative_injector import HardNegativeBank
    # Simulate sampling
    class MockBank:
        def sample_module(self, *args, **kwargs):
            return sample_module
    
    injector.bank = MockBank()
    aug_text, aug_spans, was_injected = injector.inject_into_reconstructed_exam(
        raw_text, spans, subject="math", rng=random.Random(42)
    )

    assert was_injected is True
    # 100% span character verification
    for s in aug_spans:
        assert aug_text[s["start"] : s["end"]] == s["text"]
