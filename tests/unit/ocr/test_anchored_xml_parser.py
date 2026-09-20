import pytest
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import (
    STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE,
    find_anchor_position,
    recover_text_from_anchors,
    parse_xml_with_anchors,
    AnchoredXMLLLMExamParser,
)

SAMPLE_OCR_TEXT = """PHẦN I. Câu trắc nghiệm nhiều lựa chọn. Thí sinh trả lời từ câu 1 đến câu 2.

Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the questions.
Renewable energy has become a central focus of global climate policy over the past decade.
Solar and wind power are leading the transition toward a zero-emission future.

Câu 1. What is the main topic of the passage?
A. The history of fossil fuels
B. Solar and wind power leading energy transition
C. The discovery of electricity
D. Space exploration technologies

Câu 2. According to the passage, renewable energy is central to what?
A. Global climate policy
B. Financial markets
C. Traditional mining
D. Automotive sales
"""


def test_system_prompt_defines_xml_and_tags():
    assert "<question_label>" in STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE
    assert "<section>" in STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE
    assert "<stimulus id=" in STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE
    assert "<stem>" in STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE
    assert "<option_label>" in STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE


def test_anchor_resolution_in_xml():
    tagged_xml = """<section>PHẦN I. Câu trắc nghiệm nhiều lựa chọn. Thí sinh trả lời từ câu 1 đến câu 2.</section>

<stimulus id="stim_1" start_anchor="Read the following passage" end_anchor="zero-emission future." />

<question_label>Câu 1.</question_label> <stem>What is the main topic of the passage?</stem>
- <option_label>A.</option_label> <option_text>The history of fossil fuels</option_text>
- <option_label>B.</option_label> <option_text>Solar and wind power leading energy transition</option_text>
- <option_label>C.</option_label> <option_text>The discovery of electricity</option_text>
- <option_label>D.</option_label> <option_text>Space exploration technologies</option_text>
"""

    spans, stimuli, questions = parse_xml_with_anchors(SAMPLE_OCR_TEXT, tagged_xml)
    assert "stim_1" in stimuli
    assert "Renewable energy has become a central focus" in stimuli["stim_1"]
    assert len(questions) == 1
    assert questions[0]["question_number"] == "Câu 1."
    assert questions[0]["stem"] == "What is the main topic of the passage?"
    assert len(questions[0]["options"]) == 4
    assert questions[0]["options"][1]["label"] == "B"
    assert questions[0]["section"] == "PHẦN I. Câu trắc nghiệm nhiều lựa chọn. Thí sinh trả lời từ câu 1 đến câu 2."


def test_anchored_section_tag_is_not_allowed():
    """Verify that anchored stimulus is allowed, but anchored section is ignored/not created as a span."""
    tagged_xml = """<section start_anchor="PHẦN I. Câu trắc nghiệm" end_anchor="câu 1 đến câu 2." />
<stimulus id="stim_1" start_anchor="Read the following passage" end_anchor="zero-emission future." />
<question_label>Câu 1.</question_label> <stem>What is the main topic of the passage?</stem>
- <option_label>A.</option_label> <option_text>The history of fossil fuels</option_text>
"""
    spans, stimuli, questions = parse_xml_with_anchors(SAMPLE_OCR_TEXT, tagged_xml)
    # Stimulus anchor must be resolved
    assert "stim_1" in stimuli
    assert "Renewable energy has become a central focus" in stimuli["stim_1"]
    # Section anchor must NOT be resolved into spans (anchored stimulus only)
    section_spans = [s for s in spans if s["label"] == "section"]
    assert len(section_spans) == 0


def test_xml_mock_completion():
    def mock_completion(messages, model=None, provider=None, max_tokens=None, **kwargs):
        system_content = messages[0]["content"]
        assert "<question_label>" in system_content

        if len(messages) == 2:
            return """<section>PHẦN I. Câu trắc nghiệm nhiều lựa chọn. Thí sinh trả lời từ câu 1 đến câu 2.</section>
<stimulus id="stim_1" start_anchor="Read the following passage" end_anchor="zero-emission future." />
<question_label>Câu 1.</question_label> <stem>What is the main topic of the passage?</stem>
- <option_label>A.</option_label> <option_text>The history of fossil fuels</option_text>
- <option_label>B.</option_label> <option_text>Solar and wind power leading energy transition</option_text>
- <option_label>C.</option_label> <option_text>The discovery of electricity</option_text>
- <option_label>D.</option_label> <option_text>Space exploration technologies</option_text>
<|END|>
"""
        return ""

    parser = AnchoredXMLLLMExamParser()
    res = parser.parse_exam_chunk(SAMPLE_OCR_TEXT, completion_fn=mock_completion)

    assert len(res["questions"]) == 1
    assert len(res["questions"][0]["options"]) == 4
    assert res["questions"][0]["options"][1]["label"] == "B"
    assert res["method"] == "llm_xml_anchored"


def test_anchored_parser_with_codex_provider():
    """Verify that AnchoredXMLLLMExamParser and ParserAgentWorker initialize and route correctly with provider='codex'."""
    from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
    from sequence_labelling.annotator.annotate_ocr import OCRAnnotator
    from unittest.mock import patch, MagicMock

    parser = AnchoredXMLLLMExamParser(model="gpt-5.6-luna", provider="codex")
    assert parser.provider == "codex"
    assert parser.model == "gpt-5.6-luna"

    worker = ParserAgentWorker(model="gpt-5.6-luna", provider="codex")
    assert worker.provider == "codex"
    assert worker.model == "gpt-5.6-luna"

    annotator = OCRAnnotator(model="gpt-5.6-luna", provider="codex")
    assert annotator.provider == "codex"
    assert annotator.client is None

    # Test annotate_text with codex mocked chat
    mock_xml = """<question_label>Câu 1.</question_label> <stem>Test stem</stem>
- <option_label>A.</option_label> <option_text>Opt A</option_text>
<|END|>"""
    with patch("sequence_labelling.annotator.annotate_ocr.chat", return_value=mock_xml):
        res = annotator.annotate_text("Câu 1. Test stem\nA. Opt A")
        assert res["annotated"] is True
        assert len(res["spans"]) >= 3
