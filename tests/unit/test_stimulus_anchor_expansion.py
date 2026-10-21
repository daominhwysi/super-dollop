"""
test_stimulus_anchor_expansion.py — Unit Tests for Stimulus Anchor Expansion & Linking.
"""

from pathlib import Path
import pytest

from sequence_labelling.annotator.anchor_helper import AnchorHelper
from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
from sequence_labelling.parser.parser import parse_spans_into_structured_questions


def test_anchor_helper_expand_stimulus_anchors():
    xml = """
<stimulus id="stim_1" start_anchor="This is the start" end_anchor="end of passage." />

<section>Reading Passage</section>

This is the start of the reading comprehension text that discusses important matters. Here is more text before the end of passage.

<question_label>Question 1:</question_label> <stem>What is the topic?</stem>
<option_label>A.</option_label> <option_text>Topic A</option_text>
<option_label>B.</option_label> <option_text>Topic B</option_text>
"""
    expanded = AnchorHelper.expand_stimulus_anchors(xml)
    assert '<stimulus id="stim_1">' in expanded
    assert "</stimulus>" in expanded
    assert "start of the reading comprehension text" in expanded


def test_parse_xml_annotations_with_self_closing_stimulus():
    xml = """
<stimulus id="stim_1" start_anchor="Scientists have discovered" end_anchor="in future expeditions." />

Scientists have discovered a new deep-sea coral reef off the coast of Tahiti. The pristine condition of the reef suggests it has remained untouched by global warming, offering hope for marine ecosystems in future expeditions.

<question_label>Question 1:</question_label> <stem>Where was the reef found?</stem>
<option_label>A.</option_label> <option_text>Tahiti</option_text>
<option_label>B.</option_label> <option_text>Hawaii</option_text>
"""
    raw_text, spans = parse_xml_annotations(xml)
    labels = [s["label"] for s in spans]

    assert "stimulus" in labels
    assert "question_label" in labels
    assert "stem" in labels
    assert "option_label" in labels
    assert "option_text" in labels

    stim_span = next(s for s in spans if s["label"] == "stimulus")
    assert "coral reef" in stim_span["text"]
    assert stim_span["start"] < stim_span["end"]
    # Check that raw text matches span slice
    assert raw_text[stim_span["start"] : stim_span["end"]] == stim_span["text"]


def test_parse_spans_reading_subprompt_preserves_active_stimulus():
    raw_text = (
        "Reading passage text here.\n\n"
        "Questions 1-5: Read the passage and mark the best choice.\n\n"
        "Question 1: What is the main idea?\nA. Choice 1\nB. Choice 2\n"
    )

    spans = [
        {
            "start": 0,
            "end": 26,
            "label": "stimulus",
            "text": "Reading passage text here.",
            "params": {"id": "stim_1"},
        },
        {
            "start": 28,
            "end": 85,
            "label": "section",
            "text": "Questions 1-5: Read the passage and mark the best choice.",
            "params": {},
        },
        {
            "start": 87,
            "end": 98,
            "label": "question_label",
            "text": "Question 1:",
            "params": {},
        },
        {
            "start": 99,
            "end": 122,
            "label": "stem",
            "text": "What is the main idea?",
            "params": {},
        },
        {
            "start": 123,
            "end": 125,
            "label": "option_label",
            "text": "A.",
            "params": {},
        },
        {
            "start": 126,
            "end": 134,
            "label": "option_text",
            "text": "Choice 1",
            "params": {},
        },
    ]

    questions, stimuli = parse_spans_into_structured_questions(raw_text, spans)
    assert len(questions) == 1
    q = questions[0]
    # Crucial test: stimulus_id should NOT be None despite intermediate reading section!
    assert q["stimulus_id"] == "stim_1"
    assert q["stimulus_text"] == "Reading passage text here."
    assert "stim_1" in stimuli
