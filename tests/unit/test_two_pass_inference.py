import pytest
import re
from model.inference.predict import build_tagged_xml


def test_refine_canonical_stimuli_trims_banners_and_microspans():
    two_pass_mod = pytest.importorskip("scratch.two_pass_experimental")
    refine_canonical_stimuli = two_pass_mod.refine_canonical_stimuli
    text = (
        "PART 6 - TEXT COMPLETION\n\n"
        "To all employees: Please note that the annual safety review will occur on Friday.\n"
        "Ensure all compliance forms are submitted before Thursday afternoon.\n\n"
        "Question 1: What is the memo about?\nA. Safety\nB. Travel\n\n"
        "Question 2: When is it?\nA. Friday\nB. Monday"
    )
    # Simulate a raw neural prediction that swallowed the section banner
    banner_end = len("PART 6 - TEXT COMPLETION\n\n")
    letter_end = text.find("Question 1:")
    
    pass1_spans = [
        # Spurious micro-span
        {"label": "stimulus", "start": 0, "end": 4, "text": "PART"},
        # Stimulus with leading banner
        {"label": "stimulus", "start": 0, "end": letter_end, "text": text[0:letter_end]}
    ]
    q1_idx = text.find("Question 1:")
    q2_idx = text.find("Question 2:")
    q_spans = [
        {"label": "question_label", "start": q1_idx, "end": q1_idx + 11, "text": "Question 1:"},
        {"label": "question_label", "start": q2_idx, "end": q2_idx + 11, "text": "Question 2:"}
    ]

    refined = refine_canonical_stimuli(text, pass1_spans, q_spans, min_length=50)

    # Micro-span must be suppressed
    assert len(refined) == 1
    stim = refined[0]
    # Leading banner 'PART 6 - TEXT COMPLETION' must be stripped
    assert not stim["text"].startswith("PART 6")
    assert stim["text"].startswith("To all employees")
    assert "Question 1:" not in stim["text"]


def test_build_tagged_xml_zero_corruption():
    text = "Section I: Reading\nPassage A...\nQuestion 1. What?\nA. 1\nB. 2"
    spans = [
        {"label": "section", "start": 0, "end": 18, "text": "Section I: Reading"},
        {"label": "stimulus", "start": 19, "end": 31, "text": "Passage A..."},
        {"label": "question_label", "start": 32, "end": 43, "text": "Question 1."},
        {"label": "stem", "start": 44, "end": 49, "text": "What?"},
        {"label": "option_label", "start": 50, "end": 52, "text": "A."},
        {"label": "option_text", "start": 53, "end": 54, "text": "1"},
        {"label": "option_label", "start": 55, "end": 57, "text": "B."},
        {"label": "option_text", "start": 58, "end": 59, "text": "2"}
    ]
    xml = build_tagged_xml(text, spans)
    stripped = re.sub(r"</?[a-z_]+>", "", xml)
    assert stripped == text
