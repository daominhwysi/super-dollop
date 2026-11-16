import pytest
from tools.evaluation.evaluate_model import (
    get_normalized_span_bounds,
    evaluate_normalized_exact_matching,
    evaluate_char_mask_metrics,
    calc_f1,
    check_zero_corruption,
    check_cloze_stemless_validity
)


def test_get_normalized_span_bounds():
    text = "   Câu 1.    Đây là nội dung câu hỏi."
    # Span with surrounding whitespace: "   Câu 1.   "
    ns, ne, sub = get_normalized_span_bounds(text, 0, 13)
    assert sub == "Câu 1."
    assert text[ns:ne] == "Câu 1."
    assert ns == 3
    assert ne == 9


def test_normalized_exact_matching_whitespace_insensitivity():
    raw_text = "Câu 1. A. Option A text B. Option B text"
    # Gold has trailing space in option label
    gold_spans = [
        {"label": "question_label", "start": 0, "end": 7},  # "Câu 1. "
        {"label": "option_label", "start": 7, "end": 10},   # "A. "
    ]
    # Pred has strict subword boundary without trailing space
    pred_spans = [
        {"label": "question_label", "start": 0, "end": 6},  # "Câu 1."
        {"label": "option_label", "start": 7, "end": 9},    # "A."
    ]

    tp, fp, fn, g_cnt, p_cnt = evaluate_normalized_exact_matching(
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        label="question_label",
        raw_text=raw_text
    )
    assert tp == 1
    assert fp == 0
    assert fn == 0
    assert g_cnt == 1
    assert p_cnt == 1

    tp_opt, fp_opt, fn_opt, _, _ = evaluate_normalized_exact_matching(
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        label="option_label",
        raw_text=raw_text
    )
    assert tp_opt == 1
    assert fp_opt == 0
    assert fn_opt == 0


def test_char_mask_metrics():
    text_len = 100
    gold_spans = [{"label": "stimulus", "start": 10, "end": 50}]  # length 40
    pred_spans = [{"label": "stimulus", "start": 10, "end": 46}]  # length 36, overlap 36

    inter, p_len, g_len, prec, rec, f1, iou = evaluate_char_mask_metrics(
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        label="stimulus",
        text_len=text_len
    )
    assert inter == 36
    assert p_len == 36
    assert g_len == 40
    assert prec == 1.0
    assert rec == 36 / 40  # 0.9
    assert round(iou, 4) == round(36 / 40, 4)  # 0.9
    assert round(f1, 4) == round(2 * 1.0 * 0.9 / (1.0 + 0.9), 4)  # ~0.9474


def test_zero_corruption_and_cloze_stemless():
    raw = "Câu 1. A. text"
    xml_valid = "<question><question_label>Câu 1.</question_label> <option_label>A.</option_label> <option_text>text</option_text></question>"
    assert check_zero_corruption(raw, xml_valid) is True

    valid_cloze, stemless_count = check_cloze_stemless_validity(xml_valid)
    assert valid_cloze is True
    assert stemless_count == 1
