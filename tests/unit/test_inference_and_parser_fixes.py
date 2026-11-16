import random
import re
import sys

from model.inference.predict import (
    _protect_figure_tokens,
    _tokenizer_supports_typed_latex,
    build_tagged_xml,
)
from model.module.dataset import EXAM_SPECIAL_TOKENS, MATH_TYPED_TOKENS
from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
from tools.dataset.build_training_dataset import augment_real_document
from tools.evaluation.evaluate_model import check_zero_corruption


class _VocabTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_math_special_tokens_defined_and_augmented():
    for tok in MATH_TYPED_TOKENS:
        assert tok in EXAM_SPECIAL_TOKENS
    assert "[LATEX]" in EXAM_SPECIAL_TOKENS

    raw = "Câu 1: Cho hàm số $f(x) = x^2 + 1$ và biến $a$ cùng biểu thức $(x^3 + 2x^2 - 5x + 1)/(x + 1)$."
    spans = [
        {"label": "question_label", "start": 0, "end": 6, "text": "Câu 1:"},
        {"label": "stem", "start": 7, "end": len(raw), "text": raw[7:]},
    ]
    saw_typed = set()
    for seed in range(25):
        out = augment_real_document(
            raw,
            spans,
            rng=random.Random(seed),
            latex_mask_prob=1.0,
            use_typed_latex=True,
        )
        assert out is not None
        aug_text, _ = out
        for tok in MATH_TYPED_TOKENS:
            if tok in aug_text:
                saw_typed.add(tok)
    assert "[MATH_CALC]" in saw_typed
    assert "[MATH_VAR_SHORT]" in saw_typed


def test_tokenizer_supports_typed_latex_detection():
    assert _tokenizer_supports_typed_latex(_VocabTokenizer({"[CLS]": 0, "[LATEX]": 1})) is False
    assert _tokenizer_supports_typed_latex(_VocabTokenizer({"[CLS]": 0, "[MATH_CALC]": 100})) is True


def test_protect_figure_tokens_prevents_interior_tag_split():
    text = (
        'Cho đồ thị:\n<figure id="fig_1" description="Đồ thị $a < b$ và $x > 0$." bbox="1,2,3,4" />\nHỏi x?'
    )
    fig_start = text.index("<figure")
    fig_end = text.index("/>") + 2
    token_preds = [
        {"start": 0, "end": fig_start, "tag": "B-stem", "text": text[:fig_start]},
        {"start": fig_start, "end": fig_start + 15, "tag": "I-stem", "text": text[fig_start:fig_start + 15]},
        {"start": fig_start + 15, "end": fig_start + 35, "tag": "O", "text": text[fig_start + 15:fig_start + 35]},
        {"start": fig_start + 35, "end": fig_end, "tag": "B-option_label", "text": text[fig_start + 35:fig_end]},
        {"start": fig_end, "end": len(text), "tag": "I-stem", "text": text[fig_end:]},
    ]
    _protect_figure_tokens(text, token_preds)
    # Surrounding entity is stem, so every token inside <figure .../> must stay I-stem
    for tok in token_preds[1:4]:
        assert tok["tag"] == "I-stem"

    # Reconstructed XML must not contain any open/close tag inside <figure .../>
    xml = build_tagged_xml(
        text,
        [{"label": "stem", "start": 0, "end": len(text), "text": text}],
    )
    fig_match = re.search(r"<figure\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*/>", xml)
    assert fig_match is not None
    assert "<stem>" not in fig_match.group(0) and "</stem>" not in fig_match.group(0)


def test_parse_xml_annotations_preserves_latex_less_than_inside_explanation():
    xml = (
        "<question_label>Câu 1.</question_label> "
        "<stem>Tìm điều kiện của $a$.</stem> "
        "<explanation>Ta có $a < b$ nên hàm nghịch biến trên $(0;1)$ và $b > 0$.</explanation>"
    )
    raw, spans = parse_xml_annotations(xml)
    assert "$a < b$" in raw
    expls = [s for s in spans if s["label"] == "explanation"]
    assert len(expls) == 1
    assert "$a < b$" in expls[0]["text"]
    assert check_zero_corruption(raw, xml)


def test_paired_stimulus_preserves_figure_attributes():
    xml = (
        '<stimulus>Đọc đoạn sau đây thật kỹ.\n'
        '<figure id="fig_1" description="Hình chứa từ Kết thúc đoạn. bên trong alt" bbox="10,10,50,50" />\n'
        'Nội dung chính tiếp tục. Kết thúc đoạn.</stimulus>\n\n'
        '<question_label>Câu 1.</question_label>'
    )
    raw, spans = parse_xml_annotations(xml)
    stims = [s for s in spans if s["label"] == "stimulus"]
    assert len(stims) == 1
    assert "Nội dung chính tiếp tục. Kết thúc đoạn." in stims[0]["text"]
    assert check_zero_corruption(raw, xml)


def test_build_training_dataset_pretokenize_default_true():
    import argparse
    import inspect
    import tools.dataset.build_training_dataset as btd

    src = inspect.getsource(btd.main)
    assert 'default=True' in src and '--no-pretokenize' in src
