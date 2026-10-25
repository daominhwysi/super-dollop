import pytest
from model.module.dataset import classify_latex
from model.inference.predict import get_latex_spans, build_tagged_xml


def test_classify_latex_categories():
    # 1. Calculations / Equations
    assert classify_latex("$x = 5$") == "[MATH_CALC]"
    assert classify_latex(r"$$f(x) \approx 2.5$$") == "[MATH_CALC]"
    assert classify_latex(r"$A \equiv B$") == "[MATH_CALC]"

    # 2. Variables (Short: len <= 3)
    assert classify_latex("$x$") == "[MATH_VAR_SHORT]"
    assert classify_latex("$y_1$") == "[MATH_VAR_SHORT]"

    # 3. Variables (Medium: 4 <= len <= 10)
    assert classify_latex("$f(x)$") == "[MATH_VAR_MED]"
    assert classify_latex("$x^2+1$") == "[MATH_VAR_MED]"

    # 4. Expressions (Short: 11 <= len <= 30)
    assert classify_latex(r"$\sqrt{x^2 + y^2}$") == "[MATH_EXPR_SHORT]"

    # 5. Expressions (Long: len > 30)
    assert classify_latex(r"$\frac{\alpha + \beta}{\gamma + \delta} + \sin(\theta) + \cos(\phi)$") == "[MATH_EXPR_LONG]"


def test_zero_corruption_with_typed_latex_masking():
    raw_text = "Câu 1: Cho $x$ và $$y = f(x) + 1$$. Giá trị lớn nhất là?\nA. $1$\nB. $2$"
    latex_spans = get_latex_spans(raw_text)

    # All latex spans should extract valid latex formulas
    assert len(latex_spans) == 4
    for start, end in latex_spans:
        formula = raw_text[start:end]
        placeholder = classify_latex(formula)
        assert placeholder.startswith("[MATH_")

    # Zero corruption check on XML reconstruction
    segments = [
        {"start": 0, "end": 6, "label": "question_label", "text": "Câu 1:"},
        {"start": 7, "end": 56, "label": "stem", "text": raw_text[7:56]}
    ]
    xml = build_tagged_xml(raw_text, segments)
    import re
    stripped = re.sub(r"<[^>]+>", "", xml)
    assert stripped == raw_text, "Stripping tags MUST strictly reproduce the exact raw text!"


def test_atomic_token_pattern_special_tokens():
    from tools.dataset.build_training_dataset import tokenize_with_offsets

    text = "Câu 1. Điền vào chỗ trống: <blank /> hoặc <blank/> hoặc [BLANK]. Công thức là [LATEX] và [MATH_VAR_SHORT]."
    tokens, offsets = tokenize_with_offsets(text)

    # Special tokens must be kept atomic and whole
    assert "<blank />" in tokens
    assert "<blank/>" in tokens
    assert "[BLANK]" in tokens
    assert "[LATEX]" in tokens
    assert "[MATH_VAR_SHORT]" in tokens
