"""
test_randomized_prompt_builder.py — Tests for DynamicStyleProfile and RandomizedPromptBuilder.
"""

from sequence_labelling.annotator.agy_pdf_ocr import (
    AGY_DIVERSE_RAW_OCR_BASE_PROMPT,
    AgyPDFOCRConverter,
    DynamicStyleProfile,
    RandomizedPromptBuilder,
)


def test_dynamic_style_profile_sampling_is_diverse():
    """Verify that random sampling generates diverse configurations."""
    profiles = [DynamicStyleProfile.sample_random() for _ in range(25)]
    question_formats = {p.question_prefix_format for p in profiles}
    option_layouts = {p.option_layout for p in profiles}
    choice_formats = {p.choice_letter_format for p in profiles}

    # Should observe multiple variations across 25 samples
    assert len(question_formats) >= 3
    assert len(option_layouts) >= 2
    assert len(choice_formats) >= 2


def test_dynamic_style_profile_deterministic_with_seed():
    """Verify that supplying a seed yields identical profiles."""
    p1 = DynamicStyleProfile.sample_random(seed=42)
    p2 = DynamicStyleProfile.sample_random(seed=42)
    p3 = DynamicStyleProfile.sample_random(seed=99)

    assert p1.to_dict() == p2.to_dict()
    assert p1.to_dict() != p3.to_dict()


def test_prompt_builder_contains_core_invariants():
    """Verify that randomized prompts strictly retain content preservation invariants."""
    profile = DynamicStyleProfile(
        question_prefix_format="bold_câu",
        option_layout="vertical_list",
        choice_letter_format="brackets",
        table_format="markdown_pipe",
        spatial_density="ultra_compact",
        stimulus_container="boxed",
    )
    prompt = RandomizedPromptBuilder.build_system_prompt(profile)

    # Invariants must exist
    assert "CORE PRESERVATION INVARIANT:" in prompt
    assert "Keep 100% of educational text" in prompt
    assert "STRICT DISALLOWANCE OF SEQUENCE LABELLING TAGS:" in prompt

    # Specific directives must be present
    assert 'Render question labels in bold: "**Câu {n}:**"' in prompt
    assert "Arrange each choice on its own individual line" in prompt
    assert 'Format choice markers in square brackets: "[A]"' in prompt
    assert "Render data tables as standard Markdown pipe tables" in prompt
    assert "Use ultra-compact spacing" in prompt
    assert "Enclose reading passages with border lines" in prompt


def test_agy_converter_initialization_defaults():
    """Verify converter options and default flags."""
    converter = AgyPDFOCRConverter(randomize_style=True, style_seed=123)
    assert converter.randomize_style is True
    assert converter.style_seed == 123

    converter_fixed = AgyPDFOCRConverter(randomize_style=False)
    assert converter_fixed.randomize_style is False


def test_build_prompt_directives_standalone():
    """Verify build_prompt_directives produces the directives block without base prompt."""
    profile = DynamicStyleProfile(
        question_prefix_format="colon_câu",
        option_layout="two_column_2x2",
    )
    directives = RandomizedPromptBuilder.build_prompt_directives(profile)
    assert "DYNAMIC STYLE DIRECTIVES FOR THIS DOCUMENT" in directives
    assert 'Render question labels with colon: "Câu {n}:"' in directives
    assert "Arrange choices in a 2x2 grid" in directives
    # The static base prompt should NOT be inside directives block
    assert "You are an expert Vietnamese examination document OCR" not in directives
    # Invariants should be included
    assert "CORE PRESERVATION INVARIANT:" in directives

