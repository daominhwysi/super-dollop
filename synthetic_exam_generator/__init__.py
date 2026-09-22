"""Synthetic exam generation and reconstruction package for Vietnamese educational exams."""

from synthetic_exam_generator.generator import (
    Subject,
    QuestionType,
    Difficulty,
    generate_single_question,
    SUBJECT_DISPLAY,
    QUESTION_TYPE_DISPLAY,
    DIFFICULTY_DISPLAY,
)
from synthetic_exam_generator.exam_compiler import (
    generate_single_exam,
    run_batch_exams_generator,
    get_available_curricula,
)
from synthetic_exam_generator.curriculum import (
    generate_curriculum,
    generate_all_curricula,
    load_curriculum,
    get_curriculum_path,
)
from synthetic_exam_generator.reconstructor import (
    reconstruct_exam,
    reconstruct_question,
    ReconstructorConfig,
    spans_to_xml,
)
from synthetic_exam_generator.parser import (
    parse_question_xml,
    clean_stimulus_text,
    clean_stem_text,
    check_and_clean_options,
)
from synthetic_exam_generator.deepseek_client import chat

__all__ = [
    "Subject",
    "QuestionType",
    "Difficulty",
    "generate_single_question",
    "SUBJECT_DISPLAY",
    "QUESTION_TYPE_DISPLAY",
    "DIFFICULTY_DISPLAY",
    "generate_single_exam",
    "run_batch_exams_generator",
    "get_available_curricula",
    "generate_curriculum",
    "generate_all_curricula",
    "load_curriculum",
    "get_curriculum_path",
    "reconstruct_exam",
    "reconstruct_question",
    "ReconstructorConfig",
    "spans_to_xml",
    "parse_question_xml",
    "clean_stimulus_text",
    "clean_stem_text",
    "check_and_clean_options",
    "chat",
]
