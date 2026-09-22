import unittest
import random
from synthetic_exam_generator.template_bank import (
    SECTION_TEMPLATES,
    EXPLANATION_PREFIXES,
    END_EXAM_SOLUTION_HEADERS,
    BAREM_TABLE_HEADER_STYLES,
    ANSWER_GRID_HEADER_PAIRS,
    get_random_section_header,
    get_random_explanation_prefix,
    get_random_end_solution_header,
    format_barem_table,
    format_answer_grid
)

class TestTemplateBank(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_section_templates_count_and_structure(self):
        total_sections = sum(len(templates) for templates in SECTION_TEMPLATES.values())
        self.assertGreaterEqual(total_sections, 80, "Should have at least 80 section templates")
        self.assertIn("thpt_mc", SECTION_TEMPLATES)
        self.assertIn("thpt_tf", SECTION_TEMPLATES)
        self.assertIn("thpt_sa", SECTION_TEMPLATES)
        self.assertIn("lit_reading", SECTION_TEMPLATES)
        self.assertIn("lit_writing_social", SECTION_TEMPLATES)
        self.assertIn("lit_writing_literary", SECTION_TEMPLATES)
        self.assertIn("toeic_parts", SECTION_TEMPLATES)

    def test_get_random_section_header(self):
        header = get_random_section_header("thpt_mc", self.rng, num_q=12)
        self.assertIsInstance(header, str)
        self.assertGreater(len(header), 5)
        if "{num_q}" in header:
            self.assertIn("12", header)

    def test_explanation_prefixes_count_and_sampling(self):
        self.assertGreaterEqual(len(EXPLANATION_PREFIXES), 15)
        self.assertGreaterEqual(len(END_EXAM_SOLUTION_HEADERS), 10)

        prefix = get_random_explanation_prefix(self.rng)
        self.assertIsInstance(prefix, str)
        self.assertGreater(len(prefix), 2)

        end_hdr = get_random_end_solution_header(self.rng)
        self.assertIsInstance(end_hdr, str)
        self.assertIn("#", end_hdr)

    def test_format_barem_table_styles(self):
        questions_data = [
            (1, "Xác định đúng thể thơ tự do", "0.50"),
            (2, "Nêu rõ 2 hình ảnh biểu tượng trong bài thơ", "0.75"),
            (3, "Phân tích tác dụng của biện pháp nhân hóa", "1.00")
        ]
        self.assertGreaterEqual(len(BAREM_TABLE_HEADER_STYLES), 12)

        # Test formatting across all header styles
        for idx in range(len(BAREM_TABLE_HEADER_STYLES)):
            table_str = format_barem_table(questions_data, self.rng, style_idx=idx)
            self.assertIn("|", table_str)
            self.assertIn("0.50", table_str)
            self.assertIn("0.75", table_str)
            self.assertIn("1.00", table_str)

    def test_format_answer_grid_types(self):
        questions_data = [(i, "A" if i % 2 == 0 else "B") for i in range(1, 21)]

        # 1. Horizontal
        grid_h = format_answer_grid(questions_data, self.rng, format_type="horizontal")
        self.assertIn("|", grid_h)
        self.assertIn("---", grid_h)

        # 2. Vertical
        grid_v = format_answer_grid(questions_data, self.rng, format_type="vertical")
        self.assertIn("|", grid_v)

        # 3. Compact text
        grid_c = format_answer_grid(questions_data, self.rng, format_type="compact_text")
        self.assertIn("1", grid_c)
        self.assertIn("2", grid_c)

        # 4. Bullet / Dash
        grid_d = format_answer_grid(questions_data, self.rng, format_type="dash_separated")
        self.assertIn("1:", grid_d)

if __name__ == '__main__':
    unittest.main()
