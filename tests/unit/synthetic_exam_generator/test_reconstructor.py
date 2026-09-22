import unittest
from synthetic_exam_generator.reconstructor import (
    reconstruct_question,
    reconstruct_exam,
    ReconstructorConfig,
    generate_ordering_choices
)

class TestReconstructor(unittest.TestCase):
    def test_standard_multiple_choice(self):
        q_data = {
            "is_group": False,
            "stem": "Họ nguyên hàm của hàm số $f(x) = x^2$ là",
            "options": [
                "\\frac{1}{3}x^3 + C.",
                "2x^3 + C.",
                "3x^3 + C.",
                "\\frac{1}{2}x^3 + C."
            ],
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        
        # Test with a specific configuration
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=5)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        expected_text = (
            "Câu 5: Họ nguyên hàm của hàm số $f(x) = x^2$ là\n"
            "A. \\frac{1}{3}x^3 + C.\n"
            "B. 2x^3 + C.\n"
            "C. 3x^3 + C.\n"
            "D. \\frac{1}{2}x^3 + C."
        )
        
        self.assertEqual(raw_text, expected_text)
        
        # Verify spans
        for span in spans:
            start = span["start"]
            end = span["end"]
            label = span["label"]
            text = span["text"]
            
            # Check slice matches
            self.assertEqual(raw_text[start:end], text)
            
            if label == "question_label":
                self.assertEqual(text, "Câu 5: ")
            elif label == "stem":
                self.assertEqual(text, "Họ nguyên hàm của hàm số $f(x) = x^2$ là")
            elif label == "option_label":
                self.assertIn(text, ["A. ", "B. ", "C. ", "D. "])
            elif label == "option_text":
                self.assertIn(text, q_data["options"])

    def test_standard_true_false(self):
        q_data = {
            "is_group": False,
            "stem": "Một vật chuyển động thẳng...",
            "options": [
                "Phương trình đường thẳng AB là...",
                "Góc giữa hai vectơ..."
            ],
            "question_type": "true_false",
            "subject": "physics",
            "grade": 11,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="**Câu {num}:** ",
            option_prefix_style="bold_lowercase_paren",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=2)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        expected_text = (
            "**Câu 2:** Một vật chuyển động thẳng...\n"
            "**a)** Phương trình đường thẳng AB là...\n"
            "**b)** Góc giữa hai vectơ..."
        )
        
        self.assertEqual(raw_text, expected_text)
        
        # Check slice matches
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])

    def test_short_answer(self):
        q_data = {
            "is_group": False,
            "stem": "Tính nguyên hàm...",
            "options": [],
            "question_type": "short_answer",
            "subject": "math_algebra",
            "grade": 12,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="Q{num}. ",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=10)
        self.assertEqual(enriched["raw_text"], "Q10. Tính nguyên hàm...")

    def test_ordering_question(self):
        q_data = {
            "is_group": False,
            "stem": "Sắp xếp thứ tự các bước:",
            "options": ["Bước 1", "Bước 2", "Bước 3"],
            "question_type": "ordering",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            ordering_item_label_style="char",
            ordering_item_prefix_template="* {label}. ",
            ordering_choice_separator=" – ",
            seed="my_stable_seed",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=16)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        # Verify ordering item text and prefixes are labeled as stem
        stem_segments = [s["text"] for s in spans if s["label"] == "stem"]
        
        self.assertIn("Sắp xếp thứ tự các bước:", stem_segments)
        self.assertIn("Bước 1", stem_segments)
        self.assertIn("Bước 2", stem_segments)
        self.assertIn("Bước 3", stem_segments)
        self.assertIn("* a. ", stem_segments)
        self.assertIn("* b. ", stem_segments)
        self.assertIn("* c. ", stem_segments)
        
        # Verify multiple choice options representing permutations are added
        option_texts = [s["text"] for s in spans if s["label"] == "option_text"]
        option_labels = [s["text"] for s in spans if s["label"] == "option_label"]
        
        self.assertEqual(len(option_texts), 4)
        self.assertEqual(len(option_labels), 4)
        
        # The correct option should be present: "a – b – c"
        self.assertIn("a – b – c", option_texts)
        
        # Each permutation option should contain a mix of a, b, c separated by ' – '
        for opt in option_texts:
            self.assertEqual(len(opt.split(" – ")), 3)
            
        # Check that answer is resolved to A, B, C, or D
        self.assertIn("answer", enriched)
        self.assertIn(enriched["answer"], ["A", "B", "C", "D"])
        
        # Verify the choice text corresponding to answer equals correct sequence "a – b – c"
        correct_index = ["A", "B", "C", "D"].index(enriched["answer"])
        self.assertEqual(option_texts[correct_index], "a – b – c")
        
        # Ensure spans are valid slices of raw_text
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])

    def test_group_question(self):
        q_data = {
            "is_group": True,
            "context": "Đây là ngữ cảnh chung.",
            "questions": [
                {
                    "stem": "Câu hỏi 1",
                    "options": ["A1", "B1"]
                },
                {
                    "stem": "Câu hỏi 2",
                    "options": []
                }
            ],
            "question_type": "group_multiple_choice",
            "subject": "history",
            "grade": 12,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            separator_context_questions="\n",
            separator_questions="\n",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        expected_text = (
            "Đây là ngữ cảnh chung.\n"
            "Câu 1: Câu hỏi 1\n"
            "A. A1\n"
            "B. B1\n"
            "Câu 2: Câu hỏi 2"
        )
        
        self.assertEqual(raw_text, expected_text)
        
        # Check slice matches
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])
        self.assertEqual(spans[0]["label"], "stimulus")
        self.assertEqual(spans[0]["text"], "Đây là ngữ cảnh chung.")

    def test_group_question_with_stimulus_field(self):
        q_data = {
            "is_group": True,
            "stimulus": "Đoạn văn đọc hiểu chung.",
            "questions": [
                {
                    "stem": "Câu hỏi 1",
                    "options": ["A", "B"]
                }
            ],
            "question_type": "group_multiple_choice",
            "subject": "literature",
            "grade": 11,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            separator_stimulus_questions="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        spans = enriched["spans"]
        self.assertEqual(spans[0]["label"], "stimulus")
        self.assertEqual(spans[0]["text"], "Đoạn văn đọc hiểu chung.")
        q_data = {
            "is_group": False,
            "stem": "Hỏi han gì đó...",
            "options": ["O1", "O2", "O3", "O4"],
            "question_type": "multiple_choice",
            "subject": "geography",
            "grade": 8,
            "difficulty": "recognize"
        }
        
        # With default randomized config, two calls on the same question dict (no config passed)
        # should produce identical output because seed defaults to the question content.
        enriched1 = reconstruct_question(q_data)
        enriched2 = reconstruct_question(q_data)
        
        self.assertEqual(enriched1["raw_text"], enriched2["raw_text"])
        self.assertEqual(enriched1["spans"], enriched2["spans"])

    def test_reconstruct_literature_no_options(self):
        # Literature question has no options list
        q_data = {
            "is_group": False,
            "stem": "Đề bài tự luận Ngữ văn...",
            "question_type": "short_answer",
            "subject": "literature",
            "grade": 12,
            "difficulty": "high_application"
        }
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        self.assertEqual(enriched["raw_text"], "Câu 1: Đề bài tự luận Ngữ văn...")
        
        spans = enriched["spans"]
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["label"], "question_label")
        self.assertEqual(spans[1]["label"], "stem")

    def test_reconstruct_english_cloze_blank_normalization(self):
        # English cloze passages should normalize spaces/underscores/dots into [BLANK]
        q_data = {
            "is_group": True,
            "context": "This is a passage with (1)_____ word and (2).... sentence gap.",
            "questions": [
                {
                    "stem": "Question 1: What does it mean _____?",
                    "options": ["A", "B"]
                },
                {
                    "stem": "Question 2",
                    "options": ["C", "D"]
                }
            ],
            "question_type": "group_multiple_choice",
            "subject": "english",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            question_prefix_template="Question {num}: ",
            option_prefix_style="capital_dot",
            separator_context_questions="\n",
            separator_questions="\n",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        
        # Check that context has been normalized
        self.assertTrue(raw_text.startswith("This is a passage with (1) <blank /> word and (2) <blank /> sentence gap."))
        # Check that group sub-question stem has been normalized
        self.assertIn("Question 1: What does it mean <blank/>?", raw_text)

    def test_reconstruct_english_standard_blank_normalization(self):
        q_data = {
            "is_group": False,
            "stem": "He ___ a book yesterday.",
            "options": ["read", "reads", "reading", "has read"],
            "question_type": "multiple_choice",
            "subject": "english",
            "grade": 12,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            question_prefix_template="Question {num}: ",
            option_prefix_style="capital_dot",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        
        self.assertEqual(raw_text, "Question 1: He <blank/> a book yesterday.\nA. read\nB. reads\nC. reading\nD. has read")

    def test_reconstruct_entire_exam(self):
        from synthetic_exam_generator.reconstructor import reconstruct_exam
        exam_data = {
            "exam_id": "test_exam_123",
            "subject": "physics",
            "grade": 11,
            "sections": {
                "PHẦN I. Trắc nghiệm": [
                    {
                        "is_group": False,
                        "stem": "Câu hỏi vật lý.",
                        "options": ["A1", "B1"],
                        "question_type": "multiple_choice",
                        "subject": "physics",
                        "grade": 11
                    }
                ]
            }
        }
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            separator_questions="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_exam(exam_data, config)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        self.assertIn("ĐỀ THI MÔN: VẬT LÝ - LỚP 11", raw_text)
        self.assertIn("PHẦN I. Trắc nghiệm", raw_text)
        self.assertIn("Câu 1: Câu hỏi vật lý.", raw_text)
        self.assertIn("A. A1", raw_text)
        
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])

    def test_inject_vietnamese_typos(self):
        from synthetic_exam_generator.reconstructor import inject_vietnamese_typos, get_stable_random
        rng = get_stable_random("test_seed")
        text = "hàm số đồng biến trên khoảng"
        typo_text = inject_vietnamese_typos(text, typo_rate=1.0, rng=rng)
        self.assertNotEqual(text, typo_text)

    def test_randomize_blank_tokens(self):
        from synthetic_exam_generator.reconstructor import randomize_blank_tokens, get_stable_random
        rng = get_stable_random("test_seed_blank")
        text = "This is a <blank/> space."
        randomized = randomize_blank_tokens(text, rng)
        self.assertNotIn("<blank/>", randomized)

    def test_process_latex_variations(self):
        from synthetic_exam_generator.reconstructor import process_latex_variations, get_stable_random
        rng = get_stable_random("test_seed_latex")
        text = "Hàm số $y = x^2$ liên tục."
        
        masked = process_latex_variations(text, "[LATEX]", mask_prob=1.0, rng=rng)
        self.assertIn("[LATEX]", masked)
        self.assertNotIn("y = x^2", masked)
        
        raw_var = process_latex_variations(text, "[LATEX]", mask_prob=0.0, rng=rng)
        self.assertTrue(any(x in raw_var for x in ["$y = x^2$", "\\( y = x^2 \\)", "y = x^2"]))

    def test_option_dropping(self):
        q_data = {
            "is_group": False,
            "stem": "Câu hỏi test option drop.",
            "options": ["A1", "B1", "C1", "D1"],
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(option_drop_prob=1.0, seed="drop_seed")
        enriched = reconstruct_question(q_data, config)
        options_in_spans = [s["text"] for s in enriched["spans"] if s["label"] == "option_text"]
        self.assertTrue(len(options_in_spans) < 4)

    def test_option_permutation(self):
        q_data = {
            "is_group": False,
            "stem": "Câu hỏi trắc nghiệm.",
            "options": ["Đáp án A (Đúng)", "Đáp án B", "Đáp án C", "Đáp án D"],
            "answer": "A",
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(enable_permutations=True, seed="perm_seed")
        
        shuffled = False
        for i in range(10):
            config.seed = f"seed_{i}"
            enriched = reconstruct_question(q_data, config)
            options_in_spans = [s["text"] for s in enriched["spans"] if s["label"] == "option_text"]
            if options_in_spans != q_data["options"]:
                shuffled = True
                new_ans_letter = enriched.get("answer", "")
                opt_letters = ["A", "B", "C", "D"]
                if new_ans_letter in opt_letters:
                    new_idx = opt_letters.index(new_ans_letter)
                    self.assertEqual(options_in_spans[new_idx], "Đáp án A (Đúng)")
                break
                
    def test_collapse_consecutive_whitespaces(self):
        from synthetic_exam_generator.reconstructor import collapse_consecutive_whitespaces
        raw_text = "Câu 1:    Cho hàm số $y=f(x)$.\t\tA. 1   B. 2\t\t\tC. 3  D. 4"
        spans = [
            {"start": 0, "end": 7, "label": "question_label", "text": "Câu 1: "},
            {"start": 10, "end": 30, "label": "stem", "text": "Cho hàm số $y=f(x)$."},
            {"start": 32, "end": 35, "label": "option_label", "text": "A. "},
            {"start": 35, "end": 36, "label": "option_text", "text": "1"},
            {"start": 39, "end": 42, "label": "option_label", "text": "B. "},
            {"start": 42, "end": 43, "label": "option_text", "text": "2"},
            {"start": 46, "end": 49, "label": "option_label", "text": "C. "},
            {"start": 49, "end": 50, "label": "option_text", "text": "3"},
            {"start": 52, "end": 55, "label": "option_label", "text": "D. "},
            {"start": 55, "end": 56, "label": "option_text", "text": "4"}
        ]
        collapsed_text, updated_spans = collapse_consecutive_whitespaces(raw_text, spans)
        
        # Verify no consecutive spaces or tabs exist
        self.assertNotIn("  ", collapsed_text)
        self.assertNotIn("\t", collapsed_text)
        
        # Verify span offsets exactly index the correct text
        for s in updated_spans:
            sub = collapsed_text[s["start"]:s["end"]].strip()
            self.assertEqual(sub, s["text"].strip())

    def test_reconstruct_with_collapse_whitespace(self):
        q_data = {
            "is_group": False,
            "stem": "Câu hỏi với khoảng trắng rộng.",
            "options": ["Đáp án 1", "Đáp án 2", "Đáp án 3", "Đáp án 4"],
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            collapse_whitespace_prob=1.0,
            inline_option_prob=1.0,
            seed="collapse_seed"
        )
        enriched = reconstruct_question(q_data, config)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        self.assertNotIn("  ", raw_text)
        self.assertNotIn("\t", raw_text)
        for s in spans:
            self.assertEqual(raw_text[s["start"]:s["end"]].strip(), s["text"].strip())

    def test_grid_2x2_layout(self):
        q_data = {
            "is_group": False,
            "stem": "Cho hình lập phương ABCD.A'B'C'D'.",
            "options": ["10 cm", "20 cm", "30 cm", "40 cm"],
            "question_type": "multiple_choice",
            "subject": "math_geometry",
            "grade": 12,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            grid_2x2_prob=1.0,
            option_prefix_style="capital_dot",
            seed="grid_seed",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        # In 2x2 grid, there should be a newline between B and C
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        self.assertEqual(len(lines), 3) # Line 1: stem, Line 2: A & B, Line 3: C & D
        self.assertIn("A.", lines[1])
        self.assertIn("B.", lines[1])
        self.assertIn("C.", lines[2])
        self.assertIn("D.", lines[2])
        
        for s in spans:
            self.assertEqual(raw_text[s["start"]:s["end"]].strip(), s["text"].strip())

    def test_flatten_newlines(self):
        q_data = {
            "is_group": False,
            "stem": "Dòng 1 stem\nDòng 2 stem",
            "options": ["A1\nA2", "B1", "C1", "D1"],
            "question_type": "multiple_choice",
            "subject": "physics",
            "grade": 11,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            flatten_newlines_prob=1.0,
            seed="flatten_seed"
        )
        enriched = reconstruct_question(q_data, config)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        self.assertNotIn("\n", raw_text)
        for s in spans:
            self.assertEqual(raw_text[s["start"]:s["end"]].strip(), s["text"].strip())

    def test_math_intervals_latex_validation(self):
        from synthetic_exam_generator.latex_utils import is_valid_latex
        # Half-open intervals
        self.assertTrue(is_valid_latex("(-\\infty; 0]"))
        self.assertTrue(is_valid_latex("[8; +\\infty)"))
        self.assertTrue(is_valid_latex("(1; 2]"))
        self.assertTrue(is_valid_latex("[0; 1)"))
        self.assertTrue(is_valid_latex("(-\\infty, 5]"))
        # Invalid / plain words
        self.assertFalse(is_valid_latex("(đây là khoảng)"))

    def test_inline_explanation(self):
        q_data = {
            "is_group": False,
            "stem": "Tính giá trị tích phân $I$.",
            "options": ["1", "2", "3", "4"],
            "answer": "B",
            "explanation": "Ta có $I = \\int_0^1 2x dx = 1$. Do đó chọn B.",
            "question_type": "multiple_choice",
            "subject": "math_algebra",
            "grade": 12,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            inline_explanation=True,
            randomize_q_num=False,
            seed="expl_seed"
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        expl_spans = [s for s in spans if s["label"] == "explanation"]
        self.assertEqual(len(expl_spans), 1)
        self.assertIn("Do đó chọn B.", expl_spans[0]["text"])
        for s in spans:
            self.assertEqual(raw_text[s["start"]:s["end"]], s["text"])

    def test_reconstruct_exam_barem_modes(self):
        from synthetic_exam_generator.reconstructor import reconstruct_exam
        exam_data = {
            "subject": "chemistry",
            "grade": 12,
            "sections": {
                "Phần 1. Trắc nghiệm": [
                    {
                        "is_group": False,
                        "stem": "Công thức của glucozơ là",
                        "options": ["C6H12O6", "C12H22O11", "C2H5OH", "CH3COOH"],
                        "answer": "A",
                        "explanation": "Glucozơ có công thức phân tử là C6H12O6.",
                        "question_type": "multiple_choice"
                    },
                    {
                        "is_group": False,
                        "stem": "Chất nào sau đây là ancol?",
                        "options": ["C2H5OH", "CH3CHO", "CH3COOH", "HCOOCH3"],
                        "answer": "A",
                        "explanation": "C2H5OH là ancol etylic.",
                        "question_type": "multiple_choice"
                    }
                ]
            }
        }

        # 1. Mode 0: Pure Exam (prob_no_barem=1.0)
        c0 = ReconstructorConfig(prob_no_barem=1.0, prob_inline_barem=0, prob_answer_grid=0, prob_end_section=0, prob_table_barem=0, seed="m0")
        res0 = reconstruct_exam(exam_data, c0)
        labels0 = [s["label"] for s in res0["spans"]]
        self.assertNotIn("explanation", labels0)

        # 2. Mode 1: Inline Explanation (prob_inline_barem=1.0)
        c1 = ReconstructorConfig(prob_no_barem=0, prob_inline_barem=1.0, prob_answer_grid=0, prob_end_section=0, prob_table_barem=0, seed="m1")
        res1 = reconstruct_exam(exam_data, c1)
        labels1 = [s["label"] for s in res1["spans"]]
        self.assertIn("explanation", labels1)

        # 3. Mode 2: End-of-exam Section (prob_end_section=1.0)
        c2 = ReconstructorConfig(prob_no_barem=0, prob_inline_barem=0, prob_answer_grid=0, prob_end_section=1.0, prob_table_barem=0, seed="m2")
        res2 = reconstruct_exam(exam_data, c2)
        labels2 = [s["label"] for s in res2["spans"]]
        self.assertIn("explanation", labels2)
        self.assertIn("ĐÁP ÁN", res2["raw_text"])

        # 4. Mode 3: Table Barem (prob_table_barem=1.0)
        c3 = ReconstructorConfig(prob_no_barem=0, prob_inline_barem=0, prob_answer_grid=0, prob_end_section=0, prob_table_barem=1.0, seed="m3")
        res3 = reconstruct_exam(exam_data, c3)
        labels3 = [s["label"] for s in res3["spans"]]
        self.assertIn("explanation", labels3)
        self.assertIn("| Câu |", res3["raw_text"])

        # 5. Mode 4: Answer Grid (prob_answer_grid=1.0)
        c4 = ReconstructorConfig(prob_no_barem=0, prob_inline_barem=0, prob_answer_grid=1.0, prob_end_section=0, prob_table_barem=0, seed="m4")
        res4 = reconstruct_exam(exam_data, c4)
        labels4 = [s["label"] for s in res4["spans"]]
        self.assertIn("explanation", labels4)
        self.assertIn("BẢNG ĐÁP ÁN", res4["raw_text"])

        # Verify all spans in all modes match raw_text slices
        for res in [res0, res1, res2, res3, res4]:
            for s in res["spans"]:
                self.assertEqual(res["raw_text"][s["start"]:s["end"]], s["text"])

    def test_group_question_permutation_answer_remapping(self):
        q_group = {
            "is_group": True,
            "stimulus": "Đọc đoạn văn sau và trả lời câu hỏi.",
            "subject": "literature",
            "grade": 12,
            "questions": [
                {
                    "stem": "Nội dung chính là gì?",
                    "options": ["Đáp án đúng", "Sai 1", "Sai 2", "Sai 3"],
                    "answer": "A"
                }
            ]
        }
        # Run with permutations enabled
        config = ReconstructorConfig(
            enable_permutations=True,
            seed="perm_seed_123"
        )
        res = reconstruct_question(q_group, config)
        self.assertIn("questions", res)
        perm_sub_q = res["questions"][0]
        # The correct option text is 'Đáp án đúng'
        opt_letters = ["A", "B", "C", "D"]
        correct_idx = perm_sub_q["options"].index("Đáp án đúng")
        expected_ans = opt_letters[correct_idx]
        self.assertEqual(perm_sub_q["answer"], expected_ans)

    def test_exam_answer_grid_matches_permuted_answers(self):
        exam_data = {
            "subject": "chemistry",
            "grade": 12,
            "sections": {
                "Phần 1. Trắc nghiệm": [
                    {
                        "is_group": False,
                        "stem": "Công thức của glucozơ là",
                        "options": ["C6H12O6", "C12H22O11", "C2H5OH", "CH3COOH"],
                        "answer": "A",
                        "question_type": "multiple_choice"
                    }
                ]
            }
        }
        config = ReconstructorConfig(
            enable_permutations=True,
            prob_answer_grid=1.0,
            prob_no_barem=0,
            prob_inline_barem=0,
            prob_end_section=0,
            prob_table_barem=0,
            seed="perm_exam_seed_456"
        )
        res = reconstruct_exam(exam_data, config)
        # Find reconstructed option matching C6H12O6
        raw_text = res["raw_text"]
        for letter in ["A", "B", "C", "D"]:
            if f"{letter}. C6H12O6" in raw_text:
                # Answer grid should indicate this letter
                self.assertIn(f"1 | {letter}", raw_text)

    def test_reconstruct_exam_flatten_newlines(self):
        from synthetic_exam_generator.reconstructor import reconstruct_exam
        exam_data = {
            "subject": "physics",
            "grade": 11,
            "sections": {
                "Phần 1. Trắc nghiệm": [
                    {
                        "is_group": False,
                        "stem": "Một vật dao động điều hòa.\nChu kì dao động là T.",
                        "options": ["A. 1s", "B. 2s", "C. 3s", "D. 4s"],
                        "answer": "A",
                        "question_type": "multiple_choice"
                    },
                    {
                        "is_group": False,
                        "stem": "Tần số dao động f bằng gì?",
                        "options": ["A. 1/T", "B. 2/T", "C. T", "D. 2T"],
                        "answer": "A",
                        "question_type": "multiple_choice"
                    }
                ]
            }
        }
        config = ReconstructorConfig(
            flatten_newlines_prob=1.0,
            prob_no_barem=1.0,
            prob_inline_barem=0,
            prob_answer_grid=0,
            prob_end_section=0,
            prob_table_barem=0,
            seed="exam_flatten_123"
        )
        res = reconstruct_exam(exam_data, config)
        raw_text = res["raw_text"]
        spans = res["spans"]

        self.assertNotIn("\n", raw_text)
        for s in spans:
            self.assertEqual(raw_text[s["start"]:s["end"]].strip(), s["text"].strip())

    def test_reconstruct_exam_collapse_whitespace(self):
        from synthetic_exam_generator.reconstructor import reconstruct_exam
        exam_data = {
            "subject": "math",
            "grade": 12,
            "sections": {
                "Phần 1. Trắc nghiệm": [
                    {
                        "is_group": False,
                        "stem": "Tìm tập xác định của hàm số.",
                        "options": ["D = R", "D = R \\ {0}", "D = [0; 1]", "D = (0; +inf)"],
                        "answer": "A",
                        "question_type": "multiple_choice"
                    }
                ]
            }
        }
        config = ReconstructorConfig(
            collapse_whitespace_prob=1.0,
            min_inline_spaces=5,
            max_inline_spaces=10,
            inline_option_prob=1.0,
            prob_no_barem=1.0,
            prob_inline_barem=0,
            prob_answer_grid=0,
            prob_end_section=0,
            prob_table_barem=0,
            seed="exam_collapse_456"
        )
        res = reconstruct_exam(exam_data, config)
        raw_text = res["raw_text"]
        spans = res["spans"]

        self.assertNotIn("  ", raw_text)
        self.assertNotIn("\t", raw_text)
        for s in spans:
            self.assertEqual(raw_text[s["start"]:s["end"]].strip(), s["text"].strip())

if __name__ == '__main__':
    unittest.main()
