import unittest
import random
import re
from tools.build_training_dataset import augment_real_document


class TestRealDataAugmentation(unittest.TestCase):

    def test_tab_to_space_normalization(self):
        raw_text = "Câu 1:\tMột vật dao động\tđiều hòa.\tA. 1s\tB. 2s"
        spans = [
            {"start": 0, "end": 7, "label": "question_label", "text": "Câu 1:"},
            {"start": 8, "end": 32, "label": "stem", "text": "Một vật dao động\tđiều hòa."},
            {"start": 33, "end": 39, "label": "option_label", "text": "A. 1s"},
            {"start": 40, "end": 46, "label": "option_label", "text": "B. 2s"},
        ]
        rng = random.Random(42)
        res = augment_real_document(
            raw_text,
            spans,
            rng,
            flatten_newlines_prob=0.0,
            newline_to_space_prob=0.0,
            tab_to_space=True
        )
        self.assertIsNotNone(res)
        aug_text, aug_spans = res

        # Verify all tabs have been replaced by single space
        self.assertNotIn("\t", aug_text)
        self.assertIn("Câu 1:", aug_text)
        self.assertIn("Một vật dao động điều hòa.", aug_text)

        # Verify exact span offset alignment
        for s in aug_spans:
            sub = aug_text[s["start"]:s["end"]].strip()
            self.assertEqual(sub, s["text"].strip())

    def test_flatten_newlines_to_space(self):
        raw_text = "Câu 1:\nMột vật dao động.\nChu kì T = 2s.\nA. 1s\nB. 2s\nC. 3s\nD. 4s"
        spans = [
            {"start": 0, "end": 6, "label": "question_label", "text": "Câu 1:"},
            {"start": 7, "end": 40, "label": "stem", "text": "Một vật dao động.\nChu kì T = 2s."},
            {"start": 41, "end": 46, "label": "option_label", "text": "A. 1s"},
            {"start": 47, "end": 52, "label": "option_label", "text": "B. 2s"},
            {"start": 53, "end": 58, "label": "option_label", "text": "C. 3s"},
            {"start": 59, "end": 64, "label": "option_label", "text": "D. 4s"},
        ]
        rng = random.Random(123)
        res = augment_real_document(
            raw_text,
            spans,
            rng,
            flatten_newlines_prob=1.0,
            newline_to_space_prob=0.0,
            tab_to_space=True
        )
        self.assertIsNotNone(res)
        aug_text, aug_spans = res

        # In full flatten mode, no newlines should remain
        self.assertNotIn("\n", aug_text)
        self.assertNotIn("\r", aug_text)

        # Spans must match the transformed text exactly
        for s in aug_spans:
            sub = aug_text[s["start"]:s["end"]].strip()
            self.assertTrue(len(sub) > 0)
            self.assertEqual(sub, s["text"].strip())

    def test_markdown_table_protection_during_flattening(self):
        raw_text = (
            "Câu 2: Cho bảng số liệu sau:\n"
            "| Thời gian | Vận tốc |\n"
            "|---|---|\n"
            "| 1s | 5 m/s |\n"
            "| 2s | 10 m/s |\n"
            "Gia tốc của vật là bao nhiêu?\n"
            "A. 5 m/s2\nB. 10 m/s2"
        )
        spans = [
            {"start": 0, "end": 6, "label": "question_label", "text": "Câu 2:"},
            {"start": 7, "end": 125, "label": "stem", "text": raw_text[7:125]},
            {"start": 126, "end": 135, "label": "option_label", "text": "A. 5 m/s2"},
            {"start": 136, "end": 146, "label": "option_label", "text": "B. 10 m/s2"}
        ]
        rng = random.Random(999)
        res = augment_real_document(
            raw_text,
            spans,
            rng,
            flatten_newlines_prob=1.0, # Flattening requested
            tab_to_space=True
        )
        self.assertIsNotNone(res)
        aug_text, aug_spans = res

        # Verify table structure is intact (lines of the table separated by newlines)
        self.assertIn("| Thời gian | Vận tốc |\n|---|---|\n| 1s | 5 m/s |\n| 2s | 10 m/s |", aug_text)

        # Verify all spans still index valid slices
        for s in aug_spans:
            sub = aug_text[s["start"]:s["end"]].strip()
            self.assertEqual(sub, s["text"].strip())

    def test_latex_math_protection_and_masking(self):
        raw_text = "Câu 3: Tính giá trị của tích phân $I = \\int_{0}^{1} x^2 dx$.\nA. 1/3\nB. 1/2"
        spans = [
            {"start": 0, "end": 6, "label": "question_label", "text": "Câu 3:"},
            {"start": 7, "end": 60, "label": "stem", "text": "Tính giá trị của tích phân $I = \\int_{0}^{1} x^2 dx$."},
            {"start": 61, "end": 67, "label": "option_label", "text": "A. 1/3"},
            {"start": 68, "end": 74, "label": "option_label", "text": "B. 1/2"}
        ]
        # Test 1: Math preserved
        rng1 = random.Random(10)
        res1 = augment_real_document(
            raw_text,
            spans,
            rng1,
            latex_mask_prob=0.0,
            latex_vary_delimiters_prob=0.0
        )
        self.assertIsNotNone(res1)
        aug_text1, _ = res1
        self.assertIn("$I = \\int_{0}^{1} x^2 dx$", aug_text1)

        # Test 2: Math masked
        rng2 = random.Random(10)
        res2 = augment_real_document(
            raw_text,
            spans,
            rng2,
            latex_mask_prob=1.0,
            latex_placeholder="[LATEX]"
        )
        self.assertIsNotNone(res2)
        aug_text2, aug_spans2 = res2
        self.assertIn("[LATEX]", aug_text2)
        for s in aug_spans2:
            sub = aug_text2[s["start"]:s["end"]].strip()
            self.assertEqual(sub, s["text"].strip())

    def test_stochastic_jitter_span_invariance(self):
        raw_text = (
            "Câu 10: Cho hàm số f(x).   Mệnh đề nào đúng?\n\n"
            "A. Hàm số đồng biến trên R.\t\t"
            "B. Hàm số nghịch biến trên R.\n"
            "C. Hàm số có cực trị.\t"
            "D. Không có cực trị."
        )
        spans = [
            {"start": 0, "end": 8, "label": "question_label", "text": "Câu 10:"},
            {"start": 9, "end": 44, "label": "stem", "text": "Cho hàm số f(x).   Mệnh đề nào đúng?"},
            {"start": 46, "end": 48, "label": "option_label", "text": "A."},
            {"start": 49, "end": 75, "label": "option_text", "text": "Hàm số đồng biến trên R."},
            {"start": 77, "end": 79, "label": "option_label", "text": "B."},
            {"start": 80, "end": 107, "label": "option_text", "text": "Hàm số nghịch biến trên R."},
        ]

        # Run 50 random iterations across all modes
        for seed in range(50):
            rng = random.Random(seed)
            res = augment_real_document(
                raw_text,
                spans,
                rng,
                flatten_newlines_prob=0.3,
                newline_to_space_prob=0.3,
                tab_to_space=True
            )
            self.assertIsNotNone(res)
            aug_text, aug_spans = res
            for s in aug_spans:
                sub = aug_text[s["start"]:s["end"]].strip()
                self.assertEqual(sub, s["text"].strip())


if __name__ == "__main__":
    unittest.main()
