import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from tools.experiments.hybrid_digital_ocr import HybridDigitalOCR


class TestHybridDigitalOCR(unittest.TestCase):
    def test_is_digital_pdf(self):
        # Test digital detection on existing test PDF
        self.assertTrue(HybridDigitalOCR.is_digital_pdf("data/sequence_labelling_input_data_raw/exam-001--8982d5d00fcd.pdf"))

    def test_stitch_document(self):
        engine = HybridDigitalOCR()
        template = "Câu 1: Cho hàm số {{FORMULA_F1_1}}. Giá trị {{FORMULA_F1_2}} bằng 2.\n\n{{TABLE_T1_1}}"
        formula_map = {
            "F1_1": "$y = f(x)$",
            "F1_2": "$f'(1)$",
        }
        table_map = {
            "T1_1": "<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>",
        }

        result = engine.stitch_document(template, formula_map, table_map, page_num=1)
        self.assertIn("$y = f(x)$", result)
        self.assertIn("$f'(1)$", result)
        self.assertIn("<table>", result)
        self.assertIn("<page_metadata>", result)
        self.assertNotIn("{{FORMULA_", result)
        self.assertNotIn("{{TABLE_", result)

    @patch("tools.experiments.hybrid_digital_ocr.chat_with_agy")
    def test_transcribe_components(self, mock_chat_agy):
        mock_chat_agy.return_value = (
            '{"F1_1": "$x^2 + 1$"}',
            "conv-123",
            {"input_tokens": 100, "output_tokens": 10},
        )

        engine = HybridDigitalOCR()
        crop = np.ones((30, 80, 3), dtype=np.uint8) * 255
        formula_items = [("F1_1", crop)]

        f_map, t_map, usage = engine.transcribe_components(formula_items, table_items=[])
        self.assertIn("F1_1", f_map)
        self.assertEqual(f_map["F1_1"], "$x^2 + 1$")
        self.assertEqual(usage["calls"], 1)


if __name__ == "__main__":
    unittest.main()
