import unittest
from synthetic_exam_generator.parser import check_and_clean_options, parse_question_xml

class TestParser(unittest.TestCase):
    def test_check_and_clean_options_standard(self):
        # Standard A., B., C., D. options
        options = [
            "A. Phản ứng tỏa nhiệt",
            "B. Phản ứng thu nhiệt",
            "C. Phản ứng tỏa nhiệt khác",
            "D. Phản ứng thu nhiệt khác"
        ]
        expected = [
            "Phản ứng tỏa nhiệt",
            "Phản ứng thu nhiệt",
            "Phản ứng tỏa nhiệt khác",
            "Phản ứng thu nhiệt khác"
        ]
        self.assertEqual(check_and_clean_options(options), expected)

    def test_check_and_clean_options_lowercase(self):
        # Lowercase a., b., c., d. options
        options = [
            "a) Phản ứng tỏa nhiệt",
            "b) Phản ứng thu nhiệt",
            "c) Phản ứng tỏa nhiệt khác",
            "d) Phản ứng thu nhiệt khác"
        ]
        expected = [
            "Phản ứng tỏa nhiệt",
            "Phản ứng thu nhiệt",
            "Phản ứng tỏa nhiệt khác",
            "Phản ứng thu nhiệt khác"
        ]
        self.assertEqual(check_and_clean_options(options), expected)

    def test_check_and_clean_options_no_labels(self):
        # Options without any labels
        options = [
            "Phản ứng tỏa nhiệt",
            "Phản ứng thu nhiệt",
            "Phản ứng tỏa nhiệt khác",
            "Phản ứng thu nhiệt khác"
        ]
        self.assertEqual(check_and_clean_options(options), options)

    def test_check_and_clean_options_mismatched_labels(self):
        # Options where one of the labels doesn't match the index
        options = [
            "A. Phản ứng tỏa nhiệt",
            "B. Phản ứng thu nhiệt",
            "C. Phản ứng tỏa nhiệt khác",
            "E. Phản ứng thu nhiệt khác" # 'E' instead of 'D'
        ]
        self.assertEqual(check_and_clean_options(options), options)

    def test_parse_question_xml_with_prefixes(self):
        # Test full XML parsing with prefixes
        xml = """
        <question>
          <stem>Câu hỏi stem</stem>
          <option>A. Phương án 1</option>
          <option>B. Phương án 2</option>
          <option>C. Phương án 3</option>
          <option>D. Phương án 4</option>
          <answer>A</answer>
          <explanation>Giải thích</explanation>
        </question>
        """
        parsed = parse_question_xml(xml)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["options"], ["Phương án 1", "Phương án 2", "Phương án 3", "Phương án 4"])

if __name__ == '__main__':
    unittest.main()
