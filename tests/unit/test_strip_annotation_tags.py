import unittest
import tempfile
import shutil
from pathlib import Path

from tools.annotation.strip_annotation_tags import (
    strip_tags_from_text,
    verify_no_sequence_tags,
    process_directory,
)


class TestStripAnnotationTags(unittest.TestCase):

    def setUp(self):
        self.sample_annotated_text = """<pages>
<page>
<other>
SỞ GD&ĐT NGHỆ AN
TRƯỜNG THPT ĐÔ LƯƠNG 3
ĐỀ THI GIỮA HỌC KỲ II
</other>

<question_label>## Câu 1.</question_label>
<stem>
Cho hàm số $y=\\dfrac{2x-1}{x+1}$. Giá trị $f'(3)$ bằng
</stem>
<option_label>A.</option_label> <option_text>$\\dfrac{4}{5}$.</option_text>
<option_label>B.</option_label> <option_text>$\\dfrac{3}{16}$.</option_text>
<option_label>C.</option_label> <option_text>$\\dfrac{2}{9}$.</option_text>
<option_label>D.</option_label> <option_text>$\\dfrac{1}{6}$.</option_text>

<question_label>## Câu 2.</question_label>
<stem>
Cho bảng số liệu sau:
<table>
<tr><th>Tên</th><th>Điểm</th></tr>
<tr><td>An</td><td>10</td></tr>
</table>
<figure id="fig-p1-1" bbox="10,20,100,200" description="Đồ thị hàm số bậc hai parabol" />
Khẳng định nào đúng?
</stem>
<option_label>A.</option_label> <option_text>Điểm An cao nhất.</option_text>
<option_label>B.</option_label> <option_text>Điểm An thấp nhất.</option_text>

<page_metadata>
{"p": 1, "w": 612, "h": 792}
</page_metadata>
</page>
</pages>
"""

    def test_strip_tags_removes_all_sequence_tags(self):
        cleaned = strip_tags_from_text(self.sample_annotated_text)
        residual = verify_no_sequence_tags(cleaned)
        self.assertEqual(residual, [])

        # Verify specific tags are absent
        for tag in ["<question_label>", "</question_label>", "<stem>", "</stem>",
                    "<option_label>", "</option_label>", "<option_text>", "</option_text>",
                    "<other>", "</other>"]:
            self.assertNotIn(tag, cleaned)

    def test_strip_tags_preserves_text_and_math(self):
        cleaned = strip_tags_from_text(self.sample_annotated_text)

        # Content must remain intact
        self.assertIn("SỞ GD&ĐT NGHỆ AN", cleaned)
        self.assertIn("TRƯỜNG THPT ĐÔ LƯƠNG 3", cleaned)
        self.assertIn("## Câu 1.", cleaned)
        self.assertIn("Cho hàm số $y=\\dfrac{2x-1}{x+1}$. Giá trị $f'(3)$ bằng", cleaned)
        self.assertIn("A. $\\dfrac{4}{5}$.", cleaned)
        self.assertIn("B. $\\dfrac{3}{16}$.", cleaned)

        # LaTeX formula preserved
        self.assertIn("$y=\\dfrac{2x-1}{x+1}$", cleaned)

    def test_strip_tags_preserves_structural_tags(self):
        cleaned = strip_tags_from_text(self.sample_annotated_text)

        # HTML table preserved
        self.assertIn("<table>", cleaned)
        self.assertIn("<tr><th>Tên</th><th>Điểm</th></tr>", cleaned)
        self.assertIn("</table>", cleaned)

        # Figure preserved
        self.assertIn('<figure id="fig-p1-1" bbox="10,20,100,200" description="Đồ thị hàm số bậc hai parabol" />', cleaned)

        # Page envelope & metadata preserved
        self.assertIn("<pages>", cleaned)
        self.assertIn("<page>", cleaned)
        self.assertIn("<page_metadata>", cleaned)
        self.assertIn('{"p": 1, "w": 612, "h": 792}', cleaned)
        self.assertIn("</page>", cleaned)
        self.assertIn("</pages>", cleaned)

    def test_process_directory_end_to_end(self):
        tmp_in = Path(tempfile.mkdtemp())
        tmp_out = Path(tempfile.mkdtemp())

        try:
            (tmp_in / "doc_001.md").write_text(self.sample_annotated_text, encoding="utf-8")
            (tmp_in / "doc_002.md").write_text(self.sample_annotated_text, encoding="utf-8")

            summary = process_directory(tmp_in, tmp_out)

            self.assertEqual(summary["total_source_files"], 2)
            self.assertEqual(summary["successfully_stripped"], 2)
            self.assertEqual(summary["failed_files"], 0)
            self.assertTrue(summary["all_passed"])

            out_files = list(tmp_out.glob("*.md"))
            self.assertEqual(len(out_files), 2)

            for f in out_files:
                content = f.read_text(encoding="utf-8")
                self.assertEqual(verify_no_sequence_tags(content), [])
                self.assertIn("Cho hàm số", content)
                self.assertIn("<table>", content)

        finally:
            shutil.rmtree(tmp_in)
            shutil.rmtree(tmp_out)


if __name__ == "__main__":
    unittest.main()
