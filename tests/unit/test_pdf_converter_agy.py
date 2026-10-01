import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sequence_labelling.annotator.pdf_converter import (
    PDFOCRConverter,
    prune_code_blocks,
    prune_think_tags,
)


class TestPDFConverterAGY(unittest.TestCase):
    def test_prune_code_blocks(self):
        text = "```xml\n<pages><page>Content</page></pages>\n```"
        self.assertEqual(prune_code_blocks(text), "<pages><page>Content</page></pages>")

        text_md = "```markdown\n# Header\n```"
        self.assertEqual(prune_code_blocks(text_md), "# Header")

        clean = "<pages><page>Clean</page></pages>"
        self.assertEqual(prune_code_blocks(clean), clean)

    def test_init_with_agy_provider(self):
        converter = PDFOCRConverter(
            provider="agy",
            model="gemini-3.8-flash",
            concurrency=5,  # Should be capped at 2 for local CLI
        )
        self.assertEqual(converter.provider, "agy")
        self.assertEqual(converter.model, "gemini-3.8-flash")
        self.assertEqual(converter.concurrency, 2)
        self.assertIsNone(converter.client)

    def test_init_default_agy_model(self):
        converter = PDFOCRConverter(provider="agy", model=None)
        self.assertEqual(converter.provider, "agy")
        self.assertTrue("gemini" in converter.model)

    @patch("sequence_labelling.annotator.pdf_converter.fitz.open")
    @patch("sequence_labelling.annotator.pdf_converter.chat_with_agy")
    def test_convert_pdf_routes_to_chat_with_agy(self, mock_chat_agy, mock_fitz_open):
        mock_chat_agy.return_value = (
            "<pages><page>Page 1 Text<page_metadata>{\"p\": 1}</page_metadata></page></pages>",
            "conv-mock-123",
            {"input_tokens": 1000, "output_tokens": 50},
        )

        # Mock PyMuPDF doc and page
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # fake JPEG header
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        converter = PDFOCRConverter(
            provider="agy",
            model="gemini-3.8-flash",
            enable_figure_detection=False,
        )

        with patch.object(Path, "exists", return_value=True):
            result = converter.convert_pdf("dummy.pdf")

        self.assertIn("Page 1 Text", result)
        mock_chat_agy.assert_called_once()
        args, kwargs = mock_chat_agy.call_args
        self.assertEqual(kwargs.get("model"), "gemini-3.8-flash")
        images = kwargs.get("images")
        self.assertIsNotNone(images)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0][0], "page_1.jpg")
        self.assertEqual(images[0][1], b"\xff\xd8\xff\xe0\x00\x10JFIF")

    @patch("sequence_labelling.annotator.pdf_converter.fitz.open")
    def test_convert_pdf_openai_provider_builds_messages(self, mock_fitz_open):
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="<pages><page>OpenAI Page<page_metadata>{\"p\": 1}</page_metadata></page></pages>"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        converter = PDFOCRConverter(
            provider="codex",
            api_key="mock-key",
            enable_figure_detection=False,
        )
        converter.client = mock_client

        with patch.object(Path, "exists", return_value=True):
            result = converter.convert_pdf("dummy.pdf")

        self.assertIn("OpenAI Page", result)
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertIn("messages", call_kwargs)
        messages = call_kwargs["messages"]
        self.assertTrue(len(messages) >= 1)
        self.assertEqual(messages[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
