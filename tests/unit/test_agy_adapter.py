import json
import unittest
from unittest.mock import MagicMock, patch

from sequence_labelling.llm.agy_adapter import (
    format_messages_to_prompt,
    _clean_response,
    _parse_agy_json_output,
    chat_with_agy,
)
from sequence_labelling.llm.deepseek_client import chat


class TestAgyAdapter(unittest.TestCase):
    def test_format_messages_to_prompt(self):
        messages = [
            {"role": "system", "content": "You are a test system."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        sys_text, user_text = format_messages_to_prompt(messages)
        self.assertEqual(sys_text, "You are a test system.")
        self.assertIn("Question 1", user_text)
        self.assertIn("Assistant: Answer 1", user_text)
        self.assertIn("Question 2", user_text)

    def test_clean_response(self):
        raw = "<think>Some reasoning</think><<<START>>>Result content<<<END>>>"
        cleaned = _clean_response(raw)
        self.assertEqual(cleaned, "Result content")

    def test_parse_agy_json_output(self):
        sample_json = json.dumps({
            "conversation_id": "conv-test-123",
            "status": "SUCCESS",
            "response": "Hello world\n",
            "usage": {"input_tokens": 100, "output_tokens": 5}
        })
        text, conv_id, usage = _parse_agy_json_output(sample_json)
        self.assertEqual(text, "Hello world")
        self.assertEqual(conv_id, "conv-test-123")
        self.assertEqual(usage.get("input_tokens"), 100)

    @patch("sequence_labelling.llm.deepseek_client.chat_with_agy")
    def test_chat_routes_to_agy(self, mock_chat_agy):
        mock_chat_agy.return_value = ("Parsed questions XML", "conv-mock-id", {"tokens": 50})

        res = chat(
            prompt="Câu 1: 1+1=? A. 1 B. 2",
            provider="agy",
            model="gemini-3.8-flash-high",
        )
        self.assertEqual(res, "Parsed questions XML")
        mock_chat_agy.assert_called_once()

    def test_resolve_model_and_effort(self):
        from sequence_labelling.llm.agy_adapter import _resolve_model_and_effort

        # Case 1: Model with -high suffix
        m, e = _resolve_model_and_effort("gemini-3.8-flash-high", None)
        self.assertEqual(m, "gemini-3.8-flash-high")
        self.assertIsNone(e)

        # Case 2: Base model with explicit thinking
        m, e = _resolve_model_and_effort("gemini-3.8-flash", "medium")
        self.assertEqual(m, "gemini-3.8-flash-medium")
        self.assertIsNone(e)

        # Case 3: Base model without thinking defaults to high for gemini
        m, e = _resolve_model_and_effort("gemini-3.8-flash", None)
        self.assertEqual(m, "gemini-3.8-flash-high")
        self.assertIsNone(e)

    @patch("sequence_labelling.llm.agy_adapter.subprocess.run")
    def test_chat_with_agy_multi_image_sandbox(self, mock_subproc):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "conversation_id": "conv-vision-1",
            "status": "SUCCESS",
            "response": "Vision OCR Markdown Result",
            "usage": {"input_tokens": 200, "output_tokens": 50},
        })
        mock_subproc.return_value = mock_proc

        images = [
            ("page_1.jpg", b"img_bytes_1"),
            ("page_2.jpg", b"img_bytes_2"),
        ]
        text, conv_id, usage = chat_with_agy(
            prompt="Perform OCR on pages",
            model="gemini-3.8-flash",
            images=images,
        )

        self.assertEqual(text, "Vision OCR Markdown Result")
        self.assertEqual(conv_id, "conv-vision-1")
        cmd = mock_subproc.call_args[0][0]
        self.assertTrue(any("agy" in arg for arg in cmd))
        self.assertIn("--sandbox", cmd)
        self.assertIn("--agent", cmd)
        self.assertIn("vision-ocr", cmd)
        call_kwargs = mock_subproc.call_args[1]
        vision_input = call_kwargs.get("input", "")
        self.assertIn("'page_1.jpg'", vision_input)
        self.assertIn("'page_2.jpg'", vision_input)

    @patch("sequence_labelling.llm.agy_adapter.subprocess.run")
    def test_chat_with_agy_text_mode_agent(self, mock_subproc):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "conversation_id": "conv-text-1",
            "status": "SUCCESS",
            "response": "<question_label>Câu 1.</question_label>",
            "usage": {"input_tokens": 50, "output_tokens": 10},
        })
        mock_subproc.return_value = mock_proc

        text, conv_id, usage = chat_with_agy(
            prompt="Annotate this question",
            model="gemini-3.8-flash",
        )

        self.assertEqual(text, "<question_label>Câu 1.</question_label>")
        cmd = mock_subproc.call_args[0][0]
        self.assertIn("--agent", cmd)
        self.assertIn("text-only", cmd)
        self.assertIn("--disable-slash-commands", cmd)


if __name__ == "__main__":
    unittest.main()

