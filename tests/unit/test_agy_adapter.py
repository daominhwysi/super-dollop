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
        self.assertEqual(m, "gemini-3.8-flash")
        self.assertEqual(e, "high")

        # Case 2: Base model with explicit thinking
        m, e = _resolve_model_and_effort("gemini-3.8-flash", "medium")
        self.assertEqual(m, "gemini-3.8-flash")
        self.assertEqual(e, "medium")

        # Case 3: Base model without thinking defaults to high for gemini
        m, e = _resolve_model_and_effort("gemini-3.8-flash", None)
        self.assertEqual(m, "gemini-3.8-flash")
        self.assertEqual(e, "high")


if __name__ == "__main__":
    unittest.main()
