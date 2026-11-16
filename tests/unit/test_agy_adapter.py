import unittest
from unittest.mock import MagicMock, patch

from sequence_labelling.llm.agy_adapter import DEFAULT_AGY_MODEL, chat_with_agy
from sequence_labelling.llm.deepseek_client import chat


class TestAgyAdapter(unittest.TestCase):
    @patch("sequence_labelling.llm.deepseek_client.client")
    def test_chat_routes_to_agy(self, mock_client):
        mock_msg = MagicMock()
        mock_msg.content = "Parsed questions XML"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        res = chat(
            prompt="Câu 1: 1+1=? A. 1 B. 2",
            provider="agy",
            model="gemini-3.8-flash-high",
        )
        self.assertEqual(res, "Parsed questions XML")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.8-flash-high")
        self.assertEqual(call_kwargs["extra_body"]["provider"], "agy")

    @patch("sequence_labelling.llm.agy_adapter._client")
    def test_chat_with_agy_forwards_prompt_and_usage(self, mock_client):
        mock_msg = MagicMock()
        mock_msg.content = "<question_label>Câu 1.</question_label>"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        mock_client.chat.completions.create.return_value = MagicMock(
            id="conv-proxy-1",
            choices=[mock_choice],
            usage=mock_usage,
        )

        text, conv_id, usage = chat_with_agy(
            prompt="Annotate this question",
            system="You are an annotator",
            caller="unit-test",
        )

        self.assertEqual(text, "<question_label>Câu 1.</question_label>")
        self.assertEqual(conv_id, "conv-proxy-1")
        self.assertEqual(usage, {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], DEFAULT_AGY_MODEL)
        self.assertEqual(
            call_kwargs["messages"],
            [
                {"role": "system", "content": "You are an annotator"},
                {"role": "user", "content": "Annotate this question"},
            ],
        )
        self.assertEqual(call_kwargs["extra_body"]["caller"], "unit-test")

    @patch("sequence_labelling.llm.agy_adapter._client")
    def test_chat_with_agy_forwards_messages_and_thinking(self, mock_client):
        mock_msg = MagicMock()
        mock_msg.content = "OK"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_client.chat.completions.create.return_value = MagicMock(
            id="conv-proxy-2",
            choices=[mock_choice],
            usage=None,
        )

        messages = [{"role": "user", "content": "Hello"}]
        text, conv_id, usage = chat_with_agy(
            messages=messages,
            model="gpt-6-luna",
            thinking="medium",
            timeout=30.0,
            account="acc-1",
            agent="reviewer",
        )

        self.assertEqual(text, "OK")
        self.assertEqual(conv_id, "conv-proxy-2")
        self.assertEqual(usage, {})
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-6-luna")
        self.assertEqual(call_kwargs["messages"], messages)
        self.assertEqual(call_kwargs["timeout"], 30.0)
        self.assertEqual(
            call_kwargs["extra_body"],
            {"thinking": "medium", "account": "acc-1", "agent": "reviewer", "caller": "sequence_labelling"},
        )


if __name__ == "__main__":
    unittest.main()

