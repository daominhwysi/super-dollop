from unittest.mock import MagicMock, patch
import pytest

from sequence_labelling.llm.deepseek_client import chat


def test_chat_proxy_forwards_provider_and_thinking():
    """Verify chat() forwards model, messages, provider, and thinking via proxy extra_body."""
    mock_msg = MagicMock()
    mock_msg.content = "mock response"
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock(choices=[mock_choice])

    with patch("sequence_labelling.llm.deepseek_client.client.chat.completions.create", return_value=mock_resp) as mock_create:
        resp = chat(
            prompt="test prompt",
            provider="codex",
            model="gpt-6-luna",
            thinking="high",
            max_tokens=512,
        )
        assert resp == "mock response"
        mock_create.assert_called_once_with(
            model="gpt-6-luna",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "test prompt"},
            ],
            stream=False,
            max_tokens=512,
            extra_body={"provider": "codex", "thinking": "high"},
        )


def test_chat_proxy_multi_turn_messages():
    """Verify multi-turn messages pass through verbatim to the proxy client."""
    mock_msg = MagicMock()
    mock_msg.content = "APPROVED"
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock(choices=[mock_choice])

    messages = [
        {"role": "system", "content": "You are a sequence annotator."},
        {"role": "user", "content": "Annotate: Câu 1. Math question."},
        {"role": "assistant", "content": "<question_label>Câu 1.</question_label> <stem>Math question.</stem>"},
        {"role": "user", "content": "Audit output above."},
    ]

    with patch("sequence_labelling.llm.deepseek_client.client.chat.completions.create", return_value=mock_resp) as mock_create:
        resp = chat(messages=messages, provider="codex", model="gpt-6-luna", thinking="medium")
        assert resp == "APPROVED"
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["messages"] == messages
        assert call_kwargs["extra_body"] == {"provider": "codex", "thinking": "medium"}


def test_chat_proxy_propagates_client_error():
    """Verify proxy client exceptions propagate without silent swallowing."""
    with patch(
        "sequence_labelling.llm.deepseek_client.client.chat.completions.create",
        side_effect=RuntimeError("Proxy connection error"),
    ):
        with pytest.raises(RuntimeError, match="Proxy connection error"):
            chat(prompt="Test prompt", provider="codex")
