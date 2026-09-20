import pytest
from unittest.mock import patch, MagicMock

openai_codex = pytest.importorskip("openai_codex")
from openai_codex.api import ReasoningEffort
from sequence_labelling.llm.deepseek_client import chat


def test_codex_thinking_effort_mapping():
    """Verify that all thinking parameter variations map properly to valid ReasoningEffort values without AttributeError."""
    test_cases = [
        ("max", ReasoningEffort.xhigh),
        ("xhigh", ReasoningEffort.xhigh),
        ("high", ReasoningEffort.high),
        ("medium", ReasoningEffort.medium),
        ("low", ReasoningEffort.low),
        ("minimal", ReasoningEffort.minimal),
        ("none", ReasoningEffort.none),
        ("disabled", ReasoningEffort.none),
        (True, ReasoningEffort.high),
        (False, ReasoningEffort.none),
        (ReasoningEffort.medium, ReasoningEffort.medium),
    ]

    for thinking_input, expected_effort in test_cases:
        mock_result = MagicMock()
        mock_result.error = None
        mock_result.final_response = "mock response"

        mock_thread = MagicMock()
        mock_thread.run.return_value = mock_result

        mock_session = MagicMock()
        mock_session.thread_start.return_value = mock_thread
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None

        with patch("sequence_labelling.llm.deepseek_client.Codex", return_value=mock_session), \
             patch("sequence_labelling.llm.deepseek_client.get_provider_api_key", return_value="dummy_key"), \
             patch("sequence_labelling.llm.llm_logger.log_llm_call"):

            resp = chat(
                prompt="test prompt",
                provider="codex",
                thinking=thinking_input,
            )
            assert resp == "mock response"
            mock_thread.run.assert_called_once()
            _, kwargs = mock_thread.run.call_args
            assert kwargs.get("effort") == expected_effort, f"Failed for thinking input: {thinking_input}"


def test_codex_multi_turn_messages_formatting():
    """Verify that multi-turn messages (system, user, assistant, user) format correctly for Codex thread."""
    mock_result = MagicMock()
    mock_result.error = None
    mock_result.final_response = "APPROVED"

    mock_thread = MagicMock()
    mock_thread.run.return_value = mock_result

    mock_session = MagicMock()
    mock_session.thread_start.return_value = mock_thread
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None

    messages = [
        {"role": "system", "content": "You are a sequence annotator."},
        {"role": "user", "content": "Annotate: Câu 1. Math question."},
        {"role": "assistant", "content": "<question_label>Câu 1.</question_label> <stem>Math question.</stem>"},
        {"role": "user", "content": "Audit Role A output above."},
    ]

    with patch("sequence_labelling.llm.deepseek_client.Codex", return_value=mock_session), \
         patch("sequence_labelling.llm.deepseek_client.get_provider_api_key", return_value="dummy_key"), \
         patch("sequence_labelling.llm.llm_logger.log_llm_call"):

        resp = chat(
            messages=messages,
            provider="codex",
            thinking="high",
        )
        assert resp == "APPROVED"
        mock_session.thread_start.assert_called_once()
        _, start_kwargs = mock_session.thread_start.call_args
        assert start_kwargs.get("developer_instructions") == "You are a sequence annotator."

        mock_thread.run.assert_called_once()
        user_prompt_arg, _ = mock_thread.run.call_args
        assert "Annotate: Câu 1. Math question." in user_prompt_arg[0]
        assert "### Assistant Response:" in user_prompt_arg[0]
        assert "<question_label>Câu 1.</question_label>" in user_prompt_arg[0]
        assert "Audit Role A output above." in user_prompt_arg[0]


def test_codex_error_handling():
    """Verify RuntimeError is raised if Codex turn returns an error."""
    mock_result = MagicMock()
    mock_result.error = "Codex RPC failure"
    mock_result.final_response = None

    mock_thread = MagicMock()
    mock_thread.run.return_value = mock_result

    mock_session = MagicMock()
    mock_session.thread_start.return_value = mock_thread
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None

    with patch("sequence_labelling.llm.deepseek_client.Codex", return_value=mock_session), \
         patch("sequence_labelling.llm.deepseek_client.get_provider_api_key", return_value="dummy_key"):

        with pytest.raises(RuntimeError, match="Codex turn error: Codex RPC failure"):
            chat(prompt="Test prompt", provider="codex")
