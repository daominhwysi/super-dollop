"""
Unified LLM Client for Synthetic Exam Generator — Proxied via unified-llm-proxy (port 8765).
"""

from __future__ import annotations

import os
from typing import Optional, Any, List, Dict
from openai import OpenAI

try:
    from synthetic_exam_generator.config import load_config
    _cfg = load_config()
except Exception:
    _cfg = {}

GEN_CFG = _cfg.get("generation", {})
DEFAULT_MODEL = GEN_CFG.get("model", "gemini-3.8-flash")
DEFAULT_PROVIDER = GEN_CFG.get("provider")
DEFAULT_THINKING = GEN_CFG.get("thinking", "low")

PROXY_BASE_URL = os.environ.get("UNIFIED_LLM_PROXY_URL", "http://127.0.0.1:8765/v1")
client = OpenAI(base_url=PROXY_BASE_URL, api_key="sk-proxy")
deepseek_client = client


def chat(
    prompt: Optional[str] = None,
    system: str = "You are a helpful assistant",
    model: Optional[str] = None,
    thinking: Optional[Any] = None,
    provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
    messages: Optional[Any] = None,
    stream: bool = False,
) -> str:
    target_model = model or DEFAULT_MODEL
    target_provider = provider if provider is not None else (None if model else DEFAULT_PROVIDER)
    target_thinking = thinking if thinking is not None else DEFAULT_THINKING

    if messages is not None:
        chat_messages = list(messages)
    else:
        chat_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt or ""},
        ]

    kwargs: Dict[str, Any] = {
        "model": target_model,
        "messages": chat_messages,
        "stream": stream,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    extra_body: Dict[str, Any] = {"caller": "synthetic_exam_generator"}
    if target_provider:
        extra_body["provider"] = target_provider
    if target_thinking is not None:
        extra_body["thinking"] = target_thinking
    kwargs["extra_body"] = extra_body

    if stream:
        resp_stream = client.chat.completions.create(**kwargs)
        collected = []
        for chunk in resp_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                collected.append(chunk.choices[0].delta.content)
        return "".join(collected)
    else:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


if __name__ == "__main__":
    print(chat("Hello", model="gemini-3.8-flash"))
