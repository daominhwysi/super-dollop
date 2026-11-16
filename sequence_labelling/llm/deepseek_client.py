"""
Unified LLM Client — Proxied via unified-llm-proxy (port 8765).
Supports all models and providers via central gateway.
"""

from __future__ import annotations

import os
from typing import Optional, Any, Sequence, Tuple, List, Dict
from openai import OpenAI

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    PARSER_MAX_TOKENS,
)

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
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
    images: Optional[Sequence[Tuple[str, bytes]]] = None,
    stream: bool = False,
) -> str:
    target_model = model or PARSER_MODEL or "gemini-3.8-flash"
    target_provider = provider or PARSER_PROVIDER
    target_max_tokens = max_tokens or PARSER_MAX_TOKENS

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
    if target_max_tokens:
        kwargs["max_tokens"] = target_max_tokens

    extra_body: Dict[str, Any] = {}
    if target_provider:
        extra_body["provider"] = target_provider
    if thinking is not None:
        extra_body["thinking"] = thinking
    if extra_body:
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
    result = chat("Hello! Respond with: PONG")
    print(result)
