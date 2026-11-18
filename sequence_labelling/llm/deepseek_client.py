"""
Unified LLM Client — Proxied via unified-llm-proxy (port 8765).
Supports all models and providers via central gateway.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional, Any, Sequence, Tuple, List, Dict
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    PARSER_MAX_TOKENS,
    providers as CONFIG_PROVIDERS,
)

logger = logging.getLogger(__name__)

PROXY_BASE_URL = os.environ.get("UNIFIED_LLM_PROXY_URL", "http://127.0.0.1:8765/v1")
client = OpenAI(base_url=PROXY_BASE_URL, api_key="sk-proxy")
deepseek_client = client

_KNOWN_PROVIDERS = set((CONFIG_PROVIDERS or {}).keys()) | {
    "agy",
    "codex",
    "deepseek",
    "aixingialaire",
    "commandcode",
    "nvidia",
    "xah",
    "vilao",
}


def _resolve_provider_and_account(
    provider: Optional[str], account: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve provider and optional AGY account (supports 'agy:account' or direct account alias)."""
    if not provider:
        return provider, account
    prov = provider.strip()
    if ":" in prov:
        base, acct = prov.split(":", 1)
        return base.strip() or "agy", account or acct.strip() or None
    if prov.lower() not in _KNOWN_PROVIDERS and account is None:
        # Treat unrecognized provider names (e.g. 'daominhwysi') as an AGY account alias.
        return "agy", prov
    return prov, account


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
    account: Optional[str] = None,
    max_retries: int = 4,
) -> str:
    target_model = model or PARSER_MODEL or "gemini-3.8-flash"
    raw_provider = provider or PARSER_PROVIDER
    target_provider, target_account = _resolve_provider_and_account(raw_provider, account)
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
    if target_account:
        extra_body["account"] = target_account
    if thinking is not None:
        extra_body["thinking"] = thinking
    if extra_body:
        kwargs["extra_body"] = extra_body

    attempts = max(1, int(max_retries))
    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        try:
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
        except (APIConnectionError, APITimeoutError) as exc:
            last_err = exc
        except InternalServerError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code not in (502, 503, 504):
                raise
            last_err = exc

        if attempt + 1 < attempts:
            delay = min(12.0, 1.5 * (2 ** attempt)) + random.uniform(0.0, 0.5)
            logger.warning(
                f"[LLM Client] Transient error ({last_err}); retrying in {delay:.1f}s "
                f"(attempt {attempt + 2}/{attempts})..."
            )
            time.sleep(delay)

    assert last_err is not None
    raise last_err


if __name__ == "__main__":
    result = chat("Hello! Respond with: PONG")
    print(result)
