import os
from typing import Optional, Any, Sequence, Tuple, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

try:
    from openai_codex import Codex, Sandbox, ApprovalMode
    CODEX_AVAILABLE = True
except ImportError:
    CODEX_AVAILABLE = False

from sequence_labelling.llm.token_tracker import log_response, log_token_usage

# Locate .env by searching up directory hierarchy
current_dir = Path(__file__).resolve().parent
env_path = None
for p in [current_dir] + list(current_dir.parents):
    if (p / ".env").exists():
        env_path = p / ".env"
        break

if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    PARSER_MAX_TOKENS,
    get_provider_base_url,
    get_provider_api_key,
)
from sequence_labelling.llm.agy_adapter import chat_with_agy

# ── setup clients ─────────────────────────────────────────────────────────────
deepseek_key = get_provider_api_key("deepseek")
deepseek_client = (
    OpenAI(api_key=deepseek_key, base_url=get_provider_base_url("deepseek"))
    if deepseek_key
    else None
)

nvidia_key = get_provider_api_key("nvidia")
nvidia_client = (
    OpenAI(api_key=nvidia_key, base_url=get_provider_base_url("nvidia"))
    if nvidia_key
    else None
)

vilao_key = get_provider_api_key("vilao")
vilao_client = (
    OpenAI(api_key=vilao_key, base_url=get_provider_base_url("vilao"))
    if vilao_key
    else None
)

xah_key = get_provider_api_key("xah")
xah_client = (
    OpenAI(api_key=xah_key, base_url=get_provider_base_url("xah"))
    if xah_key
    else None
)

commandcode_key = get_provider_api_key("commandcode")
commandcode_client = (
    OpenAI(api_key=commandcode_key, base_url=get_provider_base_url("commandcode"))
    if commandcode_key
    else None
)

aixingialaire_key = get_provider_api_key("aixingialaire")
aixingialaire_client = (
    OpenAI(api_key=aixingialaire_key, base_url=get_provider_base_url("aixingialaire"))
    if aixingialaire_key
    else None
)

# Alias client for test mocking compatibility
client = deepseek_client


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
    """
    Call the LLM chat API using model and provider configured in config.yaml.
    Supports either single prompt string or multi-turn messages list.
    """
    target_model = model or PARSER_MODEL
    if isinstance(target_model, str):
        target_model = target_model.strip()
    target_provider = provider or PARSER_PROVIDER or "xah"
    if isinstance(target_provider, str):
        target_provider = target_provider.strip().lower()
    target_max_tokens = max_tokens or PARSER_MAX_TOKENS

    if messages is not None:
        chat_messages = list(messages)
    else:
        chat_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt or ""},
        ]

    kwargs = {
        "messages": chat_messages,
        "model": target_model,
        "stream": stream,
    }
    if target_max_tokens:
        kwargs["max_tokens"] = target_max_tokens
    
    if target_provider in ["codex", "openai_codex"]:
        if not CODEX_AVAILABLE:
            raise ImportError(
                "openai-codex package is not installed. Install via `uv pip install openai-codex`."
            )

        dev_instructions = system
        user_prompt = prompt or ""

        if messages is not None:
            formatted_turns = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    dev_instructions = content
                elif role == "user":
                    formatted_turns.append(content)
                elif role == "assistant":
                    formatted_turns.append(f"### Assistant Response:\n{content}\n")
            if formatted_turns:
                user_prompt = "\n\n".join(formatted_turns)

        base_instructions = (
            "You are a pure text processing engine. "
            "You have no tools, no workspace access, and no file system access. "
            "Process only the input text provided."
        )

        from openai_codex.api import ReasoningEffort

        # Normalize model name for Codex (e.g. phatchau036/gpt-5.6-luna -> gpt-5.6-luna)
        codex_model = target_model or "gpt-5.6-luna"
        if isinstance(codex_model, str):
            codex_model = codex_model.strip()
        if "/" in codex_model:
            codex_model = codex_model.split("/")[-1].strip()

        # Map thinking parameter to Codex ReasoningEffort
        effort_val = None
        if thinking is not None:
            if isinstance(thinking, ReasoningEffort):
                effort_val = thinking
            elif thinking is True:
                effort_val = ReasoningEffort.high
            elif thinking is False or thinking == 0:
                effort_val = ReasoningEffort.none
            elif isinstance(thinking, (int, float)):
                if thinking >= 3:
                    effort_val = ReasoningEffort.high
                elif thinking == 2:
                    effort_val = ReasoningEffort.medium
                elif thinking == 1:
                    effort_val = ReasoningEffort.low
                else:
                    effort_val = ReasoningEffort.none
            else:
                thinking_str = str(thinking).lower().strip()
                effort_map = {
                    "none": ReasoningEffort.none,
                    "minimal": ReasoningEffort.minimal,
                    "low": ReasoningEffort.low,
                    "medium": ReasoningEffort.medium,
                    "high": ReasoningEffort.high,
                    "xhigh": ReasoningEffort.xhigh,
                    "max": ReasoningEffort.xhigh,
                    "disabled": ReasoningEffort.none,
                }
                effort_val = effort_map.get(thinking_str, ReasoningEffort.medium)

        import time
        start_time = time.time()
        codex_key = get_provider_api_key("codex")
        with Codex() as codex_session:
            if codex_key:
                try:
                    codex_session.login_api_key(codex_key)
                except Exception:
                    pass
            thread = codex_session.thread_start(
                model=codex_model,
                base_instructions=base_instructions,
                developer_instructions=dev_instructions,
                approval_mode=ApprovalMode.auto_review,
                sandbox=Sandbox.read_only,
            )
            run_kwargs = {}
            if effort_val is not None:
                run_kwargs["effort"] = effort_val
            result = thread.run(user_prompt, **run_kwargs)
        duration_sec = time.time() - start_time

        if result.error:
            raise RuntimeError(f"Codex turn error: {result.error}")

        from sequence_labelling.llm.llm_logger import log_llm_call
        log_llm_call(
            messages=chat_messages,
            response=result,
            model=target_model,
            provider=target_provider,
            duration_sec=duration_sec
        )

        return result.final_response or ""

    elif target_provider in ["agy", "antigravity"]:
        import time
        from sequence_labelling.llm.llm_logger import log_llm_call

        start_time = time.time()
        reply_text, conv_id, usage = chat_with_agy(
            prompt=prompt,
            system=system,
            model=target_model,
            thinking=thinking,
            messages=chat_messages,
            image_bytes=image_bytes,
            image_filename=image_filename,
            images=images,
            caller="deepseek_client",
            auto_log=False,
        )
        duration_sec = time.time() - start_time

        log_llm_call(
            messages=chat_messages,
            response=reply_text,
            model=target_model,
            provider=target_provider,
            duration_sec=duration_sec,
            usage=usage,
        )
        return reply_text

    elif target_provider == "nvidia":
        if nvidia_client is None:
            raise ValueError("Error: Provider requires NVIDIA_API_KEY but it is not set.")
        active_client = nvidia_client
        thinking_bool = False
        if thinking is True or (isinstance(thinking, str) and thinking in ["high", "max"]):
            thinking_bool = True
        elif thinking is None:
            thinking_bool = True
        kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": thinking_bool}}
        
    elif target_provider == "xah":
        if xah_client is None:
            raise ValueError("Error: Model routes to Xah.io but neither XAH_API_KEY nor LLM_API_KEY is set.")
        active_client = xah_client

    elif target_provider == "vilao":
        if vilao_client is None:
            raise ValueError("Error: Model routes to Vilao.ai but LLM_API_KEY is not set.")
        active_client = vilao_client

    elif target_provider == "commandcode":
        if commandcode_client is None:
            raise ValueError("Error: Model routes to CommandCode but neither CMD_API_KEY nor COMMANDCODE_API_KEY is set.")
        active_client = commandcode_client

    elif target_provider == "aixingialaire":
        if aixingialaire_client is None:
            raise ValueError("Error: Provider requires AIXINGIALAIRE_API_KEY but it is not set.")
        active_client = aixingialaire_client
        effort = None
        if thinking is True:
            effort = "high"
        elif isinstance(thinking, str) and thinking in ["low", "medium", "high", "max"]:
            effort = thinking

        if effort is not None:
            kwargs["reasoning_effort"] = effort

    else:
        if deepseek_client is None:
            raise ValueError("Error: DEEPSEEK_API_KEY is not set.")
        active_client = deepseek_client
        kwargs["model"] = target_model
        
        effort = None
        if thinking is True:
            effort = "high"
        elif isinstance(thinking, str) and thinking in ["low", "medium", "high", "max"]:
            effort = thinking

        if effort is not None:
            kwargs["reasoning_effort"] = effort

    import time
    max_retries = 5
    max_total_timeout = 1800.0  # 30 minutes max budget per request
    overall_start_time = time.time()
    base_backoff = 5.0
    use_streaming = stream or target_provider == "aixingialaire"
    reply_text = ""

    for attempt in range(1, max_retries + 1):
        elapsed = time.time() - overall_start_time
        if elapsed >= max_total_timeout:
            print(f"⏱️ [Timeout Exceeded] Total retry time {elapsed:.1f}s exceeded 30 minute limit for provider '{target_provider}'. Skipping.", flush=True)
            raise TimeoutError(f"Provider '{target_provider}' exceeded 30 minute timeout limit ({elapsed:.1f}s elapsed)")

        try:
            start_time = time.time()
            if use_streaming:
                kwargs["stream"] = True
                stream_resp = active_client.chat.completions.create(**kwargs)
                collected = []
                for chunk in stream_resp:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        collected.append(chunk.choices[0].delta.content)
                reply_text = "".join(collected)
                duration_sec = time.time() - start_time
                break
            else:
                response = active_client.chat.completions.create(**kwargs)
                duration_sec = time.time() - start_time
                reply_text = response.choices[0].message.content or ""
                break
        except Exception as e:
            err_msg = str(e)
            elapsed = time.time() - overall_start_time
            is_524 = (
                "524" in err_msg
                or "origin_response_timeout" in err_msg
                or "Proxy Read Timeout" in err_msg
            )
            is_retryable = (
                is_524
                or "429" in err_msg
                or "rate limit" in err_msg.lower()
                or "timeout" in err_msg.lower()
                or "timed out" in err_msg.lower()
                or "502" in err_msg
                or "503" in err_msg
                or "504" in err_msg
                or "connection" in err_msg.lower()
            )
            if not is_retryable or attempt >= max_retries or elapsed >= max_total_timeout:
                print(f"❌ [API Error] Provider '{target_provider}' stopped after attempt {attempt}/{max_retries} ({elapsed:.1f}s elapsed): {e}", flush=True)
                raise

            # Determine sleep delay
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_hdr = getattr(e.response, "headers", {}).get("retry-after")
                if retry_after_hdr and retry_after_hdr.isdigit():
                    retry_after = float(retry_after_hdr)

            if retry_after is None:
                m = re.search(r"['\"]retry_after['\"]\s*:\s*(\d+)", err_msg)
                if m:
                    retry_after = float(m.group(1))

            if retry_after is not None:
                sleep_sec = min(60.0, max(5.0, retry_after))
            elif is_524:
                # Cloudflare 524 origin read timeout: backoff capped at 60s
                sleep_sec = min(60.0, 10.0 * attempt)
            else:
                sleep_sec = min(60.0, base_backoff * (1.5 ** (attempt - 1)))

            # If sleep would exceed 30 minute max budget, cap it or abort
            if elapsed + sleep_sec >= max_total_timeout:
                print(f"⏱️ [Timeout Exceeded] Sleeping {sleep_sec:.1f}s would exceed 30 minute limit. Aborting retries.", flush=True)
                raise TimeoutError(f"Provider '{target_provider}' exceeded 30 minute timeout limit")

            print(
                f"⚠️ [API Retry {attempt}/{max_retries}] Provider '{target_provider}' error ({'524 Cloudflare Origin Timeout' if is_524 else e}). "
                f"Elapsed: {elapsed:.1f}s/1800s. Sleeping {sleep_sec:.1f}s before retry...",
                flush=True
            )
            time.sleep(sleep_sec)

    from sequence_labelling.llm.llm_logger import log_llm_call
    log_llm_call(
        messages=kwargs.get("messages", []),
        response=reply_text,
        model=target_model,
        provider=target_provider,
        duration_sec=duration_sec
    )

    return reply_text


if __name__ == "__main__":
    result = chat("Hello")
    print(result)
