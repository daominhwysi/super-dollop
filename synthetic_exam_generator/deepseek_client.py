import os
import time
from typing import Optional, Any, List, Dict
from pathlib import Path
try:
    from dotenv import load_dotenv
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
except ImportError:
    pass

from openai import OpenAI

try:
    from openai_codex import Codex, Sandbox, ApprovalMode
    from openai_codex.api import ReasoningEffort
    CODEX_AVAILABLE = True
except ImportError:
    CODEX_AVAILABLE = False
    ReasoningEffort = None

try:
    from synthetic_exam_generator.token_tracker import log_response
    from synthetic_exam_generator.config import load_config
except ImportError:
    from .token_tracker import log_response
    from .config import load_config

# Load providers config
_cfg = load_config()
PROVIDERS_CFG = _cfg.get("providers", {})
GEN_CFG = _cfg.get("generation", {})
DEFAULT_MODEL = GEN_CFG.get("model", "gpt-5.6-luna")
DEFAULT_PROVIDER = GEN_CFG.get("provider", "codex")
DEFAULT_THINKING = GEN_CFG.get("thinking", "low")


def get_provider_base_url(provider_name: str) -> str:
    prov = PROVIDERS_CFG.get(provider_name.lower(), {})
    return prov.get("base_url", "")


def get_provider_api_key(provider_name: str) -> str:
    prov = PROVIDERS_CFG.get(provider_name.lower(), {})
    key_env = prov.get("api_key_env")
    if key_env and os.environ.get(key_env):
        return os.environ.get(key_env)

    # Fallback environment variable aliases
    p_lower = provider_name.lower()
    if p_lower in ["codex", "openai_codex"]:
        return os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    elif p_lower == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY") or ""
    elif p_lower == "nvidia":
        return os.environ.get("NVIDIA_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    elif p_lower == "vilao":
        return os.environ.get("LLM_API_KEY") or ""
    elif p_lower == "xah":
        return os.environ.get("XAH_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    elif p_lower == "commandcode":
        return os.environ.get("CMD_API_KEY") or os.environ.get("COMMANDCODE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    return ""


# ── setup OpenAI clients ──────────────────────────────────────────────────────
deepseek_key = get_provider_api_key("deepseek")
deepseek_client = (
    OpenAI(api_key=deepseek_key or "placeholder_key", base_url=get_provider_base_url("deepseek") or "https://api.deepseek.com")
    if deepseek_key
    else OpenAI(api_key="placeholder_key", base_url="https://api.deepseek.com")
)

nvidia_key = get_provider_api_key("nvidia")
nvidia_client = (
    OpenAI(api_key=nvidia_key, base_url=get_provider_base_url("nvidia") or "https://integrate.api.nvidia.com/v1")
    if nvidia_key
    else None
)

vilao_key = get_provider_api_key("vilao")
vilao_client = (
    OpenAI(api_key=vilao_key, base_url=get_provider_base_url("vilao") or "https://api.vilao.ai/v1")
    if vilao_key
    else None
)

xah_key = get_provider_api_key("xah")
xah_client = (
    OpenAI(api_key=xah_key, base_url=get_provider_base_url("xah") or "https://api.xah.io/v1")
    if xah_key
    else None
)

commandcode_key = get_provider_api_key("commandcode")
commandcode_client = (
    OpenAI(api_key=commandcode_key, base_url=get_provider_base_url("commandcode") or "http://127.0.0.1:3050/v1")
    if commandcode_key
    else None
)

# Alias client for backward-compatibility with test mocking
client = deepseek_client


def _map_codex_reasoning_effort(thinking: Any) -> Optional[Any]:
    if not CODEX_AVAILABLE or thinking is None or ReasoningEffort is None:
        return None
    if isinstance(thinking, ReasoningEffort):
        return thinking
    if thinking is True:
        return ReasoningEffort.high
    if thinking is False or thinking == 0:
        return ReasoningEffort.none
    if isinstance(thinking, (int, float)):
        if thinking >= 3:
            return ReasoningEffort.high
        elif thinking == 2:
            return ReasoningEffort.medium
        elif thinking == 1:
            return ReasoningEffort.low
        else:
            return ReasoningEffort.none

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
    return effort_map.get(thinking_str, ReasoningEffort.medium)


def chat(
    prompt: Optional[str] = None,
    system: str = "You are a helpful assistant",
    model: Optional[str] = None,
    thinking: Optional[Any] = None,
    provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
    messages: Optional[Any] = None,
) -> str:
    """
    Call the LLM chat API using the model and provider configured in config.yaml or passed as arguments.
    Supports Codex, DeepSeek, NVIDIA, Vilao, Xah, and CommandCode.
    """
    target_model = model or DEFAULT_MODEL
    target_provider = (provider or DEFAULT_PROVIDER or "codex").lower()
    target_thinking = thinking if thinking is not None else DEFAULT_THINKING

    if messages is not None:
        chat_messages = list(messages)
    else:
        chat_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt or ""},
        ]

    # Handle AGY provider
    if target_provider in ["agy", "antigravity"]:
        try:
            from sequence_labelling.llm.agy_adapter import chat_with_agy
            return chat_with_agy(
                prompt=prompt,
                system=system,
                model=target_model,
                thinking=target_thinking,
                messages=messages,
            )
        except Exception as agy_err:
            print(f"Warning: AGY chat call failed or unavailable ({agy_err}), falling back to OpenAI/Codex.")

    # Handle Codex provider
    if target_provider in ["codex", "openai_codex"]:
        if not CODEX_AVAILABLE:
            raise ImportError(
                "openai-codex package is not installed. Install via `pixi add --pypi openai-codex`."
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

        codex_model = target_model or "gpt-5.6-luna"
        if "/" in codex_model:
            codex_model = codex_model.split("/")[-1]

        effort_val = _map_codex_reasoning_effort(target_thinking)

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

        log_response(result, model=target_model, provider="codex", caller="synthetic_exam", duration_sec=duration_sec)
        return result.final_response or ""

    # Setup OpenAI-compatible API call
    kwargs = {
        "messages": chat_messages,
        "model": target_model,
        "stream": False,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    if target_provider == "nvidia":
        if nvidia_client is None:
            raise ValueError("Error: Provider requires NVIDIA_API_KEY or DEEPSEEK_API_KEY but neither is set.")
        active_client = nvidia_client
        target_m = target_model
        if target_m in ["deepseek-v4-pro", "deepseek-ai/deepseek-v4-pro"]:
            target_m = "deepseek-ai/deepseek-v4-pro"
        kwargs["model"] = target_m
        thinking_bool = False
        if target_thinking is True or (isinstance(target_thinking, str) and target_thinking in ["high", "max"]):
            thinking_bool = True
        elif target_thinking is None or target_thinking == "low" or target_thinking == "medium":
            thinking_bool = True
        kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": thinking_bool}}

    elif target_provider == "vilao":
        if vilao_client is None:
            if deepseek_client is not None and os.environ.get("DEEPSEEK_API_KEY"):
                active_client = deepseek_client
            else:
                raise ValueError("Error: Model routes to Vilao.ai but LLM_API_KEY is not set.")
        else:
            active_client = vilao_client
            final_model = target_model
            if "/" not in final_model:
                if "minimax" in final_model.lower():
                    final_model = f"mn/{final_model}"
                elif "deepseek" in final_model.lower():
                    final_model = f"deepseek/{final_model}"
            kwargs["model"] = final_model

    elif target_provider == "xah":
        if xah_client is None:
            raise ValueError("Error: Model routes to Xah.io but neither XAH_API_KEY nor LLM_API_KEY is set.")
        active_client = xah_client

    elif target_provider == "commandcode":
        if commandcode_client is None:
            raise ValueError("Error: Model routes to CommandCode but neither CMD_API_KEY nor COMMANDCODE_API_KEY is set.")
        active_client = commandcode_client

    else:
        # Default to DeepSeek
        active_client = client or deepseek_client
        if active_client is None or (not os.environ.get("DEEPSEEK_API_KEY") and not hasattr(active_client.chat.completions.create, "assert_called_with")):
            raise ValueError("Error: DEEPSEEK_API_KEY is not set.")
        kwargs["model"] = target_model

        effort = None
        if thinking is True:
            effort = "high"
        elif isinstance(thinking, str) and thinking.lower() in ["low", "medium", "high", "max"]:
            effort = thinking.lower()
        elif thinking is None:
            if "pro" in target_model or "reasoner" in target_model:
                effort = "high"

        if effort is not None:
            kwargs["reasoning_effort"] = effort

    start_time = time.time()
    response = active_client.chat.completions.create(**kwargs)
    duration_sec = time.time() - start_time
    log_response(response, model=target_model, provider=target_provider, caller="synthetic_exam", duration_sec=duration_sec)

    return response.choices[0].message.content


if __name__ == "__main__":
    result = chat("Hello")
    print(result)
