"""
providers.py — Unified AI provider abstraction for CodeForge.

Every provider speaks the same interface: send messages + tools,
get back (content_blocks, stop_reason). The forge engine doesn't
care which API is underneath.

Supported:
  - Anthropic (Claude models)
  - OpenAI (GPT-4o, o1, etc.)
  - OpenRouter (any model via unified API)
  - Google Gemini
  - Moonshot AI (Kimi)
  - Alibaba Qwen
  - Mistral
  - Groq
  - Together AI
  - Ollama (local)
  - Custom (any OpenAI-compatible endpoint)
"""

from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Unified response type
# ---------------------------------------------------------------------------

@dataclass
class ForgeResponse:
    """Normalized response from any provider."""
    content_blocks: list[dict]   # [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
    stop_reason: str              # "end_turn" | "tool_use" | "max_tokens" | "stop"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, "ProviderMeta"] = {}


@dataclass
class ProviderMeta:
    id: str
    name: str
    description: str
    default_model: str
    suggested_models: list[str]
    needs_base_url: bool = False
    base_url_hint: str = ""
    docs_url: str = ""


def _reg(meta: ProviderMeta):
    PROVIDER_REGISTRY[meta.id] = meta
    return meta


ANTHROPIC_META  = _reg(ProviderMeta("anthropic",  "Anthropic",      "Claude models (Opus, Sonnet, Haiku)",           "claude-opus-4-6",             ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],         docs_url="https://docs.anthropic.com"))
OPENAI_META     = _reg(ProviderMeta("openai",      "OpenAI",         "GPT-4o, o3, o1 models",                        "gpt-4o",                       ["gpt-4o", "gpt-4o-mini", "o3", "o1", "o1-mini"],                             docs_url="https://platform.openai.com/docs"))
OPENROUTER_META = _reg(ProviderMeta("openrouter",  "OpenRouter",     "Access 200+ models via one API",               "anthropic/claude-opus-4-6",     ["anthropic/claude-opus-4-6", "openai/gpt-4o", "google/gemini-2.0-flash-001", "meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-r1"],  docs_url="https://openrouter.ai/docs"))
GEMINI_META     = _reg(ProviderMeta("gemini",      "Google Gemini",  "Gemini 2.0, 1.5 Pro, Flash models",           "gemini-2.0-flash-001",         ["gemini-2.0-flash-001", "gemini-1.5-pro", "gemini-1.5-flash"],               docs_url="https://ai.google.dev/docs"))
MOONSHOT_META   = _reg(ProviderMeta("moonshot",    "Moonshot AI",    "Kimi long-context models",                     "moonshot-v1-128k",              ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],                    docs_url="https://platform.moonshot.cn/docs"))
QWEN_META       = _reg(ProviderMeta("qwen",        "Alibaba Qwen",   "Qwen 2.5, Qwen-Long models",                   "qwen-max",                      ["qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-coder-32b-instruct"],        docs_url="https://dashscope.aliyuncs.com/docs"))
MISTRAL_META    = _reg(ProviderMeta("mistral",     "Mistral",        "Mistral Large, Codestral models",              "mistral-large-latest",          ["mistral-large-latest", "codestral-latest", "mistral-medium-latest"],         docs_url="https://docs.mistral.ai"))
GROQ_META       = _reg(ProviderMeta("groq",        "Groq",           "Ultra-fast inference (Llama, Mixtral)",        "llama-3.3-70b-versatile",       ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],    docs_url="https://console.groq.com/docs"))
TOGETHER_META   = _reg(ProviderMeta("together",    "Together AI",    "Open-source models at scale",                  "meta-llama/Llama-3.3-70B-Instruct-Turbo", ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1", "Qwen/QwQ-32B-Preview"], docs_url="https://docs.together.ai"))
OLLAMA_META     = _reg(ProviderMeta("ollama",      "Ollama (local)", "Run models locally — no API key needed",       "qwen2.5-coder:32b",             ["qwen2.5-coder:32b", "llama3.3:70b", "deepseek-coder-v2:16b", "codestral:22b"], needs_base_url=True, base_url_hint="http://localhost:11434", docs_url="https://ollama.ai"))
CUSTOM_META     = _reg(ProviderMeta("custom",      "Custom (OpenAI-compatible)", "Any OpenAI-compatible endpoint",   "your-model-name",               [],                                                                             needs_base_url=True, base_url_hint="https://your-endpoint.com/v1"))


# ---------------------------------------------------------------------------
# Base provider class
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 8192,
    ) -> ForgeResponse:
        """Send a completion request, return normalized ForgeResponse."""
        ...

    @property
    @abstractmethod
    def provider_id(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseProvider):
    provider_id = "anthropic"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key, model)
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)

    def complete(self, system, messages, tools, max_tokens=8192) -> ForgeResponse:
        # Anthropic uses native tool format — pass directly
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=_clean_messages_anthropic(messages),
        )
        blocks = []
        for b in resp.content:
            if hasattr(b, "type"):
                if b.type == "text":
                    blocks.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    blocks.append({
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.input,
                    })
        stop = resp.stop_reason or "end_turn"
        return ForgeResponse(
            content_blocks=blocks,
            stop_reason=stop,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (covers OpenAI, OpenRouter, Moonshot, Qwen,
#   Groq, Together, Mistral, Ollama, Custom)
# ---------------------------------------------------------------------------

class OpenAICompatProvider(BaseProvider):
    """
    Handles any provider that speaks the OpenAI chat completions API with
    parallel tool calling. This covers the vast majority of modern providers.
    """

    def __init__(self, api_key: str, model: str, base_url: str, provider_id_str: str = "openai"):
        super().__init__(api_key, model, base_url)
        self._provider_id = provider_id_str
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def provider_id(self):
        return self._provider_id

    def complete(self, system, messages, tools, max_tokens=8192) -> ForgeResponse:
        # Convert Anthropic-style tool definitions → OpenAI function format
        oai_tools = _anthropic_tools_to_openai(tools)
        oai_messages = _messages_to_openai(system, messages)

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        blocks: list[dict] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    inp = json.loads(tc.function.arguments)
                except Exception:
                    inp = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": inp,
                })

        stop = "tool_use" if msg.tool_calls else "end_turn"
        if choice.finish_reason in ("length", "max_tokens"):
            stop = "max_tokens"

        usage = resp.usage
        return ForgeResponse(
            content_blocks=blocks,
            stop_reason=stop,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


# ---------------------------------------------------------------------------
# Google Gemini provider
# ---------------------------------------------------------------------------

class GeminiProvider(BaseProvider):
    provider_id = "gemini"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key, model)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai

    def complete(self, system, messages, tools, max_tokens=8192) -> ForgeResponse:
        import google.generativeai as genai
        from google.generativeai.types import content_types

        # Build Gemini tool declarations
        gemini_tools = _anthropic_tools_to_gemini(tools)

        model_obj = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            tools=gemini_tools if gemini_tools else None,
        )

        gemini_history = _messages_to_gemini(messages[:-1])
        last_msg = messages[-1]["content"] if messages else ""
        if isinstance(last_msg, list):
            # tool results
            last_msg = _tool_results_to_gemini_text(last_msg)

        chat = model_obj.start_chat(history=gemini_history)
        resp = chat.send_message(last_msg)

        blocks: list[dict] = []
        candidate = resp.candidates[0] if resp.candidates else None
        if not candidate:
            return ForgeResponse(content_blocks=[], stop_reason="end_turn", model=self.model)

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                blocks.append({"type": "text", "text": part.text})
            elif hasattr(part, "function_call") and part.function_call.name:
                fc = part.function_call
                blocks.append({
                    "type": "tool_use",
                    "id": f"gemini_{fc.name}_{len(blocks)}",
                    "name": fc.name,
                    "input": dict(fc.args),
                })

        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return ForgeResponse(content_blocks=blocks, stop_reason=stop, model=self.model)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_OPENAI_COMPAT_URLS: dict[str, str] = {
    "openai":      "https://api.openai.com/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "moonshot":    "https://api.moonshot.cn/v1",
    "qwen":        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "mistral":     "https://api.mistral.ai/v1",
    "groq":        "https://api.groq.com/openai/v1",
    "together":    "https://api.together.xyz/v1",
}


def build_provider(provider_id: str, api_key: str, model: str, base_url: Optional[str] = None) -> BaseProvider:
    """Construct the right provider object from config."""
    if provider_id == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)

    if provider_id == "gemini":
        return GeminiProvider(api_key=api_key, model=model)

    if provider_id in _OPENAI_COMPAT_URLS or provider_id in ("ollama", "custom"):
        url = base_url or _OPENAI_COMPAT_URLS.get(provider_id, "")
        if not url:
            raise ValueError(f"base_url required for provider '{provider_id}'")
        return OpenAICompatProvider(api_key=api_key or "ollama", model=model, base_url=url, provider_id_str=provider_id)

    raise ValueError(f"Unknown provider: '{provider_id}'. Run `codeforge config` to set up.")


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

def _clean_messages_anthropic(messages: list) -> list:
    """Ensure all message content is serializable for the Anthropic API."""
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            clean_content = []
            for block in content:
                if isinstance(block, dict):
                    clean_content.append(block)
                elif hasattr(block, "__dict__"):
                    d = vars(block)
                    # Anthropic SDK objects — convert to dict
                    if hasattr(block, "type"):
                        if block.type == "text":
                            clean_content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            clean_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            })
                        else:
                            clean_content.append(d)
                    else:
                        clean_content.append(d)
            cleaned.append({**msg, "content": clean_content})
        else:
            cleaned.append(msg)
    return cleaned


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI function calling format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def _messages_to_openai(system: str, messages: list) -> list:
    """Convert Anthropic-style message list to OpenAI format."""
    result = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            # Could be assistant tool_use blocks or user tool_result blocks
            if role == "assistant":
                text_parts = []
                tool_calls = []
                for block in content:
                    b = block if isinstance(block, dict) else (vars(block) if hasattr(block, "__dict__") else {})
                    btype = b.get("type", "")
                    if btype == "text":
                        text_parts.append(b.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        })
                oai_msg: dict = {"role": "assistant"}
                if text_parts:
                    oai_msg["content"] = " ".join(text_parts)
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)

            elif role == "user":
                # Tool results
                for block in content:
                    b = block if isinstance(block, dict) else {}
                    if b.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id", ""),
                            "content": b.get("content", ""),
                        })
                    else:
                        # Regular user content block
                        result.append({"role": "user", "content": str(b)})

    return result


def _anthropic_tools_to_gemini(tools: list[dict]) -> list:
    """Convert Anthropic tool defs to Gemini function declarations."""
    try:
        import google.generativeai as genai
        from google.generativeai.types import content_types
        declarations = []
        for t in tools:
            schema = t.get("input_schema", {})
            # Gemini needs properties to not have 'type' at top level
            props = schema.get("properties", {})
            required = schema.get("required", [])
            fd = genai.protos.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: _schema_prop_to_gemini(v)
                        for k, v in props.items()
                    },
                    required=required,
                ),
            )
            declarations.append(fd)
        return [genai.protos.Tool(function_declarations=declarations)]
    except Exception:
        return []


def _schema_prop_to_gemini(prop: dict):
    """Convert a JSON schema property to a Gemini Schema object."""
    import google.generativeai as genai
    type_map = {
        "string": genai.protos.Type.STRING,
        "integer": genai.protos.Type.INTEGER,
        "number": genai.protos.Type.NUMBER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array": genai.protos.Type.ARRAY,
        "object": genai.protos.Type.OBJECT,
    }
    ptype = type_map.get(prop.get("type", "string"), genai.protos.Type.STRING)
    return genai.protos.Schema(type=ptype, description=prop.get("description", ""))


def _messages_to_gemini(messages: list) -> list:
    """Convert messages to Gemini history format."""
    history = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        gemini_role = "user" if role == "user" else "model"
        if isinstance(content, str) and content:
            history.append({"role": gemini_role, "parts": [content]})
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text:
                history.append({"role": gemini_role, "parts": [text]})
    return history


def _tool_results_to_gemini_text(blocks: list) -> str:
    """Flatten tool result blocks into a text string for Gemini."""
    parts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            parts.append(f"Tool result: {b.get('content', '')}")
    return "\n".join(parts) or "Continue."
