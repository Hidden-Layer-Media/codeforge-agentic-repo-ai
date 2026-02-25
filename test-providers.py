"""
Tests for CodeForge provider abstraction and config system.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from codeforge.providers import (
    ForgeResponse,
    PROVIDER_REGISTRY,
    build_provider,
    _anthropic_tools_to_openai,
    _messages_to_openai,
)
from codeforge.config import ForgeConfig, ProviderConfig


# ── Provider Registry ────────────────────────────────────────────────────────

class TestProviderRegistry:
    def test_all_expected_providers_present(self):
        expected = {
            "anthropic", "openai", "openrouter", "gemini",
            "moonshot", "qwen", "mistral", "groq", "together",
            "ollama", "custom",
        }
        assert expected.issubset(set(PROVIDER_REGISTRY.keys()))

    def test_each_provider_has_required_fields(self):
        for pid, meta in PROVIDER_REGISTRY.items():
            assert meta.id == pid, f"{pid}: id mismatch"
            assert meta.name, f"{pid}: missing name"
            assert meta.default_model is not None, f"{pid}: missing default_model"

    def test_ollama_needs_base_url(self):
        assert PROVIDER_REGISTRY["ollama"].needs_base_url is True

    def test_custom_needs_base_url(self):
        assert PROVIDER_REGISTRY["custom"].needs_base_url is True

    def test_anthropic_has_suggested_models(self):
        assert len(PROVIDER_REGISTRY["anthropic"].suggested_models) >= 3

    def test_openrouter_has_diverse_models(self):
        models = PROVIDER_REGISTRY["openrouter"].suggested_models
        # Should include models from multiple providers
        providers_represented = set()
        for m in models:
            if "/" in m:
                providers_represented.add(m.split("/")[0])
        assert len(providers_represented) >= 2


# ── Tool Format Conversion ───────────────────────────────────────────────────

class TestToolFormatConversion:
    def setup_method(self):
        self.sample_tools = [
            {
                "name": "write_file",
                "description": "Write a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "File content"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "run_command",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],Config
                },
            },
        ]

    def test_anthropic_to_openai_conversion(self):
        result = _anthropic_tools_to_openai(self.sample_tools)
        assert len(result) == 2
        for tool in result:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    def test_openai_tools_have_descriptions(self):
        result = _anthropic_tools_to_openai(self.sample_tools)
        assert result[0]["function"]["description"] == "Write a file"

    def test_messages_to_openai_adds_system(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _messages_to_openai("You are helpful.", msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_messages_to_openai_basic_turn(self):
        msgs = [
            {"role": "user", "content": "Build me a thing"},
            {"role": "assistant", "content": [{"type": "text", "text": "Sure!"}]},
        ]
        result = _messages_to_openai("system", msgs)
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Sure!"

    def test_messages_to_openai_tool_use_blocks(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Writing file..."},
                    {
                        "type": "tool_use",
                        "id": "tu_123",
                        "name": "write_file",
                        "input": {"path": "main.py", "content": "print('hi')"},
                    },
                ],
            }
        ]
        result = _messages_to_openai("sys", msgs)
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "write_file"

    def test_messages_to_openai_tool_results(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_123",
                        "content": '{"status": "written"}',
                    }
                ],
            }
        ]
        result = _messages_to_openai("sys", msgs)
        tool_result_msg = result[1]
        assert tool_result_msg["role"] == "tool"
        assert tool_result_msg["tool_call_id"] == "tu_123"


# ── Config System ────────────────────────────────────────────────────────────

class TestForgeConfig:
    def test_load_returns_default_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("codeforge.config.CONFIG_FILE", tmp_path / "nonexistent.json")
        monkeypatch.setattr("codeforge.config.CONFIG_DIR", tmp_path)
        config = ForgeConfig.load()
        assert config.active_provider == ""
        assert config.onboarding_complete is False

    def test_set_provider_uses_default_model(self):
        config = ForgeConfig()
        config.set_provider("anthropic", api_key="sk-test")
        assert config.providers["anthropic"].model == "claude-opus-4-6"

    def test_set_provider_respects_explicit_model(self):
        config = ForgeConfig()
        config.set_provider("openai", api_key="sk-test", model="gpt-4o-mini")
        assert config.providers["openai"].model == "gpt-4o-mini"

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.json"
        monkeypatch.setattr("codeforge.config.CONFIG_FILE", cfg_file)
        monkeypatch.setattr("codeforge.config.CONFIG_DIR", tmp_path)

        config = ForgeConfig()
        config.set_provider("anthropic", api_key="sk-ant-test", model="claude-opus-4-6")
        config.active_provider = "anthropic"
        config.onboarding_complete = True
        config.default_max_cycles = 42
        config.save()

        loaded = ForgeConfig.load()
        assert loaded.active_provider == "anthropic"
        assert loaded.providers["anthropic"].api_key == "sk-ant-test"
        assert loaded.default_max_cycles == 42
        assert loaded.onboarding_complete is True

    def test_get_active_returns_none_when_unset(self):
        config = ForgeConfig()
        assert config.get_active() is None

    def test_get_active_returns_provider(self):
        config = ForgeConfig()
        config.set_provider("openai", api_key="sk-test", model="gpt-4o")
        config.active_provider = "openai"
        active = config.get_active()
        assert active is not None
        assert active.provider_id == "openai"

    def test_summary_line_no_provider(self):
        config = ForgeConfig()
        assert "No provider" in config.summary_line()

    def test_summary_line_with_provider(self):
        config = ForgeConfig()
        config.set_provider("anthropic", api_key="sk-test", model="claude-opus-4-6")
        config.active_provider = "anthropic"
        line = config.summary_line()
        assert "Anthropic" in line
        assert "claude-opus-4-6" in line


# ── ProviderConfig.is_ready ──────────────────────────────────────────────────

class TestProviderConfigIsReady:
    def test_anthropic_not_ready_without_key(self):
        pc = ProviderConfig(provider_id="anthropic", api_key="", model="claude-opus-4-6")
        assert pc.is_ready() is False

    def test_anthropic_not_ready_without_model(self):
        pc = ProviderConfig(provider_id="anthropic", api_key="sk-test", model="")
        assert pc.is_ready() is False

    def test_anthropic_ready(self):
        pc = ProviderConfig(provider_id="anthropic", api_key="sk-test", model="claude-opus-4-6")
        assert pc.is_ready() is True

    def test_ollama_ready_without_api_key(self):
        pc = ProviderConfig(provider_id="ollama", api_key="", model="qwen2.5-coder:32b", base_url="http://localhost:11434")
        assert pc.is_ready() is True

    def test_ollama_not_ready_without_base_url(self):
        pc = ProviderConfig(provider_id="ollama", api_key="", model="qwen2.5-coder:32b", base_url="")
        assert pc.is_ready() is False

    def test_openrouter_ready(self):
        pc = ProviderConfig(provider_id="openrouter", api_key="sk-or-test", model="anthropic/claude-opus-4-6")
        assert pc.is_ready() is True

    def test_custom_needs_base_url(self):
        pc = ProviderConfig(provider_id="custom", api_key="key", model="my-model", base_url="")
        assert pc.is_ready() is False

    def test_custom_ready_with_base_url(self):
        pc = ProviderConfig(provider_id="custom", api_key="key", model="my-model", base_url="https://my-endpoint.com/v1")
        assert pc.is_ready() is True


# ── ForgeResponse ────────────────────────────────────────────────────────────

class TestForgeResponse:
    def test_basic_construction(self):
        r = ForgeResponse(
            content_blocks=[{"type": "text", "text": "hello"}],
            stop_reason="end_turn",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=50,
        )
        assert r.stop_reason == "end_turn"
        assert r.input_tokens == 100

    def test_tool_use_response(self):
        r = ForgeResponse(
            content_blocks=[
                {"type": "tool_use", "id": "abc", "name": "write_file", "input": {"path": "x.py", "content": ""}}
            ],
            stop_reason="tool_use",
            model="gpt-4o",
        )
        assert r.stop_reason == "tool_use"
        assert r.content_blocks[0]["name"] == "write_file"
