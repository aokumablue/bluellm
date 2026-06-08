"""Provider-layer wire-compatibility tests.

The translator emits Anthropic-shaped residue (cache_control, thinking,
stop_sequences) because its ``is_anthropic_claude_model`` check fires on the
incoming ``claude-*`` model name even though every bluellm upstream is a Chat
Completions backend. The provider's ``_prepare`` is the bluellm-specific seam
that finalizes the request for the OpenAI/Azure wire format, so it must scrub
that residue. These tests pin that contract.
"""

from __future__ import annotations

import pytest

from bluellm.config import ModelConfig
from bluellm.providers.openai_like import OpenAILikeProvider


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(OpenAILikeProvider, "_build_client", lambda self: None)
    mc = ModelConfig(
        model_name="*",
        provider="azure",
        deployment="gpt-5.4",
        api_base="https://example.openai.azure.com",
        api_key="k",  # nosec - synthetic placeholder, not a real credential
        api_version="2025-01-01-preview",
    )
    return OpenAILikeProvider(mc)


def test_cache_control_stripped_from_message_top_level_and_content_parts(provider):
    # H1: cache_control must not reach an OpenAI/Azure Chat Completions upstream.
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ],
                "cache_control": {"type": "ephemeral"},
            },
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
                ],
            },
        ],
    }
    out = provider._prepare(req)
    user_msg = out["messages"][0]
    assert "cache_control" not in user_msg
    assert "cache_control" not in user_msg["content"][0]
    sys_msg = out["messages"][1]
    assert "cache_control" not in sys_msg["content"][0]


def test_cache_control_stripped_from_tools(provider):
    # H1: cache_control on tool params also leaks (translator adds it there too).
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "f"},
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    out = provider._prepare(req)
    assert "cache_control" not in out["tools"][0]


def test_thinking_rewritten_to_reasoning_effort(provider):
    # M1: Anthropic `thinking` must become OpenAI `reasoning_effort`.
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    }
    out = provider._prepare(req)
    assert "thinking" not in out
    assert out.get("reasoning_effort") == "high"


def test_thinking_disabled_is_dropped_without_reasoning_effort(provider):
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    }
    out = provider._prepare(req)
    assert "thinking" not in out
    assert "reasoning_effort" not in out


def test_thinking_summary_preserved_on_claude_path(provider):
    # M1: a thinking summary must survive the claude-* -> reasoning_effort
    # conversion instead of being dropped to a bare effort string.
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 10000, "summary": "detailed"},
    }
    out = provider._prepare(req)
    assert out["reasoning_effort"] == {"effort": "high", "summary": "detailed"}


def test_thinking_auto_summary_env_on_claude_path(provider, monkeypatch):
    monkeypatch.setenv("BLUELLM_REASONING_AUTO_SUMMARY", "true")
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    }
    out = provider._prepare(req)
    assert out["reasoning_effort"] == {"effort": "high", "summary": "detailed"}


def test_explicit_reasoning_effort_not_overwritten_by_thinking(provider):
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "reasoning_effort": "low",
    }
    out = provider._prepare(req)
    assert out.get("reasoning_effort") == "low"
    assert "thinking" not in out


def test_stop_sequences_rewritten_to_stop(provider):
    # M2: Anthropic `stop_sequences` must become OpenAI `stop`.
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["STOP"],
    }
    out = provider._prepare(req)
    assert "stop_sequences" not in out
    assert out.get("stop") == ["STOP"]


def test_existing_stop_not_overwritten(provider):
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["A"],
        "stop": ["B"],
    }
    out = provider._prepare(req)
    assert out.get("stop") == ["B"]
    assert "stop_sequences" not in out


def test_top_k_dropped(provider):
    # H2: Anthropic `top_k` has no Chat Completions equivalent. Drop it in
    # _prepare so the upstream does not 400 and we avoid the auto-drop retry
    # round-trip.
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 40,
    }
    out = provider._prepare(req)
    assert "top_k" not in out


def test_cache_key_excludes_plaintext_api_key():
    # The provider cache key lives in a long-lived module global; the raw
    # api_key must not be stored in it.
    from bluellm.providers.openai_like import _cache_key

    mc = ModelConfig(
        model_name="*",
        provider="azure",
        deployment="d",
        api_base="https://x",
        api_key="PLAINTEXT-SECRET-KEY",  # nosec - synthetic value asserted to be redacted
        api_version="v",
    )
    key = _cache_key(mc)
    assert "PLAINTEXT-SECRET-KEY" not in key
    # a rotated key still produces a distinct cache entry
    mc2 = ModelConfig(
        model_name="*",
        provider="azure",
        deployment="d",
        api_base="https://x",
        api_key="ROTATED-KEY",  # nosec - synthetic rotated value for cache-key test
        api_version="v",
    )
    assert _cache_key(mc) != _cache_key(mc2)
