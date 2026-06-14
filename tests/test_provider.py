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
from bluellm.reliability import RetryPolicy


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


def test_stop_top_level_dropped_when_no_stop_sequences(provider):
    # M2: 上流が Chat Completions 用の `stop` を誤ってトップレベルで渡す場合
    # Azure 側で 400 になりやすいため、`stop_sequences` が無ければ落とす。
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "BREAK",
    }
    out = provider._prepare(req)
    assert out.get("stop") is None


def test_stop_sequences_wrong_type_is_dropped(provider):
    # M2: `stop_sequences` は list[str] だけ有効。
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": "not-a-list",
    }
    out = provider._prepare(req)
    assert out.get("stop") is None


_SENTINEL = object()


@pytest.mark.parametrize(
    "input_kwargs, expected_stop",
    [
        # stop_sequences 無し: stop は触らない
        ({}, _SENTINEL),
        # stop_sequences が list[str]: stop に昇格
        ({"stop_sequences": ["X"]}, ["X"]),
        # stop_sequences が不正値: stop も落とす
        ({"stop_sequences": "bad"}, _SENTINEL),
        ({"stop_sequences": [1, 2]}, _SENTINEL),
        # 直接 stop（第2ブロックのみが検証する経路）
        ({"stop": ["S"]}, ["S"]),
        ({"stop": "BREAK"}, _SENTINEL),
        # 組合せ: 既存 stop を優先（setdefault）
        ({"stop_sequences": ["A"], "stop": ["B"]}, ["B"]),
        # 組合せ: 不正 stop_sequences は直接 stop も落とす
        ({"stop_sequences": "bad", "stop": ["B"]}, _SENTINEL),
        # 組合せ: 有効 stop_sequences + 不正 stop（setdefault 不発、第2ブロックで落ちる）
        ({"stop_sequences": ["A"], "stop": "bad"}, _SENTINEL),
        # M-1: 空 stop_sequences → 指定なし扱い（stop 設定なし）
        ({"stop_sequences": []}, _SENTINEL),
        # M-1: 空 stop → 削除
        ({"stop": []}, _SENTINEL),
        # M-1: 空 stop_sequences は指定なし扱い、直接 stop は維持
        ({"stop_sequences": [], "stop": ["S"]}, ["S"]),
    ],
)
def test_normalize_stop_param_matches_inline_logic(input_kwargs, expected_stop):
    """抽出した `_normalize_stop_param` が抽出前のインラインロジックと完全一致することを証明する。

    各入力に対して reference 実装（抽出前のコードをそのまま再現）と
    `_normalize_stop_param` の結果（`stop` の有無・値、および `stop_sequences` 除去）が
    バイト一致することを確認する。
    """
    from bluellm.providers.openai_like import _normalize_stop_param

    def reference(req):
        stop_sequences = req.pop("stop_sequences", None)
        if stop_sequences:
            if isinstance(stop_sequences, list) and all(
                isinstance(x, str) for x in stop_sequences
            ):
                req.setdefault("stop", stop_sequences)
            else:
                req.pop("stop", None)
        stop = req.get("stop")
        if stop is not None:
            if isinstance(stop, list) and stop and all(isinstance(x, str) for x in stop):
                pass
            else:
                req.pop("stop", None)

    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    ref_req = {**base, **input_kwargs}
    reference(ref_req)

    new_req = {**base, **input_kwargs}
    _normalize_stop_param(new_req)

    # 抽出前後で req 全体がバイト一致
    assert new_req == ref_req
    # 期待値に対する明示的な pin
    assert new_req.get("stop", _SENTINEL) == expected_stop
    assert "stop_sequences" not in new_req


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


def test_context_management_dropped(provider):
    # Claude Code sends `context_management` on the Anthropic Messages surface,
    # but the current upstream wire format is OpenAI/Azure Chat Completions,
    # whose SDK method rejects it as an unexpected keyword argument.
    req = {
        "model": "claude-sonnet-4",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "context_management": {"clear_function_results": True},
    }
    out = provider._prepare(req)
    assert "context_management" not in out


def test_build_client_azure_openai_v1_uses_async_openai(monkeypatch):
    # Azure's OpenAI-compatible /openai/v1 endpoint should use the plain OpenAI client.
    calls = {}

    def fake_async_openai(**kwargs):
        calls["async_openai"] = kwargs
        return "openai-client"

    def fake_async_azure_openai(**kwargs):
        calls["async_azure_openai"] = kwargs
        return "azure-client"

    monkeypatch.setattr("bluellm.providers.openai_like.AsyncOpenAI", fake_async_openai)
    monkeypatch.setattr(
        "bluellm.providers.openai_like.AsyncAzureOpenAI", fake_async_azure_openai
    )
    mc = ModelConfig(
        model_name="*",
        provider="azure",
        deployment="d",
        api_base="https://example.openai.azure.com/openai/v1",
        api_key="k",  # nosec - synthetic placeholder, not a real credential
        api_version="2025-01-01-preview",
    )

    provider = OpenAILikeProvider(mc)

    assert provider._client == "openai-client"
    assert calls["async_openai"] == {
        "api_key": "k",
        "base_url": "https://example.openai.azure.com/openai/v1",
        "max_retries": 0,
    }
    assert "async_azure_openai" not in calls


def test_build_client_azure_deployments_base_url_still_uses_async_azure_openai(monkeypatch):
    # Existing fully-qualified deployments URLs must keep using AsyncAzureOpenAI.
    calls = {}

    def fake_async_openai(**kwargs):
        calls["async_openai"] = kwargs
        return "openai-client"

    def fake_async_azure_openai(**kwargs):
        calls["async_azure_openai"] = kwargs
        return "azure-client"

    monkeypatch.setattr("bluellm.providers.openai_like.AsyncOpenAI", fake_async_openai)
    monkeypatch.setattr(
        "bluellm.providers.openai_like.AsyncAzureOpenAI", fake_async_azure_openai
    )
    mc = ModelConfig(
        model_name="*",
        provider="azure",
        deployment="d",
        api_base="https://example.openai.azure.com/openai/deployments/d",
        api_key="k",  # nosec - synthetic placeholder, not a real credential
        api_version="2025-01-01-preview",
    )

    provider = OpenAILikeProvider(mc)

    assert provider._client == "azure-client"
    assert calls["async_azure_openai"] == {
        "api_key": "k",
        "api_version": "2025-01-01-preview",
        "base_url": "https://example.openai.azure.com/openai/deployments/d",
        "max_retries": 0,
    }
    assert "async_openai" not in calls


def test_build_client_azure_plain_endpoint_uses_async_azure_openai(monkeypatch):
    # Legacy Azure endpoint routing must keep using AsyncAzureOpenAI with azure_endpoint.
    calls = {}

    def fake_async_openai(**kwargs):
        calls["async_openai"] = kwargs
        return "openai-client"

    def fake_async_azure_openai(**kwargs):
        calls["async_azure_openai"] = kwargs
        return "azure-client"

    monkeypatch.setattr("bluellm.providers.openai_like.AsyncOpenAI", fake_async_openai)
    monkeypatch.setattr(
        "bluellm.providers.openai_like.AsyncAzureOpenAI", fake_async_azure_openai
    )
    mc = ModelConfig(
        model_name="*",
        provider="azure",
        deployment="d",
        api_base="https://example.openai.azure.com",
        api_key="k",  # nosec - synthetic placeholder, not a real credential
        api_version="2025-01-01-preview",
    )

    provider = OpenAILikeProvider(mc)

    assert provider._client == "azure-client"
    assert calls["async_azure_openai"] == {
        "api_key": "k",
        "api_version": "2025-01-01-preview",
        "azure_endpoint": "https://example.openai.azure.com",
        "max_retries": 0,
    }
    assert "async_openai" not in calls


def test_build_client_ollama_uses_async_openai_with_defaults(monkeypatch):
    # Ollama exposes an OpenAI-compatible endpoint; build a plain AsyncOpenAI
    # client. With no endpoint/key configured, fall back to the local Ollama
    # base_url and a dummy api_key (the SDK requires a non-empty key).
    calls = {}

    def fake_async_openai(**kwargs):
        calls["async_openai"] = kwargs
        return "ollama-client"

    monkeypatch.setattr("bluellm.providers.openai_like.AsyncOpenAI", fake_async_openai)
    mc = ModelConfig(
        model_name="*",
        provider="ollama",
        deployment="llama3.3",
        api_base=None,
        api_key=None,
        api_version=None,
    )

    provider = OpenAILikeProvider(mc)

    assert provider._client == "ollama-client"
    assert calls["async_openai"] == {
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "max_retries": 0,
    }


def test_build_client_ollama_honors_explicit_endpoint_and_key(monkeypatch):
    # Explicit endpoint/key override the local Ollama defaults.
    calls = {}

    def fake_async_openai(**kwargs):
        calls["async_openai"] = kwargs
        return "ollama-client"

    monkeypatch.setattr("bluellm.providers.openai_like.AsyncOpenAI", fake_async_openai)
    mc = ModelConfig(
        model_name="*",
        provider="ollama",
        deployment="llama3.3",
        api_base="http://ollama.internal:11434/v1",
        api_key="tok",  # nosec - synthetic placeholder, not a real credential
        api_version=None,
    )

    OpenAILikeProvider(mc)

    assert calls["async_openai"] == {
        "api_key": "tok",
        "base_url": "http://ollama.internal:11434/v1",
        "max_retries": 0,
    }


def test_get_provider_accepts_ollama(monkeypatch):
    # ollama must be an accepted provider (not rejected as unsupported).
    from bluellm.providers.openai_like import get_provider

    monkeypatch.setattr(OpenAILikeProvider, "_build_client", lambda self: "c")
    mc = ModelConfig(
        model_name="*",
        provider="ollama",
        deployment="llama3.3",
        api_base="http://localhost:11434/v1",
        api_key=None,
        api_version=None,
    )
    provider = get_provider(mc)
    assert isinstance(provider, OpenAILikeProvider)


def test_build_client_includes_timeout_when_set(monkeypatch):
    # timeout を設定したモデルは openai クライアントへ timeout を渡す。
    # max_retries は常に 0（リトライは reliability に一本化）。
    calls = {}

    def fake_async_openai(**kwargs):
        calls.update(kwargs)
        return "openai-client"

    monkeypatch.setattr("bluellm.providers.openai_like.AsyncOpenAI", fake_async_openai)
    mc = ModelConfig(
        model_name="*",
        provider="openai",
        deployment="gpt-5.4",
        api_base=None,
        api_key="k",  # nosec - synthetic placeholder
        api_version=None,
        timeout=42.0,
    )
    OpenAILikeProvider(mc)
    assert calls["max_retries"] == 0
    assert calls["timeout"] == 42.0


def test_build_client_omits_timeout_when_unset(monkeypatch):
    # timeout 未設定なら timeout kwarg を渡さない（SDK 既定タイムアウトを維持）。
    calls = {}

    def fake_async_openai(**kwargs):
        calls.update(kwargs)
        return "openai-client"

    monkeypatch.setattr("bluellm.providers.openai_like.AsyncOpenAI", fake_async_openai)
    mc = ModelConfig(
        model_name="*",
        provider="openai",
        deployment="gpt-5.4",
        api_base=None,
        api_key="k",  # nosec - synthetic placeholder
        api_version=None,
    )
    OpenAILikeProvider(mc)
    assert "timeout" not in calls
    assert calls["max_retries"] == 0


def test_cache_key_differs_by_timeout_and_retry():
    # 接続挙動に影響する timeout / retry が異なれば別クライアントを返す。
    from bluellm.providers.openai_like import _cache_key

    def mc(**over):
        base = dict(
            model_name="*",
            provider="openai",
            deployment="d",
            api_base="https://x",
            api_key="k",  # nosec - synthetic placeholder
            api_version="v",
        )
        base.update(over)
        return ModelConfig(**base)

    k_base = _cache_key(mc())
    assert _cache_key(mc(timeout=10.0)) != k_base
    assert _cache_key(mc(retry=RetryPolicy(max_attempts=9))) != k_base
    # 同一設定は同一キー（共有される）。
    assert _cache_key(mc()) == k_base


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
