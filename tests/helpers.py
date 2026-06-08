"""Shared test helpers: build openai-SDK-shaped mock objects and install a fake
client so the proxy can be exercised without a real upstream."""

from __future__ import annotations

import secrets
from types import SimpleNamespace

import bluellm.providers.openai_like as ol

# Randomized per-run test credentials so the suite never embeds
# credential-pattern literals (H11). Tests set these into the env vars that
# CONFIG_TEMPLATE resolves via os.environ/.
MASTER_KEY = secrets.token_urlsafe(16)
AZURE_KEY = secrets.token_urlsafe(16)
SALT_KEY = secrets.token_urlsafe(16)

CONFIG_TEMPLATE = """
models:
  - name: "*"
    params:
      model: azure/gpt-5.4
      endpoint: https://example.openai.azure.com
      key: os.environ/AZURE_API_KEY
      version: "2025-01-01-preview"
generals:
  key: os.environ/BLUELLM_MASTER_KEY
"""


def usage(prompt=10, completion=5):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=None
    )


def stream_chunk(content=None, finish_reason=None, tool_calls=None, use=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls, role=None, function_call=None
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id="c1", model="gpt-5.4", choices=[choice], usage=use)


def usage_only_chunk(use):
    return SimpleNamespace(id="c1", model="gpt-5.4", choices=[], usage=use)


def text_completion(text="Hello there", finish_reason="stop", tool_calls=None):
    content = None if tool_calls else text
    msg = SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls, function_call=None
    )
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
    return SimpleNamespace(
        id="cmpl-x", model="gpt-5.4", choices=[choice], usage=usage(12, 7)
    )


def tool_call(name="Read", arguments='{"path":"foo.py"}', call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def install_fake_client(monkeypatch, create_fn):
    """Replace the provider's client with one whose chat.completions.create is
    the supplied async ``create_fn(**kwargs)``."""

    class FakeCompletions:
        async def create(self, **kwargs):
            return await create_fn(**kwargs)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(
        ol.OpenAILikeProvider, "_build_client", lambda self: FakeClient()
    )
    ol._PROVIDER_CACHE.clear()
