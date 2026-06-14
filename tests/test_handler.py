"""handler.process の fallback / リトライ統合のテスト。"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from helpers import install_fake_client, stream_chunk, text_completion, usage, usage_only_chunk
from openai import APIConnectionError, BadRequestError, InternalServerError

from bluellm import handler
from bluellm.config import ModelConfig
from bluellm.reliability import RetryPolicy

# fallback の遷移自体を見たいので、各モデルのリトライは 1 回（即失敗→次候補）。
_NO_RETRY = RetryPolicy(max_attempts=1)

_BODY = {
    "model": "claude-x",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "hi"}],
}


def _mc(name: str, deployment: str, fallback_to: str | None = None) -> ModelConfig:
    return ModelConfig(
        model_name=name,
        provider="azure",
        deployment=deployment,
        api_base="https://example.openai.azure.com",
        api_key="k",  # nosec - synthetic placeholder
        api_version="v",
        retry=_NO_RETRY,
        fallback_to=fallback_to,
    )


def _response(status: int) -> httpx.Response:
    return httpx.Response(
        status, request=httpx.Request("POST", "https://example.test")
    )


async def _drain(aiter) -> list[bytes]:
    return [chunk async for chunk in aiter]


def test_falls_back_on_retryable_error(monkeypatch):
    calls: list[str] = []

    async def create(**kw):
        calls.append(kw["model"])
        if kw["model"] == "primary-dep":
            raise InternalServerError("ise", response=_response(500), body=None)
        return text_completion()

    install_fake_client(monkeypatch, create)
    is_stream, payload = asyncio.run(
        handler.process(_BODY, [_mc("primary", "primary-dep"), _mc("secondary", "secondary-dep")])
    )
    assert is_stream is False
    assert payload["usage"] == {"input_tokens": 12, "output_tokens": 7}
    assert calls == ["primary-dep", "secondary-dep"]


def test_no_fallback_on_client_error(monkeypatch):
    calls: list[str] = []

    async def create(**kw):
        calls.append(kw["model"])
        raise BadRequestError("bad", response=_response(400), body=None)

    install_fake_client(monkeypatch, create)
    with pytest.raises(BadRequestError):
        asyncio.run(
            handler.process(
                _BODY, [_mc("primary", "primary-dep"), _mc("secondary", "secondary-dep")]
            )
        )
    # 恒久エラーは fallback しない（primary のみ）。
    assert calls == ["primary-dep"]


def test_last_candidate_error_propagates(monkeypatch):
    calls: list[str] = []

    async def create(**kw):
        calls.append(kw["model"])
        raise InternalServerError("ise", response=_response(500), body=None)

    install_fake_client(monkeypatch, create)
    with pytest.raises(InternalServerError):
        asyncio.run(
            handler.process(
                _BODY, [_mc("primary", "primary-dep"), _mc("secondary", "secondary-dep")]
            )
        )
    # 全候補が retryable で失敗 → 最終候補のエラーを送出。
    assert calls == ["primary-dep", "secondary-dep"]


def test_stream_fallback(monkeypatch):
    calls: list[str] = []

    async def create(**kw):
        calls.append(kw["model"])
        if kw["model"] == "primary-dep":
            raise APIConnectionError(
                request=httpx.Request("POST", "https://example.test")
            )

        async def gen():
            yield stream_chunk(content="Hi")
            yield stream_chunk(finish_reason="stop")
            yield usage_only_chunk(usage(8, 2))

        return gen()

    install_fake_client(monkeypatch, create)
    body = {**_BODY, "stream": True}
    is_stream, payload = asyncio.run(
        handler.process(body, [_mc("primary", "primary-dep"), _mc("secondary", "secondary-dep")])
    )
    assert is_stream is True
    chunks = asyncio.run(_drain(payload))
    assert calls == ["primary-dep", "secondary-dep"]
    assert b"message_start" in b"".join(chunks)
