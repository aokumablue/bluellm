"""bluellm.reliability のリトライ判定・バックオフ・試行回数のテスト。"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from bluellm.reliability import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    call_with_retry,
    is_retryable,
)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://example.test/v1/chat/completions")


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_request())


def test_default_policy_is_sdk_equivalent():
    # 既存挙動（SDK 既定の 2 リトライ）を弱めない: max_attempts=3 = 2 リトライ。
    assert DEFAULT_RETRY_POLICY.max_attempts == 3


def test_is_retryable_true_for_transient_errors():
    assert is_retryable(APITimeoutError(request=_request()))
    assert is_retryable(APIConnectionError(request=_request()))
    assert is_retryable(RateLimitError("rl", response=_response(429), body=None))
    assert is_retryable(InternalServerError("ise", response=_response(500), body=None))


def test_is_retryable_false_for_client_errors():
    assert not is_retryable(BadRequestError("bad", response=_response(400), body=None))
    assert not is_retryable(
        AuthenticationError("auth", response=_response(401), body=None)
    )
    assert not is_retryable(ValueError("nope"))


def test_call_with_retry_succeeds_first_try():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        return "ok"

    async def no_sleep(_):  # pragma: no cover - 成功時は呼ばれない
        raise AssertionError("sleep should not be called on success")

    result = asyncio.run(call_with_retry(factory, sleep=no_sleep))
    assert result == "ok"
    assert calls["n"] == 1


def test_call_with_retry_retries_then_succeeds():
    calls = {"n": 0}
    sleeps: list[float] = []

    async def factory():
        calls["n"] += 1
        if calls["n"] < 3:
            raise APITimeoutError(request=_request())
        return "ok"

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    policy = RetryPolicy(
        max_attempts=3, initial_backoff_ms=100, max_backoff_ms=1000, jitter_ratio=0.0
    )
    result = asyncio.run(
        call_with_retry(factory, policy, sleep=fake_sleep, rng=lambda: 0.5)
    )
    assert result == "ok"
    assert calls["n"] == 3
    # jitter_ratio=0 → 100ms, 200ms（指数バックオフ）
    assert sleeps == [0.1, 0.2]


def test_call_with_retry_exhausts_and_raises():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        raise InternalServerError("ise", response=_response(500), body=None)

    async def fake_sleep(_):
        pass

    policy = RetryPolicy(max_attempts=2, jitter_ratio=0.0)
    with pytest.raises(InternalServerError):
        asyncio.run(
            call_with_retry(factory, policy, sleep=fake_sleep, rng=lambda: 0.5)
        )
    assert calls["n"] == 2


def test_call_with_retry_does_not_retry_non_retryable():
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        raise BadRequestError("bad", response=_response(400), body=None)

    async def fake_sleep(_):  # pragma: no cover - 非 retryable では呼ばれない
        raise AssertionError("sleep should not be called")

    with pytest.raises(BadRequestError):
        asyncio.run(call_with_retry(factory, sleep=fake_sleep))
    assert calls["n"] == 1


def test_call_with_retry_backoff_capped_by_max():
    sleeps: list[float] = []

    async def factory():
        raise APITimeoutError(request=_request())

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    policy = RetryPolicy(
        max_attempts=4, initial_backoff_ms=1000, max_backoff_ms=1500, jitter_ratio=0.0
    )
    with pytest.raises(APITimeoutError):
        asyncio.run(
            call_with_retry(factory, policy, sleep=fake_sleep, rng=lambda: 0.5)
        )
    # 1000ms, min(1500, 2000)=1500ms, min(1500, 4000)=1500ms
    assert sleeps == [1.0, 1.5, 1.5]


def test_call_with_retry_jitter_uses_rng():
    sleeps: list[float] = []

    async def factory():
        raise APITimeoutError(request=_request())

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    policy = RetryPolicy(
        max_attempts=2, initial_backoff_ms=1000, max_backoff_ms=10000, jitter_ratio=0.2
    )
    # rng=0.0 → jitter = 1 + 0.2*(2*0 - 1) = 0.8 → 1000ms * 0.8 = 800ms
    with pytest.raises(APITimeoutError):
        asyncio.run(
            call_with_retry(factory, policy, sleep=fake_sleep, rng=lambda: 0.0)
        )
    assert sleeps == [pytest.approx(0.8)]
