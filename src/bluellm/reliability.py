"""上流呼び出しの信頼性ヘルパ: リトライ判定（指数バックオフ＋ジッタ）と実行。

OpenAI SDK 内蔵リトライはクライアント構築時に ``max_retries=0`` で無効化し、
リトライをここに一本化する。これによりリトライ回数・バックオフ・ジッタを
設定可能にし、handler 側のプロバイダー/モデル fallback と統合できる。

retry 未設定のモデルは :data:`DEFAULT_RETRY_POLICY`（SDK 既定相当の 2 リトライ）を
使うため、既存挙動（SDK の自動リトライ）を弱めない。
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """リトライの試行回数・バックオフ上下限・ジッタ比を表す不変ポリシー。"""

    max_attempts: int = 3
    initial_backoff_ms: int = 500
    max_backoff_ms: int = 8000
    jitter_ratio: float = 0.2


# OpenAI SDK 既定相当（2 リトライ = 3 試行・指数バックオフ）。retry 未設定の
# モデルはこれを使い、既存挙動（SDK の自動リトライ）を弱めない。
DEFAULT_RETRY_POLICY = RetryPolicy()


def is_retryable(exc: BaseException) -> bool:
    """``exc`` が一時障害（再試行で回復しうる）かを判定する。

    接続失敗・タイムアウト・429（rate limit）・5xx を再試行可とし、
    400/401 などの恒久的なクライアントエラーは再試行しない。
    """
    if isinstance(
        exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


async def call_with_retry(
    factory: Callable[[], Awaitable[T]],
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """``factory`` を最大 ``policy.max_attempts`` 回呼び出す。

    retryable な例外（:func:`is_retryable`）の場合のみ、指数バックオフ
    （``initial_backoff_ms * 2**n`` を ``max_backoff_ms`` で頭打ち）に
    ジッタ（``1 ± jitter_ratio``）を乗じた秒数だけ待機して再試行する。
    最終試行でも失敗した場合、または retryable でない例外の場合はその例外を
    送出する。``sleep`` / ``rng`` はテスト用に注入できる。
    """
    attempt = 0
    while True:
        try:
            return await factory()
        except Exception as exc:
            attempt += 1
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            base_ms = min(
                policy.max_backoff_ms,
                policy.initial_backoff_ms * (2 ** (attempt - 1)),
            )
            jitter = 1.0 + policy.jitter_ratio * (2.0 * rng() - 1.0)
            await sleep(base_ms * jitter / 1000.0)
