"""HTTP 境界の防御 middleware: 単一グローバルの runaway ガードと CIDR allowlist。

いずれも標準ライブラリのみで実装する。

- runaway ガードはプロセス全体で 1 個のトークンバケットを共有し、寛容な既定値で
  常時 ON とする。単一ユーザー（Claude Code）の通常利用を妨げず、暴走（runaway）の
  みを抑止する。per-token のバケット分割やキー別の課金は行わない。
- allowlist は既定空リスト＝全許可（既存挙動と等価）。明示的に CIDR を設定した
  ときだけブロックする。

``/health`` / ``/`` は認証不要プローブなので境界チェックもバイパスする。

レスポンス生成（Anthropic エラー形式）は ``server`` 側から builder を注入して
循環 import を避ける。``TokenBucket.take`` は同期かつ ``await`` を含まないため、
単一イベントループ上ではアトミックでロック不要。
"""

from __future__ import annotations

import ipaddress
import math
import time
from typing import Callable, List, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse

# health / root は認証不要プローブなので境界チェックをバイパスする。
_BYPASS_PATHS = frozenset({"/health", "/"})

# Anthropic エラー JSON を生成する builder（server._anthropic_error）の型。
ErrorBuilder = Callable[[int, str, str], JSONResponse]


class TokenBucket:
    """単純なトークンバケット。``rps`` で補充、``burst`` を上限とする。"""

    def __init__(
        self, rps: float, burst: int, *, monotonic: Callable[[], float] = time.monotonic
    ) -> None:
        """``rps`` トークン/秒で補充し、容量 ``burst`` で頭打ちするバケットを作る。"""
        self._rps = rps
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._monotonic = monotonic
        self._updated = monotonic()

    def take(self, cost: float = 1.0) -> Tuple[bool, float]:
        """``cost`` トークンを消費しようと試みる。

        経過時間に応じて補充してから判定する。許可できれば ``(True, 0.0)``、
        できなければ ``(False, retry_after_seconds)`` を返す。
        """
        now = self._monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rps)
        if self._tokens >= cost:
            self._tokens -= cost
            return True, 0.0
        deficit = cost - self._tokens
        retry_after = deficit / self._rps if self._rps > 0 else 1.0
        return False, retry_after


def runaway_guard_middleware(rps: float, error_builder: ErrorBuilder) -> Callable:
    """単一グローバルの runaway ガード dispatch を生成する。

    プロセス全体で 1 個のトークンバケットを共有し、``rps`` トークン/秒で補充する。
    バケット容量（burst）は config では公開せず内部既定とする。``rps`` と同
    オーダーの ``burst = max(1, math.ceil(rps))`` を採用するのは、これが
    「単一ユーザー（Claude Code）を妨げず暴走のみ抑止する」という寛容な目的に
    合致するため。短時間のバースト（瞬間的に rps を超える連続リクエスト）は許容
    しつつ、持続的な暴走だけを定常レート ``rps`` で頭打ちにする。``max(1, ...)``
    は ``rps`` が 0 でも最低 1 トークンを確保し、初回リクエストを通す。

    超過時は 429（``rate_limit_error``）に ``Retry-After`` ヘッダーを付けて返す。
    """
    burst = max(1, math.ceil(rps))
    bucket = TokenBucket(rps, burst)

    async def dispatch(request: Request, call_next):
        if request.url.path in _BYPASS_PATHS:
            return await call_next(request)
        allowed, retry_after = bucket.take()
        if not allowed:
            resp = error_builder(429, "rate_limit_error", "rate limit exceeded")
            resp.headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
            return resp
        return await call_next(request)

    return dispatch


def _ip_allowed(host: str, networks: List[ipaddress._BaseNetwork]) -> bool:
    """``host`` が ``networks`` のいずれかに含まれるかを判定する。"""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(ip in net for net in networks)


def allowlist_middleware(cidrs: List[str], error_builder: ErrorBuilder) -> Callable:
    """CIDR allowlist による IP フィルタ dispatch を生成する。

    ``cidrs`` が空なら全許可（無効）。直接 peer（``request.client.host``）のみ
    判定し、スプーフ可能な X-Forwarded-For は参照しない。非許可は 403。
    """
    networks = [ipaddress.ip_network(c, strict=False) for c in cidrs]

    async def dispatch(request: Request, call_next):
        if not networks or request.url.path in _BYPASS_PATHS:
            return await call_next(request)
        client = request.client
        host = client.host if client else None
        if host is None or not _ip_allowed(host, networks):
            return error_builder(403, "permission_error", "client address not allowed")
        return await call_next(request)

    return dispatch
