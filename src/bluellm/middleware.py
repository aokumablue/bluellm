"""HTTP 境界の防御 middleware: トークンバケットによるレート制限と CIDR allowlist。

いずれも標準ライブラリのみで実装する。

- レート制限は寛容な既定値で常時 ON とし、単一ユーザー（Claude Code）の通常利用を
  妨げず runaway のみ抑止する。
- allowlist は既定空リスト＝全許可（既存挙動と等価）。明示的に CIDR を設定した
  ときだけブロックする。

``/health`` / ``/`` は認証不要プローブなので境界チェックもバイパスする。

レスポンス生成（Anthropic エラー形式）は ``server`` 側から builder を注入して
循環 import を避ける。``TokenBucket.take`` / バケット辞書操作は同期かつ ``await`` を
含まないため、単一イベントループ上ではアトミックでロック不要。
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse

# health / root は認証不要プローブなので境界チェックをバイパスする。
_BYPASS_PATHS = frozenset({"/health", "/"})

# per-token バケットのメモリ有界化（LRU 上限）。
_MAX_BUCKETS = 1024

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


def _extract_token(request: Request) -> Optional[str]:
    """``x-api-key`` または ``Authorization: Bearer`` からトークンを取り出す。"""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[len("bearer ") :]
    return None


def _rate_key(request: Request, per_token: bool) -> str:
    """レート制限のバケットキーを決定する（per-token 時はトークンのハッシュ）。"""
    if per_token:
        token = _extract_token(request)
        if token:
            return "tok:" + hashlib.sha256(token.encode()).hexdigest()
    return "global"


def rate_limit_middleware(
    rps: float, burst: int, per_token: bool, error_builder: ErrorBuilder
) -> Callable:
    """トークンバケットによるレート制限 dispatch を生成する。

    超過時は 429（``rate_limit_error``）に ``Retry-After`` ヘッダーを付けて返す。
    """
    buckets: "OrderedDict[str, TokenBucket]" = OrderedDict()

    def _bucket_for(key: str) -> TokenBucket:
        bucket = buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(rps, burst)
            buckets[key] = bucket
            if len(buckets) > _MAX_BUCKETS:
                buckets.popitem(last=False)
        else:
            buckets.move_to_end(key)
        return bucket

    async def dispatch(request: Request, call_next):
        if request.url.path in _BYPASS_PATHS:
            return await call_next(request)
        allowed, retry_after = _bucket_for(_rate_key(request, per_token)).take()
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
