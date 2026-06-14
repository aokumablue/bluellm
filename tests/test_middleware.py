"""HTTP 境界 middleware（レート制限・IP allowlist）のテスト。

dispatch は fake request で直接駆動する（TestClient の client.host が IP では
ないため allowlist 判定を素直に確認できる）。
"""

from __future__ import annotations

import asyncio

from fastapi.responses import JSONResponse

from bluellm.middleware import (
    TokenBucket,
    allowlist_middleware,
    runaway_guard_middleware,
)


def _err(status: int, err_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


class _URL:
    def __init__(self, path: str) -> None:
        self.path = path


class _Client:
    def __init__(self, host: str) -> None:
        self.host = host


class _Request:
    def __init__(self, path="/v1/messages", headers=None, host="1.2.3.4") -> None:
        self.url = _URL(path)
        self.headers = headers or {}
        self.client = _Client(host) if host is not None else None


async def _ok(_request):
    return "OK"


def _run(dispatch, request):
    return asyncio.run(dispatch(request, _ok))


# --- TokenBucket ------------------------------------------------------------


def test_token_bucket_allows_within_burst():
    bucket = TokenBucket(rps=1.0, burst=2, monotonic=lambda: 0.0)
    assert bucket.take() == (True, 0.0)
    assert bucket.take() == (True, 0.0)
    allowed, retry_after = bucket.take()
    assert allowed is False and retry_after > 0


def test_token_bucket_refills_over_time():
    clock = {"t": 0.0}
    bucket = TokenBucket(rps=2.0, burst=2, monotonic=lambda: clock["t"])
    bucket.take()
    bucket.take()  # 空
    clock["t"] = 1.0  # +1s → +2 トークン
    assert bucket.take() == (True, 0.0)


def test_token_bucket_retry_after_when_rps_zero():
    bucket = TokenBucket(rps=0.0, burst=0, monotonic=lambda: 0.0)
    allowed, retry_after = bucket.take()
    assert allowed is False and retry_after == 1.0


# --- runaway guard ----------------------------------------------------------


def test_rate_limit_blocks_when_exhausted():
    # rps=0.0 → burst=max(1, ceil(0))=1。1 回目 OK・2 回目で枯渇して 429。
    disp = runaway_guard_middleware(rps=0.0, error_builder=_err)
    req = _Request()
    assert _run(disp, req) == "OK"
    resp = _run(disp, req)
    assert isinstance(resp, JSONResponse) and resp.status_code == 429
    assert resp.headers["Retry-After"] == "1"


def test_rate_limit_bypasses_health():
    disp = runaway_guard_middleware(rps=0.0, error_builder=_err)
    assert _run(disp, _Request(path="/health")) == "OK"
    assert _run(disp, _Request(path="/")) == "OK"


# --- IP allowlist -----------------------------------------------------------


def test_allowlist_empty_allows_all():
    disp = allowlist_middleware([], _err)
    assert _run(disp, _Request(host="9.9.9.9")) == "OK"


def test_allowlist_allows_in_range():
    disp = allowlist_middleware(["10.0.0.0/8"], _err)
    assert _run(disp, _Request(host="10.1.2.3")) == "OK"


def test_allowlist_blocks_out_of_range():
    disp = allowlist_middleware(["10.0.0.0/8"], _err)
    resp = _run(disp, _Request(host="192.168.1.1"))
    assert isinstance(resp, JSONResponse) and resp.status_code == 403


def test_allowlist_blocks_unparseable_host():
    disp = allowlist_middleware(["10.0.0.0/8"], _err)
    resp = _run(disp, _Request(host="testclient"))
    assert resp.status_code == 403


def test_allowlist_blocks_when_no_client():
    disp = allowlist_middleware(["10.0.0.0/8"], _err)
    resp = _run(disp, _Request(host=None))
    assert resp.status_code == 403


def test_allowlist_bypasses_health():
    disp = allowlist_middleware(["10.0.0.0/8"], _err)
    assert _run(disp, _Request(path="/health", host="9.9.9.9")) == "OK"
