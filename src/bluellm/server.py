"""Claude Code が使用する Anthropic Messages API サーフェスを公開する FastAPI アプリ。"""

from __future__ import annotations

import json
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIStatusError

from bluellm import handler
from bluellm.auth import Authenticator, require_auth
from bluellm.config import Config, ModelConfig
from bluellm.cost import UsageLogger
from bluellm.middleware import allowlist_middleware, runaway_guard_middleware
from bluellm.observability import request_span
from bluellm.router import Router
from bluellm.translation import UnsupportedContentError

logger = logging.getLogger("bluellm")

# fallback 連鎖の最大長（暴走・過剰フェイルオーバーの安全上限）。
_MAX_FALLBACK_CHAIN = 5


def _resolve_fallback_chain(router: Router, model_name: str) -> list[ModelConfig]:
    """``model_name`` を起点に ``fallback_to`` を辿って ModelConfig 連鎖を構築する。

    先頭が primary、以降が fallback。``visited`` で循環を防ぎ、``_MAX_FALLBACK_CHAIN``
    で長さを上限化する。起点の解決失敗（KeyError）は呼び出し元へ伝播させる。
    """
    chain: list[ModelConfig] = []
    visited: set[str] = set()
    mc = router.resolve(model_name)
    while True:
        chain.append(mc)
        visited.add(mc.model_name)
        nxt = mc.fallback_to
        if not nxt or nxt in visited or len(chain) >= _MAX_FALLBACK_CHAIN:
            return chain
        try:
            mc = router.resolve(nxt)
        except KeyError:
            # 設定ミスの fallback_to は無視し、それまでの連鎖で続行する。
            logger.info("Fallback target %r not found; stopping chain", nxt)
            return chain


def _anthropic_error(status_code: int, err_type: str, message: str) -> JSONResponse:
    """Anthropic 形式のエラー JSON レスポンス（``{"type": "error", ...}``）を構築する。"""
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": err_type, "message": message}},
    )


def create_app(config: Config) -> FastAPI:
    """Anthropic Messages endpoint を公開する FastAPI アプリを構築する。

    ``config`` から router と authenticator を接続し、
    ``/health``、``/v1/messages``、``/v1/messages/count_tokens`` を登録する。
    """
    app = FastAPI(title="bluellm", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.router = Router(config)
    app.state.authenticator = Authenticator(config.general_settings.master_key)
    # usage トークン記録（常時有効・固定の日次 JSONL へ追記）。
    app.state.usage_logger = UsageLogger()

    max_bytes = config.general_settings.max_request_body_mb * 1024 * 1024

    @app.middleware("http")
    async def _limit_body_size(request: Request, call_next):
        """Content-Length が設定上限（max_request_body_mb）を超えるリクエストを 413 で拒否する（M-4 DoS 対策）。

        Content-Length ヘッダーが存在し数値の場合のみチェックする。
        ヘッダーが欠落している場合は通過させる（既存挙動と同じ、悪化なし）。
        """
        cl = request.headers.get("content-length")
        if cl is not None and cl.isdigit() and int(cl) > max_bytes:
            return _anthropic_error(413, "invalid_request_error", "request body too large")
        return await call_next(request)

    gs = config.general_settings
    # middleware は後に登録したものが最外（先に実行）になる。実行順を
    # allowlist -> runaway guard -> body size とするため、body size の後に
    # runaway guard、最後に allowlist を登録する。
    app.middleware("http")(
        runaway_guard_middleware(gs.runaway_guard_rps, _anthropic_error)
    )
    app.middleware("http")(allowlist_middleware(gs.allowlist_cidrs, _anthropic_error))

    @app.get("/health")
    async def health():
        """認証不要の死活監視プローブ。"""
        return {"status": "ok"}

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root():
        """外部プローブ向けのルート応答。"""
        return {"status": "ok"}

    @app.post("/v1/messages", dependencies=[Depends(require_auth)])
    async def messages(request: Request):
        """Anthropic Messages endpoint: リクエストを変換し、上流を呼び出し、レスポンスを変換する。

        JSON Messages レスポンスまたは SSE ストリームを返す。不正な JSON ->
        400、未知のモデル -> 404、変換不可のコンテンツ -> 400、
        上流エラーはサニタイズしてから返す。
        """
        try:
            body = await request.json()
        except Exception:
            return _anthropic_error(
                400, "invalid_request_error", "request body is not valid JSON"
            )
        try:
            model_configs = _resolve_fallback_chain(
                request.app.state.router, body.get("model", "")
            )
        except KeyError as e:
            # 本文に routing 内部構成（'*' catch-all の有無）やクライアント送信の
            # model 名を漏らさない。診断用の詳細はサーバーログにのみ残す。
            logger.info("Model routing failed: %s", e)
            return _anthropic_error(404, "not_found_error", "model not found")

        with request_span(
            "v1.messages",
            **{"bluellm.requested_model": body.get("model", "")},
        ) as span:
            try:
                is_stream, payload = await handler.process(
                    body, model_configs, request.app.state.usage_logger, span
                )
            except UnsupportedContentError as e:
                # クライアントがプロキシで変換できないコンテンツブロックを送信した
                # 場合（例: サポートされていない画像ソース）は、暗黙的に破棄せず拒否する。
                return _anthropic_error(400, "invalid_request_error", str(e))
            except Exception as e:  # サニタイズ: 上流の詳細/キーをクライアントに漏洩しない
                return _handle_upstream_error(e)

            if is_stream:
                return StreamingResponse(payload, media_type="text/event-stream")
            return JSONResponse(content=payload)

    @app.post("/v1/messages/count_tokens", dependencies=[Depends(require_auth)])
    async def count_tokens(request: Request):
        """messages + system ペイロードのおおよその token 数（文字数/4）を返す。"""
        try:
            body = await request.json()
        except Exception:
            return _anthropic_error(
                400, "invalid_request_error", "request body is not valid JSON"
            )
        text = json.dumps(body.get("messages", []), default=str)
        text += str(body.get("system", ""))
        return {"input_tokens": max(1, len(text) // 4)}

    return app


def _handle_upstream_error(e: Exception) -> JSONResponse:
    """上流/プロバイダーの例外をサニタイズされた Anthropic エラーレスポンスにマッピングする。

    サーバー側でフル（リダクション済み）の詳細をログに記録し、
    endpoint/キーの詳細が漏洩しないよう汎用メッセージを返す。
    既知の OpenAI ``APIStatusError`` コードは対応する Anthropic エラー型にマッピングされる。
    """
    # サーバー側でフル（リダクション済み）の詳細をログに記録し、汎用メッセージを返す。
    status_code = getattr(e, "status_code", None)
    logger.exception("Request failed (status=%s)", status_code)
    if isinstance(e, APIStatusError):
        code = e.status_code
        mapping = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            429: "rate_limit_error",
        }
        return _anthropic_error(
            code,
            mapping.get(code, "api_error"),
            "upstream provider returned an error",
        )
    return _anthropic_error(500, "api_error", "internal proxy error")
