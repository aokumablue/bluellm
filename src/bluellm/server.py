"""Claude Code が使用する Anthropic Messages API サーフェスを公開する FastAPI アプリ。"""

from __future__ import annotations

import json
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIStatusError

from bluellm import handler
from bluellm.auth import Authenticator, require_auth
from bluellm.config import Config
from bluellm.router import Router
from bluellm.translation import UnsupportedContentError

logger = logging.getLogger("bluellm")


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
            model_config = request.app.state.router.resolve(body.get("model", ""))
        except KeyError as e:
            return _anthropic_error(404, "not_found_error", str(e))

        try:
            is_stream, payload = await handler.process(body, model_config)
        except UnsupportedContentError as e:
            # クライアントがプロキシで変換できないコンテンツブロックを送信した場合
            # （例: サポートされていない画像ソース）は、暗黙的に破棄せず拒否する。
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
