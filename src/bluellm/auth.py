"""マスターキー認証。

マスターキーは SHA-256 ハッシュとしてのみ保存される。受信した認証情報は
ハッシュ化され、定数時間で比較される。マスターキーが設定されていない場合、
proxy はオープン（ローカル開発モード）で動作する。
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional

from fastapi import HTTPException, Request, status


def hash_token(token: str) -> str:
    """``token`` の SHA-256 ダイジェストを16進数文字列で返す。"""
    return hashlib.sha256(token.encode()).hexdigest()


class Authenticator:
    """定数時間でのマスターキー検証。マスターキーが設定されていない場合は無効。"""

    def __init__(self, master_key: Optional[str]) -> None:
        """``master_key`` の SHA-256 ハッシュのみを保存する（None の場合は認証無効）。"""
        self._master_hash: Optional[str] = hash_token(master_key) if master_key else None

    @property
    def enabled(self) -> bool:
        """マスターキーが設定されている場合（つまり認証が強制される場合）に True。"""
        return self._master_hash is not None

    def verify(self, presented: Optional[str]) -> None:
        """``presented`` がマスターキーと一致しない場合は 401 を発生させる（定数時間）。"""
        if self._master_hash is None:
            return
        if not presented or not hmac.compare_digest(
            hash_token(presented), self._master_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid x-api-key",
            )


def _extract_key(request: Request) -> Optional[str]:
    """``x-api-key`` ヘッダーまたは Bearer 認証ヘッダーからキーを取り出す。"""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    return None


async def require_auth(request: Request) -> None:
    """リクエストに対してマスターキー認証を強制する FastAPI dependency。"""
    authenticator: Authenticator = request.app.state.authenticator
    authenticator.verify(_extract_key(request))
