"""ログ用のシークレット秘匿処理。

既知のシークレット値（API key、マスター/ソルト key、受信認証ヘッダーの値）を
ここに登録する。logging フィルタがログレコード内のすべての出現箇所をマスクし、
認証情報がログに漏洩しないようにする。
"""

from __future__ import annotations

import logging
from typing import Set

_SECRETS: Set[str] = set()
_PLACEHOLDER = "***REDACTED***"


def register_secret(value: object) -> None:
    """すべてのログ出力から秘匿する値を登録する。

    空でない文字列はすべて登録される。短い値は無関係なログテキストを
    過剰にマスクする可能性があるが、認証情報については漏洩よりも
    過剰マスクを意図的に優先する。
    """
    if isinstance(value, str) and value:
        _SECRETS.add(value)


def redact(text: str) -> str:
    """``text`` 中に出現する登録済みシークレットをすべてプレースホルダーに置換する。"""
    for secret in _SECRETS:
        if secret and secret in text:
            text = text.replace(secret, _PLACEHOLDER)
    return text


class RedactionFilter(logging.Filter):
    """登録済みシークレットをメッセージとトレースバックから除去する logging フィルタ。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """レコードのレンダリング済みメッセージと例外テキストをインプレースで秘匿する。"""
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        # 例外トレースバックは上記のメッセージ処理をバイパスする（フォーマッタが
        # exc_info から個別にレンダリングするため）。そのため、ここで処理する：
        # トレースバックをレンダリングして秘匿し、フォーマッタが再利用できるよう
        # exc_text に格納し、ハンドラが生の情報を再レンダリングしないよう
        # exc_info をクリアする。
        if record.exc_info:
            import traceback

            text = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = redact(text)
            record.exc_info = None
        elif record.exc_text:
            # 上流のハンドラ/フォーマッタがすでに exc_text をレンダリングしている場合、
            # 再 emit 時に漏洩しないよう、そのキャッシュ済みコピーも秘匿する。
            record.exc_text = redact(record.exc_text)
        return True


def install() -> None:
    """redaction フィルタをルートロガーおよび uvicorn ロガーに追加する。"""
    filt = RedactionFilter()
    for name in ("", "bluellm", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).addFilter(filt)
