"""Provider抽象化（拡張ポイント）。

Providerは OpenAI Chat Completions 形式のリクエスト dict を受け取り、
上流 API を呼び出す。戻り値は openai SDK の ``ChatCompletion`` /
``ChatCompletionChunk`` と属性互換なオブジェクトであること（移植済み
translation layer が消費する）。新しいバックエンドを追加するには、この
インターフェースを実装し ``provider/`` プレフィックスで登録する。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Union


class BaseProvider:
    """上流バックエンドの interface。OpenAI 形式のリクエストを処理する。"""

    async def acreate(
        self, request: Dict[str, Any], *, stream: bool
    ) -> Union[Any, AsyncIterator[Any]]:
        """上流 API を呼び出し、レスポンスオブジェクトまたは非同期チャンクイテレータを返す。"""
        raise NotImplementedError
