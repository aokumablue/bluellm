"""usage トークン記録: 1 リクエスト 1 行を日次 JSONL に追記する。

出力先は ``~/.bluellm/costs/<yyyy-mm-dd>.jsonl`` 固定（書き込み時点の日付で
ファイルを分ける）。常時有効・設定項目なし・単価/課金計算はしない。記録失敗は
握り潰してリクエスト処理を壊さない。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlparse

logger = logging.getLogger("bluellm")

# 出力先の固定ベースディレクトリ（テストは monkeypatch で差し替える）。
_DEFAULT_BASE_DIR = Path.home() / ".bluellm" / "costs"

# 記録するキャッシュトークンキー（usage に含まれる場合のみ出力する）。
_CACHE_KEYS = ("cache_read_input_tokens", "cache_creation_input_tokens")


def _utcnow() -> datetime:
    """記録時刻（UTC）を返す。テストで now を注入できるよう関数に切り出す。"""
    return datetime.now(timezone.utc)


def endpoint_host(api_base: Optional[str]) -> str:
    """api_base からホスト名のみを抽出する（秘密のURL/パス/keyは含めない）。

    api_base が None/空（ollama 既定のローカル等）の場合は "localhost" を返す。
    """
    if not api_base:
        return "localhost"
    host = urlparse(api_base).hostname
    return host or "localhost"


class UsageLogger:
    """トークン使用量を日次 JSONL（``<base_dir>/<date>.jsonl``）に追記するロガー。"""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """``base_dir`` 未指定時は固定の ``~/.bluellm/costs`` を使う。"""
        self._base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_BASE_DIR

    def record(
        self,
        model: str,
        provider: str,
        usage: Mapping[str, Any],
        *,
        endpoint: Optional[str] = None,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        """``usage`` から input/output/cache トークンを抽出して 1 行追記する。

        書き込み時点の日付でファイル名を決め、日次でファイルを分ける。
        ``endpoint`` が真値ならホスト名として entry に追加する（生 URL や key は
        出力しない）。記録失敗（ディレクトリ作成不可・書き込み失敗等）は握り潰す。
        """
        try:
            ts = now()
            entry: Dict[str, Any] = {
                "ts": ts.isoformat(),
                "model": model,
                "provider": provider,
            }
            if endpoint:
                entry["endpoint"] = endpoint
            entry["input_tokens"] = int(usage.get("input_tokens", 0) or 0)
            entry["output_tokens"] = int(usage.get("output_tokens", 0) or 0)
            for key in _CACHE_KEYS:
                value = usage.get(key)
                if value:
                    entry[key] = int(value)
            self._base_dir.mkdir(parents=True, exist_ok=True)
            path = self._base_dir / f"{ts.strftime('%Y-%m-%d')}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # 記録は付随処理。失敗してもクライアントへの応答は壊さない。
            logger.debug("usage record failed", exc_info=True)
