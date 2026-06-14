"""モデルルーティング: 受信したモデル名を ModelConfig に解決する。

``model_name`` の完全一致が優先され、一致しない場合は ``model_name: "*"`` の
キャッチオールが使用される。Claude Code は ``claude-*`` 形式のモデル名を送信するため、
キャッチオールによって設定済みの Azure/OpenAI デプロイメントにルーティングされる。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from bluellm.config import Config, ModelConfig


class Router:
    """受信したモデル名を :class:`ModelConfig` 群に解決する（完全一致、次に ``*``）。

    同一 ``model_name`` の複数エントリ（複数エンドポイント）はリストに集約され、
    ``resolve`` はそのモデルの全エンドポイントを返す。
    """

    def __init__(self, config: Config) -> None:
        """設定のモデルリストを名前ごとにグループ化し、``*`` キャッチオール群を保持する。"""
        self._by_name: Dict[str, List[ModelConfig]] = {}
        self._wildcard: Optional[List[ModelConfig]] = None
        for mc in config.model_list:
            if mc.model_name == "*":
                self._wildcard = (self._wildcard or [])
                self._wildcard.append(mc)
            else:
                self._by_name.setdefault(mc.model_name, []).append(mc)

    def resolve(self, model_name: str) -> List[ModelConfig]:
        """``model_name`` に対応する ModelConfig 群を返す（一致すれば必ず 1 件以上）。

        完全一致グループを優先し、なければ ``*`` キャッチオール群、それも無ければ
        KeyError を送出する。
        """
        group = self._by_name.get(model_name) or self._wildcard
        if not group:
            raise KeyError(
                f"No model_list entry matches '{model_name}' and no '*' catch-all is configured"
            )
        return group
