"""モデルルーティング: 受信したモデル名を ModelConfig に解決する。

``model_name`` の完全一致が優先され、一致しない場合は ``model_name: "*"`` の
キャッチオールが使用される。Claude Code は ``claude-*`` 形式のモデル名を送信するため、
キャッチオールによって設定済みの Azure/OpenAI デプロイメントにルーティングされる。
"""

from __future__ import annotations

from typing import Dict, Optional

from bluellm.config import Config, ModelConfig


class Router:
    """受信したモデル名を :class:`ModelConfig` に解決する（完全一致、次に ``*``）。"""

    def __init__(self, config: Config) -> None:
        """設定のモデルリストを名前でインデックスし、``*`` キャッチオールを保持する。"""
        self._by_name: Dict[str, ModelConfig] = {}
        self._wildcard: Optional[ModelConfig] = None
        for mc in config.model_list:
            if mc.model_name == "*":
                self._wildcard = mc
            else:
                self._by_name[mc.model_name] = mc

    def resolve(self, model_name: str) -> ModelConfig:
        """``model_name`` に対応する ModelConfig を返す。一致がなければ KeyError を送出する。"""
        mc = self._by_name.get(model_name) or self._wildcard
        if mc is None:
            raise KeyError(
                f"No model_list entry matches '{model_name}' and no '*' catch-all is configured"
            )
        return mc
