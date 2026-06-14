"""エンドポイント負荷分散: ラウンドロビン＋エンドポイント単位サーキットブレーカ。

同一モデル名に複数エンドポイント（``(model_name, api_base)`` で識別）が設定された
場合に、select でラウンドロビン順を返しつつ、連続失敗が閾値に達したエンドポイントを
TTL の間だけ候補から除外する。全エンドポイントが除外された場合は fail-open し、
グループ全体を候補として返す（可用性を優先し、ブラックホールを避ける）。
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

from bluellm.config import ModelConfig
from bluellm.router import Router


class EndpointBalancer:
    """ラウンドロビンとサーキットブレーカでエンドポイント選択を担うバランサ。

    エンドポイントは ``(model_name, api_base)`` のタプルで識別する。連続失敗数が
    ``threshold`` に達したエンドポイントは ``ttl_seconds`` の間 select の候補から
    除外される。TTL 経過後は再び候補に戻るが、:meth:`report_success` を受けない限り
    失敗カウンタは保持される（=半開状態。回復後に再失敗すると即再除外される）。
    """

    def __init__(
        self,
        router: Router,
        threshold: int,
        ttl_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """ルーター・閾値・TTL・単調クロックを保持し、内部 state を初期化する。"""
        self._router = router
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._monotonic = monotonic
        self._rr: Dict[str, int] = {}
        self._failures: Dict[Tuple[str, Optional[str]], int] = {}
        self._excluded_until: Dict[Tuple[str, Optional[str]], float] = {}

    def select(self, name: str) -> List[ModelConfig]:
        """``name`` のエンドポイント群を健全性とラウンドロビン順を考慮して返す。

        現在除外中（``_excluded_until`` が現在時刻より未来）のエンドポイントを除いた
        healthy 群を候補とする。全てが除外されている場合は fail-open し、グループ
        全体を候補とする（空を返さない）。候補に対しモデル名ごとのカウンタで開始位置を
        回転させて並べ替えた ModelConfig のリストを返し、カウンタを 1 進める。
        未知の ``name`` は :class:`Router` の KeyError をそのまま伝播する。
        """
        group = self._router.resolve(name)
        now = self._monotonic()
        healthy = [
            mc
            for mc in group
            if self._excluded_until.get((mc.model_name, mc.api_base), 0) <= now
        ]
        candidates = healthy if healthy else group
        start = self._rr.get(name, 0) % len(candidates)
        ordered = candidates[start:] + candidates[:start]
        self._rr[name] = self._rr.get(name, 0) + 1
        return ordered

    def report_failure(self, mc: ModelConfig) -> None:
        """``mc`` の連続失敗を 1 件記録し、閾値到達でエンドポイントを除外する。

        失敗カウンタが ``threshold`` 以上になると ``ttl_seconds`` 後までエンドポイントを
        除外する。TTL 経過後は再び healthy 扱いになるが、:meth:`report_success` を
        受けない限り失敗カウンタは保持される（=半開状態。回復後に再失敗すると即再除外
        される）。
        """
        key = (mc.model_name, mc.api_base)
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= self._threshold:
            self._excluded_until[key] = self._monotonic() + self._ttl_seconds

    def report_success(self, mc: ModelConfig) -> None:
        """``mc`` の失敗カウンタと除外状態をリセットし、健全状態に戻す。"""
        key = (mc.model_name, mc.api_base)
        self._failures.pop(key, None)
        self._excluded_until.pop(key, None)
