"""EndpointBalancer のテスト: ラウンドロビン順とサーキットブレーカ挙動を検証する。"""

from __future__ import annotations

from typing import Dict, List

import pytest

from bluellm.balancer import EndpointBalancer
from bluellm.config import Config, GeneralSettings, ModelConfig
from bluellm.router import Router


def _make_balancer(
    endpoints: List[ModelConfig],
    *,
    threshold: int = 2,
    ttl_seconds: float = 30.0,
    clock: Dict[str, float],
) -> EndpointBalancer:
    """与えたエンドポイント群から実物の Router を構築し、可変クロックで包んだ balancer を返す。"""
    config = Config(model_list=endpoints, general_settings=GeneralSettings())
    router = Router(config)
    return EndpointBalancer(
        router,
        threshold=threshold,
        ttl_seconds=ttl_seconds,
        monotonic=lambda: clock["t"],
    )


def _ep(api_base: str | None) -> ModelConfig:
    """同一モデル名・異なる api_base のエンドポイント ModelConfig を生成する。"""
    return ModelConfig(
        model_name="m", provider="openai", deployment="d", api_base=api_base
    )


def _bases(configs: List[ModelConfig]) -> List[str | None]:
    """ModelConfig リストの api_base 列を返す（順序確認用）。"""
    return [mc.api_base for mc in configs]


def test_round_robin_order() -> None:
    """3 エンドポイントで select を繰り返すと開始位置が 0,1,2,0,... と進む。"""
    clock = {"t": 0.0}
    eps = [_ep("http://a"), _ep("http://b"), _ep("http://c")]
    bal = _make_balancer(eps, clock=clock)
    assert _bases(bal.select("m")) == ["http://a", "http://b", "http://c"]
    assert _bases(bal.select("m")) == ["http://b", "http://c", "http://a"]
    assert _bases(bal.select("m")) == ["http://c", "http://a", "http://b"]
    assert _bases(bal.select("m")) == ["http://a", "http://b", "http://c"]


def test_counter_advances() -> None:
    """同一 name の連続 select でラウンドロビンカウンタが前進する。"""
    clock = {"t": 0.0}
    bal = _make_balancer([_ep("http://a"), _ep("http://b")], clock=clock)
    assert bal._rr.get("m", 0) == 0
    bal.select("m")
    assert bal._rr["m"] == 1
    bal.select("m")
    assert bal._rr["m"] == 2


def test_threshold_trips_excludes_endpoint() -> None:
    """report_failure を threshold 回呼ぶと当該 ep が次の select で除外される。"""
    clock = {"t": 0.0}
    eps = [_ep("http://a"), _ep("http://b")]
    bal = _make_balancer(eps, threshold=2, clock=clock)
    bal.report_failure(eps[0])
    bal.report_failure(eps[0])
    assert _bases(bal.select("m")) == ["http://b"]


def test_ttl_recovery() -> None:
    """TTL 経過後に除外 ep が再び候補に戻る。"""
    clock = {"t": 0.0}
    eps = [_ep("http://a"), _ep("http://b")]
    bal = _make_balancer(eps, threshold=2, ttl_seconds=30.0, clock=clock)
    bal.report_failure(eps[0])
    bal.report_failure(eps[0])
    assert _bases(bal.select("m")) == ["http://b"]
    clock["t"] = 31.0
    assert sorted(_bases(bal.select("m"))) == ["http://a", "http://b"]


def test_half_open_re_exclusion() -> None:
    """回復後に再度 report_failure 1 回で即再除外される（カウンタ保持の確認）。"""
    clock = {"t": 0.0}
    eps = [_ep("http://a"), _ep("http://b")]
    bal = _make_balancer(eps, threshold=2, ttl_seconds=30.0, clock=clock)
    bal.report_failure(eps[0])
    bal.report_failure(eps[0])
    clock["t"] = 31.0
    # TTL 経過で a は再び候補に戻る。
    assert sorted(_bases(bal.select("m"))) == ["http://a", "http://b"]
    # report_success を受けていないのでカウンタは保持。1 回の失敗で即再除外。
    bal.report_failure(eps[0])
    assert _bases(bal.select("m")) == ["http://b"]


def test_fail_open_when_all_excluded() -> None:
    """全 ep を除外しても select が group 全体を返す（空を返さない）。"""
    clock = {"t": 0.0}
    eps = [_ep("http://a"), _ep("http://b")]
    bal = _make_balancer(eps, threshold=2, clock=clock)
    for ep in eps:
        bal.report_failure(ep)
        bal.report_failure(ep)
    assert sorted(_bases(bal.select("m"))) == ["http://a", "http://b"]


def test_report_success_resets() -> None:
    """report_success が失敗カウンタ・除外をリセットし、次 select で復帰させる。"""
    clock = {"t": 0.0}
    eps = [_ep("http://a"), _ep("http://b")]
    bal = _make_balancer(eps, threshold=2, clock=clock)
    bal.report_failure(eps[0])
    bal.report_failure(eps[0])
    assert _bases(bal.select("m")) == ["http://b"]
    bal.report_success(eps[0])
    assert ("m", "http://a") not in bal._failures
    assert ("m", "http://a") not in bal._excluded_until
    assert sorted(_bases(bal.select("m"))) == ["http://a", "http://b"]


def test_unknown_name_raises_keyerror() -> None:
    """未知 name の select が Router の KeyError を伝播する。"""
    clock = {"t": 0.0}
    bal = _make_balancer([_ep("http://a")], clock=clock)
    with pytest.raises(KeyError):
        bal.select("unknown")


def test_none_api_base_endpoint() -> None:
    """api_base が None のエンドポイントでもキー動作（失敗記録・除外）が成立する。"""
    clock = {"t": 0.0}
    eps = [_ep(None), _ep("http://b")]
    bal = _make_balancer(eps, threshold=2, clock=clock)
    bal.report_failure(eps[0])
    bal.report_failure(eps[0])
    assert ("m", None) in bal._excluded_until
    assert _bases(bal.select("m")) == ["http://b"]
