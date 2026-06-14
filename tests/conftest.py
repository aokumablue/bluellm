"""Shared pytest configuration for the bluellm test suite.

Its presence at the tests root fixes the rootdir so pytest prepends this
directory to ``sys.path``; combined with ``pythonpath = ["tests"]`` in
``pyproject.toml`` it makes ``from helpers import ...`` resolve robustly,
including when tests are invoked from outside the repository root. Shared
fixtures, when added, belong here.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_usage_log(tmp_path_factory, monkeypatch):
    """usage 記録のベースディレクトリを毎テスト tmp に向け、実 HOME を汚さない。

    UsageLogger() は base_dir 未指定時に ``bluellm.cost._DEFAULT_BASE_DIR`` を
    参照するため、ここを差し替えるとサーバ経由の記録も tmp に隔離される。
    """
    monkeypatch.setattr(
        "bluellm.cost._DEFAULT_BASE_DIR", tmp_path_factory.mktemp("usagelog")
    )


@pytest.fixture(autouse=True)
def _reset_otel_tracer():
    """各テスト後に OTel のモジュールグローバル tracer を初期化し漏れを防ぐ。

    CLI の serve テストは init_tracing を実呼び出しするため、これがないと
    後続テストの request_span が実 tracer を使い、副作用（export 失敗等）を招く。
    """
    import bluellm.observability as obs

    yield
    obs._TRACER = None
