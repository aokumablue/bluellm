"""OpenTelemetry 計装（observability）のテスト。

OTel 有効パスは opentelemetry が dev 依存に含まれる前提でカバーする。
ImportError パスは sys.modules を差し替えて再現する。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import bluellm.observability as obs


class _FakeSpan:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, key, value):
        self.attrs[key] = value


class _FakeTracer:
    def __init__(self):
        self.span = _FakeSpan()

    @contextmanager
    def start_as_current_span(self, name):
        yield self.span


def _settings(**over):
    base = dict(
        otel_disabled=False,
        otel_endpoint="http://127.0.0.1:4318/v1/traces",
        otel_service_name="bluellm-test",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _reset_tracer():
    obs._TRACER = None


def test_noop_span_set_is_ignored():
    # NOOP_SPAN.set は何もしない（戻り値 None）。
    assert obs.NOOP_SPAN.set("k", "v") is None


def test_request_span_noop_when_tracer_none(monkeypatch):
    monkeypatch.setattr(obs, "_TRACER", None)
    with obs.request_span("x", **{"a": 1}) as span:
        assert span is obs.NOOP_SPAN
        span.set("b", 2)  # 例外なく no-op


def test_init_tracing_disabled_returns_false():
    assert obs.init_tracing(_settings(otel_disabled=True)) is False


def test_init_tracing_returns_false_when_otel_missing(monkeypatch):
    # opentelemetry を import 不能にして no-op パスをカバーする。
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    assert obs.init_tracing(_settings()) is False


def test_init_tracing_sets_up_real_tracer():
    # OTel 有効化で TracerProvider/exporter を構築し _TRACER を設定する。
    # span は作らない（実 exporter のネットワーク export を避ける）。
    _reset_tracer()
    try:
        assert obs.init_tracing(_settings()) is True
        assert obs._TRACER is not None
    finally:
        _reset_tracer()


def test_request_span_active_records_attributes(monkeypatch):
    # tracer 有効時は _SpanRecorder を yield し、開始属性＋後付け属性を記録する。
    tracer = _FakeTracer()
    monkeypatch.setattr(obs, "_TRACER", tracer)
    with obs.request_span("v1.messages", **{"bluellm.requested_model": "claude-x"}) as span:
        assert span is not obs.NOOP_SPAN
        span.set("bluellm.input_tokens", 10)
        span.set("bluellm.skipped", None)  # None はスキップ
    assert tracer.span.attrs == {
        "bluellm.requested_model": "claude-x",
        "bluellm.input_tokens": 10,
    }


def test_span_recorder_skips_none():
    captured = {}

    class FakeSpan:
        def set_attribute(self, key, value):
            captured[key] = value

    rec = obs._SpanRecorder(FakeSpan())
    rec.set("a", 1)
    rec.set("b", None)
    assert captured == {"a": 1}
