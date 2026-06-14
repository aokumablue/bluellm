"""OpenTelemetry 計装（任意依存・既定 ON）。

``opentelemetry`` がインストールされていれば既定で有効化し、``/v1/messages`` の
リクエストごとに span を張る。未インストールまたは明示無効（``otel_disabled``）時は
完全 no-op。ランタイム必須依存にしないため import は遅延・防御的に行う。

span 属性は ``request_span`` で生成した recorder の ``set`` で付与する。``None`` 値は
スキップする。OTel 無効時は :data:`NOOP_SPAN`（属性設定を無視）を返すので、呼び出し側は
有効/無効を気にせず ``set`` を呼べる。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

logger = logging.getLogger("bluellm")

# 有効化時に Tracer を保持（無効/未初期化なら None）。
_TRACER: Any = None


class _NoopSpan:
    """OTel 無効時のダミー span。属性設定は無視する。"""

    def set(self, key: str, value: Any) -> None:
        """属性設定を無視する（no-op）。"""
        return None


NOOP_SPAN = _NoopSpan()


class _SpanRecorder:
    """実 span をラップし、``None`` 値をスキップして属性を設定する recorder。"""

    def __init__(self, span: Any) -> None:
        """``span`` をラップする。"""
        self._span = span

    def set(self, key: str, value: Any) -> None:
        """``value`` が ``None`` でなければ span 属性に設定する。"""
        if value is not None:
            self._span.set_attribute(key, value)


def _import_otel() -> SimpleNamespace:
    """OTel の必要シンボルをまとめて import する（未インストール時 ImportError）。

    import を 1 箇所に集約し、``init_tracing`` 側で ImportError を捕捉できるようにする。
    """
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    return SimpleNamespace(
        trace=trace,
        OTLPSpanExporter=OTLPSpanExporter,
        Resource=Resource,
        TracerProvider=TracerProvider,
        BatchSpanProcessor=BatchSpanProcessor,
    )


def init_tracing(settings: Any) -> bool:
    """設定に基づき OTel を初期化する。有効化できたら ``True``。

    ``settings.otel_disabled`` が真なら無効。``opentelemetry`` 未インストール時は
    警告を出さず no-op（依存最小）。OTLP HTTP exporter を ``settings.otel_endpoint``
    へ向ける。
    """
    global _TRACER
    if getattr(settings, "otel_disabled", False):
        return False
    try:
        otel = _import_otel()
    except ImportError:
        return False
    resource = otel.Resource.create({"service.name": settings.otel_service_name})
    provider = otel.TracerProvider(resource=resource)
    # HTTP exporter は endpoint の URL スキーム（http/https）で TLS 有無を決める。
    exporter = otel.OTLPSpanExporter(endpoint=settings.otel_endpoint)
    provider.add_span_processor(otel.BatchSpanProcessor(exporter))
    otel.trace.set_tracer_provider(provider)
    _TRACER = otel.trace.get_tracer("bluellm")
    return True


@contextmanager
def request_span(name: str, **attrs: Any) -> Iterator[Any]:
    """リクエスト span を開く context manager。

    OTel 有効時は span をラップした :class:`_SpanRecorder` を、無効時は
    :data:`NOOP_SPAN` を yield する。``attrs`` は開始時に属性として設定する。
    """
    tracer = _TRACER
    if tracer is None:
        yield NOOP_SPAN
        return
    with tracer.start_as_current_span(name) as span:
        recorder = _SpanRecorder(span)
        for key, value in attrs.items():
            recorder.set(key, value)
        yield recorder
