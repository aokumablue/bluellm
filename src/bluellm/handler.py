"""リクエストのオーケストレーション: 変換 -> プロバイダー呼び出し -> 逆変換。

上流呼び出しは :func:`bluellm.reliability.call_with_retry` で各モデルの
``RetryPolicy`` に従いリトライする。一時障害（retryable）でリトライを使い切り、
かつ後続の fallback 候補が残っていれば次のモデルへフェイルオーバーする。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from bluellm.config import ModelConfig
from bluellm.cost import UsageLogger
from bluellm.providers.openai_like import get_provider
from bluellm.reliability import call_with_retry, is_retryable
from bluellm.translation import BlueLLMMessagesAdapter


async def process(
    body: Dict[str, Any],
    model_configs: List[ModelConfig],
    usage_logger: Optional[UsageLogger] = None,
) -> Tuple[bool, Union[Dict[str, Any], AsyncIterator[bytes]]]:
    """(is_stream, payload) を返す。

    ``model_configs`` は先頭が primary、以降が fallback 連鎖。各候補をリトライ付きで
    試行し、retryable 例外でリトライを尽くしたら次候補へフェイルオーバーする。
    retryable でない例外（400/401 等）や最終候補の失敗はそのまま送出する。

    ``usage_logger`` が与えられた場合、成功した候補のトークン使用量を記録する
    （非ストリームは即時、ストリームは最終 usage 確定時のコールバック経由）。

    payload は非ストリーム時は Anthropic Messages レスポンス dict、
    ストリーム時は SSE バイトチャンクの非同期イテレーター。
    """
    stream = bool(body.get("stream", False))
    adapter = BlueLLMMessagesAdapter()

    openai_request, tool_name_mapping = (
        adapter.translate_completion_input_params_with_tool_mapping(dict(body))
    )

    last_index = len(model_configs) - 1
    for index, model_config in enumerate(model_configs):
        provider = get_provider(model_config)
        try:
            result = await call_with_retry(
                lambda: provider.acreate(openai_request, stream=stream),
                model_config.retry,
            )
        except Exception as exc:
            # retryable かつ後続候補があれば fallback、それ以外（恒久エラー・
            # 最終候補）は送出する。
            if is_retryable(exc) and index < last_index:
                continue
            raise

        if stream:
            on_usage = None
            if usage_logger is not None:
                mc = model_config

                def on_usage(usage: Dict[str, Any]) -> None:
                    usage_logger.record(mc.deployment, mc.provider, usage)

            sse_iter = adapter.translate_completion_output_params_streaming(
                completion_stream=result,
                model=model_config.deployment,
                tool_name_mapping=tool_name_mapping,
                is_async=True,
                on_usage=on_usage,
            )
            return True, sse_iter

        anthropic_response = adapter.translate_completion_output_params(
            response=result,
            tool_name_mapping=tool_name_mapping,
        )
        if usage_logger is not None:
            usage_logger.record(
                model_config.deployment,
                model_config.provider,
                anthropic_response.get("usage") or {},
            )
        return False, anthropic_response
