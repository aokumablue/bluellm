"""リクエストのオーケストレーション: 変換 -> プロバイダー呼び出し -> 逆変換。"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Tuple, Union

from bluellm.config import ModelConfig
from bluellm.providers.openai_like import get_provider
from bluellm.translation import BlueLLMAdapter


async def process(
    body: Dict[str, Any], model_config: ModelConfig
) -> Tuple[bool, Union[Dict[str, Any], AsyncIterator[bytes]]]:
    """(is_stream, payload) を返す。

    payload は非ストリーム時は Anthropic Messages レスポンス dict、
    ストリーム時は SSE バイトチャンクの非同期イテレーター。
    """
    stream = bool(body.get("stream", False))
    adapter = BlueLLMAdapter()

    openai_request, tool_name_mapping = (
        adapter.translate_completion_input_params_with_tool_mapping(dict(body))
    )

    provider = get_provider(model_config)

    if stream:
        completion_stream = await provider.acreate(openai_request, stream=True)
        sse_iter = adapter.translate_completion_output_params_streaming(
            completion_stream=completion_stream,
            model=model_config.deployment,
            tool_name_mapping=tool_name_mapping,
            is_async=True,
        )
        return True, sse_iter

    response = await provider.acreate(openai_request, stream=False)
    anthropic_response = adapter.translate_completion_output_params(
        response=response,
        tool_name_mapping=tool_name_mapping,
    )
    return False, anthropic_response
