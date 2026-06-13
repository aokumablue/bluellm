from bluellm.translation import _request as _request_module
from bluellm.translation._errors import UnsupportedContentError
from bluellm.translation._request import _RequestMixin
from bluellm.translation._response import _ResponseMixin
from bluellm.translation._streaming_translation import _StreamingTranslationMixin
from bluellm.translation._tools import truncate_tool_name

__all__ = [
    "BlueLLMMessagesAdapter",
    "UnsupportedContentError",
    "truncate_tool_name",
]


class BlueLLMMessagesAdapter(
    _RequestMixin,
    _ResponseMixin,
    _StreamingTranslationMixin,
):
    """Anthropic Messages 形式と OpenAI Chat Completions 形式（入力パラメーター、
    出力コンテンツ、ストリーミングチャンク）の間のコアトランスレーター。

    リクエスト間でステートレスかつ再利用可能。
    """

    def __init__(self):
        """インスタンス状態なし。アダプターはステートレス。"""
        pass


# _RequestMixin.build_reasoning_effort_param / _add_additional_properties_false は
# 本体内で ``BlueLLMMessagesAdapter`` を直接参照する（純粋移動のため本体を変更しない）。
# 合成後にここで _request モジュールの名前空間へ解決済みクラスを注入し、
# 実行時のグローバル参照を満たす（import 時は循環しない）。
_request_module.BlueLLMMessagesAdapter = BlueLLMMessagesAdapter
