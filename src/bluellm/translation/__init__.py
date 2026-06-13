from bluellm.translation._errors import UnsupportedContentError
from bluellm.translation._request import _RequestMixin
from bluellm.translation._request_common import _RequestCommonMixin
from bluellm.translation._request_thinking import _RequestThinkingMixin
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
    _RequestThinkingMixin,
    _RequestCommonMixin,
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
