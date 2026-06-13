from bluellm.translation._errors import UnsupportedContentError
from bluellm.translation._request_common import _RequestCommonMixin
from bluellm.translation._request_message import _RequestMessageMixin
from bluellm.translation._request_schema import _RequestSchemaMixin
from bluellm.translation._request_thinking import _RequestThinkingMixin
from bluellm.translation._request_tools import _RequestToolsMixin
from bluellm.translation._response import _ResponseMixin
from bluellm.translation._streaming_translation import _StreamingTranslationMixin

__all__ = [
    "BlueLLMMessagesAdapter",
    "UnsupportedContentError",
]


class BlueLLMMessagesAdapter(
    _RequestMessageMixin,
    _RequestToolsMixin,
    _RequestSchemaMixin,
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
