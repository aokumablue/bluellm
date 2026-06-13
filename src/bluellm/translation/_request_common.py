from typing import (
    Any,
    Dict,
    Optional,
    cast,
)

from bluellm.translation._errors import UnsupportedContentError
from bluellm.translation._tools import (
    _ANTHROPIC_ONLY_PARAMS,
    _TRANSLATABLE_ANTHROPIC_PARAMS,
)


class _RequestCommonMixin:
    """Anthropic Messages リクエスト変換の共通ヘルパー群。"""

    @staticmethod
    def _extract_signature_from_tool_call(tool_call: Any) -> Optional[str]:
        """
        tool call の provider_specific_fields から signature を取得する。
        provider_specific_fields のみを確認し、thinking ブロックは確認しない。
        """
        signature = None

        if (
            hasattr(tool_call, "provider_specific_fields")
            and tool_call.provider_specific_fields
        ):
            if "thought_signature" in tool_call.provider_specific_fields:
                signature = tool_call.provider_specific_fields["thought_signature"]
        elif (
            hasattr(tool_call.function, "provider_specific_fields")
            and tool_call.function.provider_specific_fields
        ):
            if "thought_signature" in tool_call.function.provider_specific_fields:
                signature = tool_call.function.provider_specific_fields[
                    "thought_signature"
                ]

        return signature

    @staticmethod
    def _extract_signature_from_tool_use_content(
        content: Dict[str, Any],
    ) -> Optional[str]:
        """
        tool_use content block の provider_specific_fields から signature を取得する。
        """
        provider_specific_fields = content.get("provider_specific_fields", {})
        if provider_specific_fields:
            return provider_specific_fields.get("signature")
        return None

    def _add_cache_control_if_applicable(
        self,
        source: Any,
        target: Any,
        model: Optional[str],
    ) -> None:
        """
        source から cache_control を取得し、保持すべき場合に target に追加する。

        このメソッドは通常の dict と TypedDict オブジェクトの両方をサポートするため Any 型を受け付ける。
        TypedDict オブジェクト（ChatCompletionTextObject、ChatCompletionImageObject など）は
        実行時には dict だが、型チェック時には特定の型を持つ。Any を使用することで、
        実行時の正確性を保ちながら両方で動作できる。

        Args:
            source: cache_control フィールドを持つ可能性のある dict または TypedDict
            target: cache_control を追加する dict または TypedDict
            model: cache_control を保持すべきかチェックするモデル名
        """
        # TypedDict オブジェクトは実行時には dict のため、.get() が使用可能
        cache_control = (
            source.get("cache_control")
            if isinstance(source, dict)
            else getattr(source, "cache_control", None)
        )
        if cache_control and model and self.is_anthropic_claude_model(model):
            # TypedDict オブジェクトは実行時に dict 操作をサポートする
            # コードベースのパターン（anthropic/chat/transformation.py:432 参照）に合わせて type ignore を使用
            if isinstance(target, dict):
                target["cache_control"] = cache_control  # type: ignore[typeddict-item]
            else:  # pragma: no cover - 防御的コード。すべての呼び出し元は dict の target を渡す
                cast(Dict[str, Any], target)["cache_control"] = cache_control

    def translatable_anthropic_params(self) -> frozenset:
        """OpenAI 形式に変換が必要な Anthropic パラメーターの一覧。"""
        return _TRANSLATABLE_ANTHROPIC_PARAMS

    def anthropic_only_params(self) -> frozenset:
        """OpenAI 上流へ転送してはならない Anthropic 専用パラメーターの一覧。"""
        return _ANTHROPIC_ONLY_PARAMS

    @staticmethod
    def is_anthropic_claude_model(model: str) -> bool:
        """
        thinking パラメーターをサポートする Anthropic Claude モデルかどうかを確認する。

        True を返す条件:
        - anthropic/* モデル
        - bedrock/*anthropic* モデル（converse を含む）
        - vertex_ai/*claude* モデル
        """
        model_lower = model.lower()
        return "anthropic" in model_lower or "claude" in model_lower

    @staticmethod
    def _translate_anthropic_image_to_openai(image_source: dict) -> str:
        """
        Anthropic の image source を OpenAI 互換の image URL に変換する。

        Anthropic がサポートする image source 形式:
        1. Base64: {"type": "base64", "media_type": "image/jpeg", "data": "..."}
        2. URL: {"type": "url", "url": "https://..."}

        フォーマットされた image URL 文字列を返す。表現できない source
        （非 object の source、``file``/file_id 参照などの未知の ``type``、
        またはペイロードのない base64/url source）に対しては、
        ブロックをサイレントに削除する値を返す代わりに
        :class:`UnsupportedContentError` を発生させる。
        """
        if not isinstance(image_source, dict):
            raise UnsupportedContentError(
                f"image source must be an object, got {type(image_source).__name__}"
            )

        source_type = image_source.get("type")

        if source_type == "base64":
            # Base64 image 形式
            media_type = image_source.get("media_type", "image/jpeg")
            image_data = image_source.get("data", "")
            if image_data:
                return f"data:{media_type};base64,{image_data}"
            raise UnsupportedContentError("base64 image source has empty 'data'")
        elif source_type == "url":
            # URL 参照の image 形式
            url = image_source.get("url", "")
            if url:
                return url
            raise UnsupportedContentError("url image source has empty 'url'")

        raise UnsupportedContentError(
            f"unsupported image source type: {source_type!r}"
        )
