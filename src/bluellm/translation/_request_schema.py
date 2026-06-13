import copy
from typing import (
    Any,
    Dict,
    Optional,
)

from bluellm.types.anthropic_request import (
    AnthropicMessagesRequest,
)
from bluellm.types.openai import (
    ChatCompletionRequest,
)


class _RequestSchemaMixin:
    """Anthropic の構造化出力スキーマを OpenAI response_format へ変換する変換群。"""

    def translate_anthropic_output_format_to_openai(
        self, output_format: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Anthropic の output_format を OpenAI の response_format に変換する。

        Anthropic output_format: {"type": "json_schema", "schema": {...}}
        OpenAI response_format: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}

        Args:
            output_format: 'type' と 'schema' を持つ Anthropic の output_format dict

        Returns:
            OpenAI 互換の response_format dict、または無効な場合は None
        """
        if not isinstance(output_format, dict):
            return None

        output_type = output_format.get("type")
        if output_type != "json_schema":
            return None

        schema = output_format.get("schema")
        if not schema:
            return None

        # 元のスキーマを変更しないようディープコピーする
        schema = copy.deepcopy(schema)
        # OpenAI の strict モードはすべての object に additionalProperties: false を要求する
        self._add_additional_properties_false(schema)

        # OpenAI の response_format 構造に変換する
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": schema,
                "strict": True,
            },
        }

    @staticmethod
    def _add_additional_properties_false(schema: dict) -> None:
        """
        object スキーマが OpenAI strict モードに準拠するよう再帰的に確認する。

        OpenAI の strict モードの要件:
        1. すべての object のネストレベルで 'additionalProperties': false
        2. すべてのプロパティキーが 'required' に列挙されていること
        """
        if not isinstance(schema, dict):
            return

        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
            for prop in schema["properties"].values():
                _RequestSchemaMixin._add_additional_properties_false(prop)

        # array の items を処理する
        if "items" in schema:
            _RequestSchemaMixin._add_additional_properties_false(
                schema["items"]
            )

        # anyOf/oneOf/allOf を処理する
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for sub_schema in schema[key]:
                    _RequestSchemaMixin._add_additional_properties_false(
                        sub_schema
                    )

        # $defs / definitions を処理する
        for key in ("$defs", "definitions"):
            if key in schema:
                for def_schema in schema[key].values():
                    _RequestSchemaMixin._add_additional_properties_false(
                        def_schema
                    )

    def _translate_output_format_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Anthropic の構造化出力設定を OpenAI の ``response_format`` に変換する。

        レガシーのトップレベル ``output_format`` フィールドと新しい
        ``output_config.format``（``output_config`` のサブキー）の両方を受け付ける。
        これにより、新しい Anthropic 構造化出力 API を使用する呼び出し元のスキーマが
        アダプターパスでサイレントに削除されることなく、非 Anthropic バックエンドに
        ``response_format`` として渡される。両方が指定された場合は ``output_format``
        が優先される。
        """
        output_format: Any = anthropic_message_request.get("output_format")
        if not output_format:
            output_config = anthropic_message_request.get("output_config")
            if isinstance(output_config, dict):
                output_format = output_config.get("format")
        if not output_format:
            return
        response_format = self.translate_anthropic_output_format_to_openai(
            output_format=output_format
        )
        if response_format:
            new_kwargs["response_format"] = response_format
