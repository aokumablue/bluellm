from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from bluellm._compat import parse_tool_call_arguments
from bluellm._compat import THOUGHT_SIGNATURE_SEPARATOR
from bluellm._compat import PolyfillResult
from bluellm.types.anthropic import (
    AnthropicFinishReason,
    AnthropicResponseContentBlockRedactedThinking,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
    ContextManagementResponse,
    UsageIteration,
)
from bluellm.types.anthropic import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from bluellm._compat import Choices, ModelResponse, Usage


class _ResponseMixin:
    """OpenAI の非ストリーミングレスポンスを Anthropic 形式へ変換する変換群。"""

    def translate_completion_output_params(
        self,
        response: ModelResponse,
        tool_name_mapping: Optional[Dict[str, str]] = None,
        polyfill_result: Optional[PolyfillResult] = None,
    ) -> Optional[AnthropicMessagesResponse]:
        """
        OpenAI レスポンスを Anthropic 形式に変換する。

        Args:
            response: OpenAI の ModelResponse
            tool_name_mapping: 切り詰めた tool 名を元の名前にマッピングするオプションの辞書。
                              OpenAI の64文字制限を超えた tool の名前を復元するために使用する。
            polyfill_result: context_management ポリフィルの PolyfillResult。
        """
        return self.translate_openai_response_to_anthropic(
            response=response,
            tool_name_mapping=tool_name_mapping,
            polyfill_result=polyfill_result,
        )

    def _translate_openai_content_to_anthropic(
        self,
        choices: List[Choices],
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """OpenAI の choices を Anthropic の content block に変換する。

        thinking/redacted_thinking、text、tool_use ブロックを出力する
        （``tool_name_mapping`` から元の tool 名を復元する）。
        """
        new_content: List[Dict[str, Any]] = []
        for choice in choices:
            # まず thinking ブロックを処理する
            if (
                hasattr(choice.message, "thinking_blocks")
                and choice.message.thinking_blocks
            ):
                for thinking_block in choice.message.thinking_blocks:
                    self._append_anthropic_thinking_block(thinking_block, new_content)
            # thinking_blocks がない場合は reasoning_content を処理する
            elif (
                hasattr(choice.message, "reasoning_content")
                and choice.message.reasoning_content
            ):
                new_content.append(
                    AnthropicResponseContentBlockThinking(
                        type="thinking",
                        thinking=str(choice.message.reasoning_content),
                        signature=None,
                    ).model_dump()
                )

            # テキストコンテンツを処理する
            if choice.message.content is not None:
                new_content.append(
                    AnthropicResponseContentBlockText(
                        type="text", text=choice.message.content
                    ).model_dump()
                )
            # tool 呼び出しを処理する（テキストコンテンツと並行して）
            if (
                choice.message.tool_calls is not None
                and len(choice.message.tool_calls) > 0
            ):
                for tool_call in choice.message.tool_calls:
                    new_content.append(
                        self._build_anthropic_tool_use_block(
                            tool_call, tool_name_mapping
                        )
                    )

        return new_content

    @staticmethod
    def _append_anthropic_thinking_block(
        thinking_block: Dict[str, Any],
        new_content: List[Dict[str, Any]],
    ) -> None:
        """OpenAI の thinking_block を Anthropic の content block へ変換し追加する。

        ``type`` が thinking なら thinking ブロック、redacted_thinking なら
        redacted_thinking ブロックを ``new_content`` に追加する。
        """
        if thinking_block.get("type") == "thinking":
            thinking_value = thinking_block.get("thinking", "")
            signature_value = thinking_block.get("signature", "")
            new_content.append(
                AnthropicResponseContentBlockThinking(
                    type="thinking",
                    thinking=(
                        str(thinking_value) if thinking_value is not None else ""
                    ),
                    signature=(
                        str(signature_value)
                        if signature_value is not None
                        else None
                    ),
                ).model_dump()
            )
        elif thinking_block.get("type") == "redacted_thinking":
            data_value = thinking_block.get("data", "")
            new_content.append(
                AnthropicResponseContentBlockRedactedThinking(
                    type="redacted_thinking",
                    data=str(data_value) if data_value is not None else "",
                ).model_dump()
            )

    def _build_anthropic_tool_use_block(
        self,
        tool_call: Any,
        tool_name_mapping: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """OpenAI の tool_call を Anthropic の tool_use content block へ変換する。

        provider_specific_fields から signature を取得し、切り詰められた tool 名を
        ``tool_name_mapping`` で元の名前に復元し、Gemini の thought-signature
        サフィックスを id から除去したうえで model_dump() した dict を返す。
        """
        # provider_specific_fields のみから signature を取得する
        signature = self._extract_signature_from_tool_call(tool_call)

        provider_specific_fields = {}
        if signature:
            provider_specific_fields["signature"] = signature

        # 切り詰められた tool 名を元の名前に復元する
        truncated_name = tool_call.function.name or ""
        original_name = (
            tool_name_mapping.get(truncated_name, truncated_name)
            if tool_name_mapping
            else truncated_name
        )

        # Gemini の thought-signature サフィックスを id から除去する（ストリーミング
        # パスと同様）。base64 文字（+ / =）は Anthropic の
        # `^[a-zA-Z0-9_-]+$` tool_use.id パターンに違反し、リプレイ時に問題が生じる。
        raw_id = tool_call.id or ""
        base_id = (
            raw_id.split(THOUGHT_SIGNATURE_SEPARATOR, 1)[0]
            if THOUGHT_SIGNATURE_SEPARATOR in raw_id
            else raw_id
        )
        tool_use_block = AnthropicResponseContentBlockToolUse(
            type="tool_use",
            id=base_id,
            name=original_name,
            input=parse_tool_call_arguments(
                tool_call.function.arguments,
                tool_name=original_name,
                context="Anthropic pass-through adapter",
            ),
        )
        # signature が存在する場合は provider_specific_fields を追加する
        if provider_specific_fields:
            tool_use_block.provider_specific_fields = provider_specific_fields
        return tool_use_block.model_dump()

    @staticmethod
    def _build_usage_dict(usage: Usage) -> Dict[str, int]:
        """OpenAI の ``Usage`` を Anthropic 形式の usage dict に変換する。

        ``input_tokens`` はキャッシュ済みtoken数を差し引いた値（``output_tokens`` と
        ともに常に設定される）。``cache_creation_input_tokens`` / ``cache_read_input_tokens``
        は対応する値が正の場合のみ設定する。``AnthropicUsage`` と ``UsageDelta`` の双方が
        実行時 dict であり同一キー集合を持つため、両者で共用する。
        """
        uncached_input_tokens = usage.prompt_tokens or 0
        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = (
                getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            )
            uncached_input_tokens -= cached_tokens

        result: Dict[str, int] = {
            "input_tokens": uncached_input_tokens,
            "output_tokens": usage.completion_tokens or 0,
        }
        if (
            hasattr(usage, "_cache_creation_input_tokens")
            and usage._cache_creation_input_tokens > 0
        ):
            result["cache_creation_input_tokens"] = usage._cache_creation_input_tokens
        if cached_tokens > 0:
            result["cache_read_input_tokens"] = cached_tokens
        return result

    @staticmethod
    def _translate_openai_finish_reason_to_anthropic(
        openai_finish_reason: str,
    ) -> AnthropicFinishReason:
        """OpenAI の ``finish_reason`` を Anthropic の ``stop_reason`` にマッピングする。"""
        if openai_finish_reason == "stop":
            return "end_turn"
        elif openai_finish_reason == "length":
            return "max_tokens"
        elif openai_finish_reason == "tool_calls":
            return "tool_use"
        elif openai_finish_reason == "content_filter":
            # Azure OpenAI は content_filter で停止する。Claude Code がフィルターされた停止を
            # 通常の end_turn 完了として読み取らないよう、Anthropic の `refusal` stop_reason
            # として表面化する。
            return "refusal"
        return "end_turn"

    def translate_openai_response_to_anthropic(
        self,
        response: ModelResponse,
        tool_name_mapping: Optional[Dict[str, str]] = None,
        polyfill_result: Optional[PolyfillResult] = None,
    ) -> AnthropicMessagesResponse:
        """
        OpenAI レスポンスを Anthropic 形式に変換する。

        Args:
            response: OpenAI の ModelResponse
            tool_name_mapping: 切り詰めた tool 名を元の名前にマッピングするオプションの辞書。
                              OpenAI の64文字制限を超えた tool の名前を復元するために使用する。
            polyfill_result: context_management ポリフィルの PolyfillResult。
        """
        ## content block を変換する
        anthropic_content = self._translate_openai_content_to_anthropic(
            choices=response.choices,  # type: ignore
            tool_name_mapping=tool_name_mapping,
        )

        if polyfill_result is not None and polyfill_result.compaction_block is not None:
            anthropic_content.insert(0, polyfill_result.compaction_block)  # type: ignore[arg-type]

        if not anthropic_content:
            # M11: Anthropic レスポンスは少なくとも1つの content block を持つ必要がある。
            # アップストリームがコンテンツも tool 呼び出しも返さない場合（例: content_filter
            # による拒否）、Claude Code が誤って処理する空配列の代わりに空のテキストブロックを出力する。
            anthropic_content.append(
                AnthropicResponseContentBlockText(type="text", text="").model_dump()
            )

        ## finish reason を取得する
        anthropic_finish_reason = self._translate_openai_finish_reason_to_anthropic(
            openai_finish_reason=response.choices[0].finish_reason  # type: ignore
        )
        # usage を取得する
        usage: Usage = getattr(response, "usage")
        anthropic_usage: AnthropicUsage = self._build_usage_dict(usage)  # type: ignore[assignment]

        if polyfill_result is not None and polyfill_result.iterations_usage is not None:
            message_iteration: UsageIteration = {
                "type": "message",
                "input_tokens": anthropic_usage["input_tokens"],
                "output_tokens": anthropic_usage["output_tokens"],
            }
            anthropic_usage["iterations"] = list(polyfill_result.iterations_usage) + [message_iteration]  # type: ignore[typeddict-unknown-key]

        translated_obj = AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,  # type: ignore
            content=anthropic_content,  # type: ignore
            stop_reason=anthropic_finish_reason,
        )

        applied_edits = (
            polyfill_result.applied_edits_for_response() if polyfill_result else None
        )
        if applied_edits:
            translated_obj["context_management"] = ContextManagementResponse(
                applied_edits=list(applied_edits)
            )

        return translated_obj
