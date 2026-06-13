from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

from bluellm._compat import THOUGHT_SIGNATURE_SEPARATOR
from bluellm._compat import PolyfillResult
from bluellm.types.anthropic_context import (
    AppliedEdit,
    ContextManagementResponse,
)
from bluellm.types.anthropic_streaming import (
    ContentBlockDelta,
    ContentJsonBlockDelta,
    ContentTextBlockDelta,
    ContentThinkingBlockDelta,
    ContentThinkingSignatureBlockDelta,
    MessageBlockDelta,
    MessageDelta,
    UsageDelta,
)
from bluellm.types.openai import (
    ChatCompletionThinkingBlock,
)
from bluellm._compat import ModelResponse, StreamingChoices, Usage

if TYPE_CHECKING:
    from bluellm.types.anthropic_streaming import ContentBlockContentBlockDict


class _StreamingTranslationMixin:
    """OpenAI のストリーミングレスポンスを Anthropic 形式へ変換する変換群。"""

    def translate_completion_output_params_streaming(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
        polyfill_result: Optional[PolyfillResult] = None,
        is_async: bool = True,
    ) -> Union[AsyncIterator[bytes], Iterator[bytes], None]:
        """
        OpenAI ストリーミングレスポンスを Anthropic 形式に変換する。

        Args:
            completion_stream: OpenAI のストリーミングレスポンス
            model: モデル名
            tool_name_mapping: 切り詰めた tool 名を元の名前にマッピングするオプションの辞書。
            polyfill_result: context_management ポリフィルの PolyfillResult。
            is_async: ``True``（デフォルト、既存の async 呼び出し元との後方互換性のため）の場合
                ``AsyncIterator[bytes]`` を返す。``False`` の場合、イベントループなしでは
                反復処理できない async イテレーターを受け取らないよう、sync の
                ``Iterator[bytes]`` を返す（sync ハンドラー向け）。
        """
        from bluellm.streaming import BlueLLMStreamWrapper

        applied_edits = (
            polyfill_result.applied_edits_for_response() if polyfill_result else None
        )
        compaction_block = (
            polyfill_result.compaction_block if polyfill_result is not None else None
        )
        iterations_usage = (
            polyfill_result.iterations_usage if polyfill_result is not None else None
        )
        anthropic_wrapper = BlueLLMStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            tool_name_mapping=tool_name_mapping,
            applied_edits=applied_edits,
            compaction_block=compaction_block,
            iterations_usage=iterations_usage,
        )
        # 適切なイベントフォーマットのため SSE ラップバージョンを返す。
        if is_async:
            return anthropic_wrapper.async_anthropic_sse_wrapper()
        return anthropic_wrapper.anthropic_sse_wrapper()

    @staticmethod
    def _translate_streaming_openai_chunk_to_anthropic_content_block(
        choices: List[Union[OpenAIStreamingChoice, StreamingChoices]],
    ) -> Tuple[
        Literal["text", "tool_use", "thinking"],
        "ContentBlockContentBlockDict",
    ]:
        """チャンクから Anthropic の content block タイプとその開始ブロックを推定する。

        このストリーミングデルタに新しい Anthropic content block を開く必要があるかどうかを
        判断するために使用する ``(block_type, content_block_start)`` を返す。
        """
        from bluellm._compat import uuid
        from bluellm.types.anthropic_streaming import TextBlock

        for choice in choices:
            if (
                choice.delta.tool_calls is not None
                and len(choice.delta.tool_calls) > 0
                and choice.delta.tool_calls[0].function is not None
            ):
                raw_id = choice.delta.tool_calls[0].id or str(uuid.uuid4())
                tool_name = choice.delta.tool_calls[0].function.name or ""
                base_id = raw_id
                thought_sig: Optional[str] = None
                if THOUGHT_SIGNATURE_SEPARATOR in raw_id:
                    parts = raw_id.split(THOUGHT_SIGNATURE_SEPARATOR, 1)
                    base_id = parts[0]
                    thought_sig = parts[1] if len(parts) > 1 else None
                tool_block: Dict[str, Any] = {
                    "type": "tool_use",
                    "id": base_id,
                    "name": tool_name,
                    "input": {},
                }
                if thought_sig:
                    tool_block["provider_specific_fields"] = {
                        "signature": thought_sig,
                    }
                return "tool_use", cast("ContentBlockContentBlockDict", tool_block)
            elif choice.delta.content is not None and len(choice.delta.content) > 0:
                return "text", TextBlock(type="text", text="")
            elif isinstance(choice, StreamingChoices) and hasattr(
                choice.delta, "thinking_blocks"
            ):
                thinking_blocks = choice.delta.thinking_blocks or []
                if len(thinking_blocks) > 0:
                    thinking_block = thinking_blocks[0]
                    if thinking_block["type"] == "thinking":
                        thinking = thinking_block.get("thinking") or ""
                        signature = thinking_block.get("signature") or ""

                        if not isinstance(thinking, str):
                            raise TypeError(
                                "streaming thinking block 'thinking' must be a str"
                            )
                        if not isinstance(signature, str):
                            raise TypeError(
                                "streaming thinking block 'signature' must be a str"
                            )

                        if thinking and signature:
                            raise ValueError(
                                "Both `thinking` and `signature` in a single streaming chunk isn't supported."
                            )

                        return "thinking", ChatCompletionThinkingBlock(
                            type="thinking", thinking=thinking, signature=signature
                        )
            # OpenAI 互換の reasoning バックエンド（例: vLLM/SGLang の reasoning パーサー）は
            # ``thinking_blocks`` なしで ``reasoning_content`` を設定する。
            # ``Delta`` は未設定時に ``thinking_blocks`` 属性を削除するため、上記のブランチは
            # 完全にスキップされる。ここで ``thinking`` ブロックを開き、マッチする
            # ``thinking_delta`` ストリームがテキストブロックに出力されないようにする。
            elif isinstance(choice, StreamingChoices) and getattr(
                choice.delta, "reasoning_content", None
            ):
                return "thinking", ChatCompletionThinkingBlock(
                    type="thinking", thinking="", signature=""
                )

        return "text", TextBlock(type="text", text="")

    @staticmethod
    def _translate_streaming_openai_chunk_to_anthropic(
        choices: List[Union[OpenAIStreamingChoice, StreamingChoices]],
    ) -> Tuple[
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"],
        Union[
            ContentTextBlockDelta,
            ContentJsonBlockDelta,
            ContentThinkingBlockDelta,
            ContentThinkingSignatureBlockDelta,
        ],
    ]:
        """ストリーミングチャンクのデルタを Anthropic の block delta に変換する。

        テキスト / tool 引数（input_json） / thinking / signature ストリーミングデルタの
        ``(delta_type, delta)`` を返す。
        """
        text: str = ""
        reasoning_content: str = ""
        reasoning_signature: str = ""
        partial_json: Optional[str] = None
        for choice in choices:
            if choice.delta.content is not None and len(choice.delta.content) > 0:
                text += choice.delta.content
            if choice.delta.tool_calls:
                partial_json = ""
                for tool in choice.delta.tool_calls:
                    if (
                        tool.function is not None
                        and tool.function.arguments is not None
                    ):
                        partial_json = (partial_json or "") + tool.function.arguments
            elif isinstance(choice, StreamingChoices) and hasattr(
                choice.delta, "thinking_blocks"
            ):
                thinking_blocks = choice.delta.thinking_blocks or []
                if len(thinking_blocks) > 0:
                    for thinking_block in thinking_blocks:
                        if thinking_block["type"] == "thinking":
                            thinking = thinking_block.get("thinking") or ""
                            signature = thinking_block.get("signature") or ""

                            if not isinstance(thinking, str):
                                raise TypeError(
                                    "streaming thinking block 'thinking' must be a str"
                                )
                            if not isinstance(signature, str):
                                raise TypeError(
                                    "streaming thinking block 'signature' must be a str"
                                )

                            reasoning_content += thinking
                            reasoning_signature += signature
            # thinking_blocks がない場合は reasoning_content を処理する
            # reasoning_content を返す OpenRouter などのプロバイダーに対応する
            elif isinstance(choice, StreamingChoices) and hasattr(
                choice.delta, "reasoning_content"
            ):
                if choice.delta.reasoning_content is not None:
                    reasoning_content += choice.delta.reasoning_content

        if reasoning_content and reasoning_signature:
            raise ValueError(
                "Both `reasoning` and `signature` in a single streaming chunk isn't supported."
            )

        if partial_json is not None:
            return "input_json_delta", ContentJsonBlockDelta(
                type="input_json_delta", partial_json=partial_json
            )
        elif reasoning_content:
            return "thinking_delta", ContentThinkingBlockDelta(
                type="thinking_delta", thinking=reasoning_content
            )
        elif reasoning_signature:
            return "signature_delta", ContentThinkingSignatureBlockDelta(
                type="signature_delta", signature=reasoning_signature
            )
        else:
            return "text_delta", ContentTextBlockDelta(type="text_delta", text=text)

    def translate_streaming_openai_response_to_anthropic(
        self,
        response: ModelResponse,
        current_content_block_index: int,
        applied_edits: Optional[List[AppliedEdit]] = None,
    ) -> Union[ContentBlockDelta, MessageBlockDelta]:
        """ストリーミング OpenAI チャンク1件を Anthropic のデルタイベントに変換する。

        ``message_delta``（stop_reason/usage を持つ最終チャンク）または
        中間チャンクの ``content_block_delta`` を生成する。
        """
        ## ベースケース - finish reason を持つ最終チャンク
        if response.choices[0].finish_reason is not None:
            delta = MessageDelta(
                stop_reason=self._translate_openai_finish_reason_to_anthropic(
                    response.choices[0].finish_reason
                ),
            )
            if getattr(response, "usage", None) is not None:
                usage_chunk: Optional[Usage] = response.usage  # type: ignore
            else:
                usage_chunk = None
            if usage_chunk is not None:
                usage_delta: UsageDelta = self._build_usage_dict(usage_chunk)  # type: ignore[assignment]
            else:
                usage_delta = UsageDelta(input_tokens=0, output_tokens=0)
            message_block = MessageBlockDelta(
                type="message_delta", delta=delta, usage=usage_delta  # type: ignore
            )
            if applied_edits:
                message_block["context_management"] = ContextManagementResponse(
                    applied_edits=list(applied_edits)
                )
            return message_block
        (
            type_of_content,
            content_block_delta,
        ) = self._translate_streaming_openai_chunk_to_anthropic(
            choices=response.choices  # type: ignore
        )
        return ContentBlockDelta(
            type="content_block_delta",
            index=current_content_block_index,
            delta=content_block_delta,
        )
