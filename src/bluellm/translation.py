import copy
import hashlib
import json
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

from bluellm._compat import is_reasoning_auto_summary_enabled

# OpenAI 形式に変換が必要な Anthropic リクエストパラメーターのセット。
# _copy_untranslated_anthropic_params でこれ以外のキーをそのまま転送するために使用する。
_TRANSLATABLE_ANTHROPIC_PARAMS: frozenset = frozenset(
    [
        "messages",
        "metadata",
        "system",
        "tool_choice",
        "tools",
        "thinking",
        "output_format",
        "output_config",
    ]
)

# OpenAI は function/tool 名を64文字に制限している
# Anthropic にはこの制限がないため、長い名前を切り詰める必要がある
OPENAI_MAX_TOOL_NAME_LENGTH = 64
TOOL_NAME_HASH_LENGTH = 8
TOOL_NAME_PREFIX_LENGTH = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1  # 55


def truncate_tool_name(name: str) -> str:
    """
    OpenAI の64文字制限を超える tool 名を切り詰める。

    複数の tool が類似した長い名前を持つ場合の衝突を避けるため、
    {55文字のプレフィックス}_{8文字のハッシュ} 形式を使用する。

    Args:
        name: 元の tool 名

    Returns:
        64文字以下の場合は元の名前、それ以外はハッシュ付きの切り詰め済み名前
    """
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name

    # 衝突を避けるため、完全な名前から決定論的ハッシュを生成する
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{name_hash}"


from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

from bluellm._compat import parse_tool_call_arguments
from bluellm._compat import THOUGHT_SIGNATURE_SEPARATOR
from bluellm._compat import PolyfillResult
from bluellm.types.anthropic import (
    ANTHROPIC_HOSTED_TOOLS,
    AllAnthropicToolsValues,
    AnthopicMessagesAssistantMessageParam,
    AnthropicFinishReason,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicMessagesUserMessageParam,
    AnthropicResponseContentBlockRedactedThinking,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
    AppliedEdit,
    ContentBlockDelta,
    ContentJsonBlockDelta,
    ContentTextBlockDelta,
    ContentThinkingBlockDelta,
    ContentThinkingSignatureBlockDelta,
    ContextManagementResponse,
    MessageBlockDelta,
    MessageDelta,
    UsageDelta,
    UsageIteration,
)
from bluellm.types.anthropic import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from bluellm.types.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionImageObject,
    ChatCompletionImageUrlObject,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionRequest,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolChoiceValues,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
)
from bluellm._compat import Choices, ModelResponse, StreamingChoices, Usage

from bluellm.streaming import BlueLLMStreamWrapper

if TYPE_CHECKING:
    from bluellm.types.anthropic import ContentBlockContentBlockDict


class UnsupportedContentError(ValueError):
    """Anthropic の content block を OpenAI リクエスト形式で表現できない場合に発生する。

    プロキシは変換できないコンテンツをサイレントに削除してはならない
    （``file``/file_id 参照などの未知の image/document source type、
    またはペイロードのない base64/url source）。そうすると、モデルはそのブロックが
    送信されなかったかのように回答してしまう。この例外により、サーバーは問題を
    400 ``invalid_request_error`` として表面化できる。
    無効なリクエスト値を示すため ``ValueError`` のサブクラスとする。
    """


class BlueLLMAdapter:
    """非ストリーミングエントリーポイント向けのメッセージ変換ファサード。

    リクエストハンドラーが使用する ``translate_completion_input_params_with_tool_mapping``
    および ``translate_completion_output_params`` を公開する。
    実際の処理は :class:`BlueLLMMessagesAdapter` が担当する。ステートレスで再利用可能。
    """

    def __init__(self) -> None:
        """インスタンス状態なし。アダプターはステートレス。"""
        pass

    def translate_completion_input_params_with_tool_mapping(
        self, kwargs
    ) -> Tuple[Optional[ChatCompletionRequest], Dict[str, str]]:
        """
        Anthropic リクエストパラメーターを OpenAI 形式に変換し、tool 名マッピングを返す。

        OpenAI の64文字制限を超える tool 名の切り詰めを処理する。
        マッピングはレスポンスの変換時に元の名前を復元するために使用する。

        Returns:
            (openai_request, tool_name_mapping) のタプル
            - tool_name_mapping は切り詰めた tool 名を元の名前にマッピングする
        """

        #########################################################
        # 必須パラメーターの検証
        #########################################################
        model = kwargs.pop("model")
        messages = kwargs.pop("messages")
        if not model:
            raise ValueError(
                "Bad Request: model is required for Anthropic Messages Request"
            )
        if not messages:
            raise ValueError(
                "Bad Request: messages is required for Anthropic Messages Request"
            )

        #########################################################
        # 型付きリクエストボディの生成
        #########################################################
        request_body = AnthropicMessagesRequest(
            model=model, messages=messages, **kwargs
        )

        (
            translated_body,
            tool_name_mapping,
        ) = BlueLLMMessagesAdapter().translate_anthropic_to_openai(
            anthropic_message_request=request_body
        )

        return translated_body, tool_name_mapping

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
        return BlueLLMMessagesAdapter().translate_openai_response_to_anthropic(
            response=response,
            tool_name_mapping=tool_name_mapping,
            polyfill_result=polyfill_result,
        )

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


class BlueLLMMessagesAdapter:
    """Anthropic Messages 形式と OpenAI Chat Completions 形式（入力パラメーター、
    出力コンテンツ、ストリーミングチャンク）の間のコアトランスレーター。

    リクエスト間でステートレスかつ再利用可能。
    """

    def __init__(self):
        """インスタンス状態なし。アダプターはステートレス。"""
        pass

    ### [BETA] `/v1/messages` エンドポイントサポート用

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

    @staticmethod
    def _is_web_search_tool(tool: Dict[str, Any]) -> bool:
        """
        Anthropic の web search tool かどうかを確認する。

        Anthropic の web search tool の特徴:
        - type が "web_search" で始まる（例: "web_search_20260209"）
        - name = "web_search"

        Args:
            tool: tool 定義の dict

        Returns:
            web search tool の場合 True
        """
        tool_type = tool.get("type", "")
        tool_name = tool.get("name", "")
        return (
            isinstance(tool_type, str) and tool_type.startswith("web_search")
        ) or tool_name == "web_search"

    def translate_anthropic_messages_to_openai(
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
        model: Optional[str] = None,
    ) -> List:
        """Anthropic メッセージを OpenAI チャットメッセージリストに変換する。

        各メッセージを :meth:`_translate_user_message` または
        :meth:`_translate_assistant_message` にディスパッチし、元の順序を保持する。
        """
        new_messages: List[AllMessageValues] = []
        for m in messages:
            ## ユーザーメッセージ ##
            if m["role"] == "user":
                self._translate_user_message(
                    cast(Dict[str, Any], m), new_messages, model
                )

            ## アシスタントメッセージ ##
            if m["role"] == "assistant":
                self._translate_assistant_message(
                    cast(Dict[str, Any], m), new_messages, model
                )

        return new_messages

    def _translate_user_message(
        self,
        m: Dict[str, Any],
        new_messages: List[AllMessageValues],
        model: Optional[str],
    ) -> None:
        """Anthropic ユーザーメッセージ1件を OpenAI メッセージに変換する。

        tool メッセージ、プレーン文字列の ``user_message``、構造化ユーザーコンテンツリストを
        構築し、元の確認順序（tool メッセージ、文字列ユーザーメッセージ、構造化
        ``{"role": "user", "content": [...]}`` ブロックの順）で ``new_messages`` に追加する。
        動作は元のインラインブランチと完全に一致する。
        """
        user_message: Optional[ChatCompletionUserMessage] = None
        tool_message_list: List[ChatCompletionToolMessage] = []
        new_user_content_list: List[
            Union[ChatCompletionTextObject, ChatCompletionImageObject]
        ] = []
        message_content = m.get("content")
        if message_content and isinstance(message_content, str):
            user_message = ChatCompletionUserMessage(
                role="user", content=message_content
            )
        elif message_content and isinstance(message_content, list):
            for content in message_content:
                if content.get("type") == "text":
                    text_obj = ChatCompletionTextObject(
                        type="text", text=content.get("text", "")
                    )
                    self._add_cache_control_if_applicable(content, text_obj, model)
                    new_user_content_list.append(text_obj)  # type: ignore
                elif content.get("type") == "image":
                    # Anthropic の image 形式を OpenAI 形式に変換する
                    source = content.get("source", {})
                    openai_image_url = self._translate_anthropic_image_to_openai(
                        cast(dict, source)
                    )
                    image_url_obj = ChatCompletionImageUrlObject(url=openai_image_url)
                    image_obj = ChatCompletionImageObject(
                        type="image_url", image_url=image_url_obj
                    )
                    self._add_cache_control_if_applicable(content, image_obj, model)
                    new_user_content_list.append(image_obj)  # type: ignore
                elif content.get("type") == "document":
                    # Anthropic の document 形式（PDF など）を OpenAI 形式に変換する
                    source = content.get("source", {})
                    openai_image_url = self._translate_anthropic_image_to_openai(
                        cast(dict, source)
                    )
                    image_url_obj = ChatCompletionImageUrlObject(url=openai_image_url)
                    doc_obj = ChatCompletionImageObject(
                        type="image_url", image_url=image_url_obj
                    )
                    self._add_cache_control_if_applicable(content, doc_obj, model)
                    new_user_content_list.append(doc_obj)  # type: ignore
                elif content.get("type") == "tool_result":
                    self._translate_tool_result(
                        cast(Dict[str, Any], content), tool_message_list, model
                    )

        if len(tool_message_list) > 0:
            new_messages.extend(tool_message_list)
        if user_message is not None:
            new_messages.append(user_message)
        if len(new_user_content_list) > 0:
            new_messages.append(
                {"role": "user", "content": new_user_content_list}  # type: ignore
            )

    def _translate_assistant_message(
        self,
        m: Dict[str, Any],
        new_messages: List[AllMessageValues],
        model: Optional[str],
    ) -> None:
        """Anthropic アシスタントメッセージ1件を OpenAI メッセージに変換する。

        テキスト（cache_control 処理を含む）、tool_use 呼び出し、thinking/redacted_thinking
        ブロックを収集し、コンテンツ形式を選択する（テキストブロックに cache_control がある
        場合はリスト形式、それ以外は連結文字列）。出力するものがある場合に
        ``new_messages`` にアシスタントメッセージを1件追加する。
        動作は元のインラインブランチと完全に一致する。
        """
        assistant_message_str: Optional[str] = None
        assistant_content_list: List[Dict[str, Any]] = (
            []
        )  # cache_control を持つ content block 用
        has_cache_control_in_text = False
        tool_calls: List[ChatCompletionAssistantToolCall] = []
        thinking_blocks: List[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ] = []
        if isinstance(m.get("content"), str):
            assistant_message_str = str(m.get("content", ""))
        elif isinstance(m.get("content"), list):
            for content in m.get("content", []):
                if isinstance(content, str):
                    assistant_message_str = str(content)
                elif isinstance(content, dict):
                    if content.get("type") == "text":
                        if self._collect_assistant_text_block(
                            content, assistant_content_list, model
                        ):
                            has_cache_control_in_text = True
                    elif content.get("type") == "tool_use":
                        tool_calls.append(
                            self._build_assistant_tool_call(content, model)
                        )
                    elif content.get("type") == "thinking":
                        thinking_block = ChatCompletionThinkingBlock(
                            type="thinking",
                            thinking=content.get("thinking") or "",
                            signature=content.get("signature") or "",
                            cache_control=content.get("cache_control", {}),
                        )
                        thinking_blocks.append(thinking_block)
                    elif content.get("type") == "redacted_thinking":
                        redacted_thinking_block = ChatCompletionRedactedThinkingBlock(
                            type="redacted_thinking",
                            data=content.get("data") or "",
                            cache_control=content.get("cache_control", {}),
                        )
                        thinking_blocks.append(redacted_thinking_block)

        if (
            assistant_message_str is not None
            or len(assistant_content_list) > 0
            or len(tool_calls) > 0
            or len(thinking_blocks) > 0
        ):
            assistant_content = self._select_assistant_content(
                assistant_message_str,
                assistant_content_list,
                has_cache_control_in_text,
            )

            assistant_message = ChatCompletionAssistantMessage(
                role="assistant",
                content=assistant_content,
                thinking_blocks=(thinking_blocks if len(thinking_blocks) > 0 else None),
            )
            if len(tool_calls) > 0:
                assistant_message["tool_calls"] = tool_calls  # type: ignore
            if len(thinking_blocks) > 0:
                assistant_message["thinking_blocks"] = thinking_blocks  # type: ignore
            new_messages.append(assistant_message)

    def _collect_assistant_text_block(
        self,
        content: Dict[str, Any],
        assistant_content_list: List[Dict[str, Any]],
        model: Optional[str],
    ) -> bool:
        """Anthropic の text ブロックを content block へ変換し蓄積する。

        cache_control を適用したうえで ``assistant_content_list`` に追加し、
        cache_control が付与された場合に ``True`` を返す（呼び出し側で
        ``has_cache_control_in_text`` の更新に使用する）。
        """
        text_block: Dict[str, Any] = {
            "type": "text",
            "text": content.get("text", ""),
        }
        self._add_cache_control_if_applicable(content, text_block, model)
        assistant_content_list.append(text_block)
        return "cache_control" in text_block

    def _build_assistant_tool_call(
        self,
        content: Dict[str, Any],
        model: Optional[str],
    ) -> ChatCompletionAssistantToolCall:
        """Anthropic の tool_use ブロックを OpenAI tool call へ変換する。

        OpenAI の64文字制限のため tool 名を切り詰め、thought_signature を
        provider_specific_fields に格納し、cache_control を適用する。
        """
        # OpenAI の64文字制限のため tool 名を切り詰める
        tool_name = truncate_tool_name(content.get("name", ""))
        function_chunk: ChatCompletionToolCallFunctionChunk = {
            "name": tool_name,
            "arguments": json.dumps(content.get("input", {})),
        }
        signature = self._extract_signature_from_tool_use_content(
            cast(Dict[str, Any], content)
        )

        if signature:
            provider_specific_fields: Dict[str, Any] = (
                function_chunk.get("provider_specific_fields") or {}
            )
            provider_specific_fields["thought_signature"] = signature
            function_chunk["provider_specific_fields"] = provider_specific_fields

        tool_call = ChatCompletionAssistantToolCall(
            id=content.get("id", ""),
            type="function",
            function=function_chunk,
        )
        self._add_cache_control_if_applicable(content, tool_call, model)
        return tool_call

    @staticmethod
    def _select_assistant_content(
        assistant_message_str: Optional[str],
        assistant_content_list: List[Dict[str, Any]],
        has_cache_control_in_text: bool,
    ) -> Any:
        """アシスタントメッセージの content 形式を選択する。

        テキストブロックに cache_control があればリスト形式、cache_control が
        なければテキストブロックを連結した文字列、それ以外は素の文字列を返す。
        """
        # テキストブロックに cache_control があればリスト形式、それ以外は文字列を使用する
        if has_cache_control_in_text and len(assistant_content_list) > 0:
            return assistant_content_list
        if len(assistant_content_list) > 0 and not has_cache_control_in_text:
            # cache_control がない場合はテキストブロックを文字列に連結する
            return "".join(block.get("text", "") for block in assistant_content_list)
        return assistant_message_str

    def _translate_tool_result(
        self,
        content: Dict[str, Any],
        tool_message_list: List[ChatCompletionToolMessage],
        model: Optional[str],
    ) -> None:
        """Anthropic の ``tool_result`` ブロック1件を OpenAI tool メッセージに変換する。

        ``tool_message_list`` に追加する（各 tool_result につき1つの結合メッセージなので、
        各 ``tool_use`` は正確に1つの tool メッセージに対応する）。先頭の
        ``tool_result_start`` スナップショットと末尾の ``is_error`` マーカー（M9）により、
        マーカーはこの tool_result が生成したメッセージにのみ付与される。
        動作は元のインラインブランチと完全に一致する。
        """
        tool_result_start = len(tool_message_list)
        if "content" not in content:
            self._append_tool_result_message(
                content, "", tool_message_list, model
            )
        elif isinstance(content.get("content"), str):
            self._append_tool_result_message(
                content, str(content.get("content", "")), tool_message_list, model
            )
        elif isinstance(content.get("content"), list):
            # 同じ ID を持つ複数の tool_result ブロックが生成されないよう、
            # すべての content アイテムを1つの tool メッセージに結合する
            # （各 tool_use は正確に1つの tool_result を持つ必要がある）
            content_items = list(content.get("content", []))

            # 単一アイテムの content は文字列/URL 形式の後方互換性を維持する
            if len(content_items) == 1:
                c = content_items[0]
                if isinstance(c, str):
                    self._append_tool_result_message(
                        content, c, tool_message_list, model
                    )
                elif isinstance(c, dict):
                    if c.get("type") == "text":
                        self._append_tool_result_message(
                            content, c.get("text", ""), tool_message_list, model
                        )
                    elif c.get("type") == "image":
                        source = c.get("source", {})
                        openai_image_url = self._translate_anthropic_image_to_openai(
                            cast(dict, source)
                        )
                        self._append_tool_result_message(
                            content, openai_image_url, tool_message_list, model
                        )
            else:
                # 複数の content アイテムは、すべてのアイテムを保持しながら
                # 1つの tool_use_id を持つ単一の tool メッセージに結合する
                combined_content_parts = self._build_combined_tool_result_content(
                    content_items
                )
                # 結合した content を持つ単一の tool メッセージを生成する
                if combined_content_parts:
                    self._append_tool_result_message(
                        content, combined_content_parts, tool_message_list, model
                    )
        if content.get("is_error"):
            # M9: Anthropic の tool_result の is_error フラグには OpenAI tool メッセージの
            # 相当フィールドがない。これがないと、モデルは失敗した tool 呼び出しを成功として
            # 扱ってしまう。エラーシグナルが保持されるよう、この tool_result が生成した
            # すべてのメッセージにマーカーを付与する。
            for tr in tool_message_list[tool_result_start:]:
                self._apply_tool_result_error_marker(tr)

    def _append_tool_result_message(
        self,
        content: Dict[str, Any],
        content_value: Any,
        tool_message_list: List[ChatCompletionToolMessage],
        model: Optional[str],
    ) -> None:
        """tool_result の content 値から OpenAI tool メッセージを構築し追加する。

        ``content`` の ``tool_use_id`` を流用してメッセージを生成し、
        cache_control を適用したうえで ``tool_message_list`` に追加する。
        各 content 形式（空/文字列/単一アイテム/結合リスト）に共通する
        ボイラープレートを集約する。
        """
        tool_result = ChatCompletionToolMessage(
            role="tool",
            tool_call_id=content.get("tool_use_id", ""),
            content=content_value,
        )
        self._add_cache_control_if_applicable(content, tool_result, model)
        tool_message_list.append(tool_result)  # type: ignore[arg-type]

    def _build_combined_tool_result_content(
        self,
        content_items: List[Any],
    ) -> List[Union[ChatCompletionTextObject, ChatCompletionImageObject]]:
        """複数の tool_result content アイテムを結合 content パーツに変換する。

        文字列および text/image dict を OpenAI の content オブジェクトへ変換して
        順に蓄積し、単一 tool メッセージへ結合するためのリストを返す。
        """
        combined_content_parts: List[
            Union[
                ChatCompletionTextObject,
                ChatCompletionImageObject,
            ]
        ] = []
        for c in content_items:
            if isinstance(c, str):
                combined_content_parts.append(
                    ChatCompletionTextObject(type="text", text=c)
                )
            elif isinstance(c, dict):
                if c.get("type") == "text":
                    combined_content_parts.append(
                        ChatCompletionTextObject(
                            type="text",
                            text=c.get("text", ""),
                        )
                    )
                elif c.get("type") == "image":
                    source = c.get("source", {})
                    openai_image_url = self._translate_anthropic_image_to_openai(
                        cast(dict, source)
                    )
                    combined_content_parts.append(
                        ChatCompletionImageObject(
                            type="image_url",
                            image_url=ChatCompletionImageUrlObject(
                                url=openai_image_url
                            ),
                        )
                    )
        return combined_content_parts

    @staticmethod
    def translate_anthropic_thinking_to_reasoning_effort(
        thinking: Dict[str, Any],
    ) -> Optional[str]:
        """
        Anthropic の thinking パラメーターを OpenAI の reasoning_effort に変換する。

        Anthropic thinking 形式: {'type': 'enabled'|'disabled', 'budget_tokens': int}
        OpenAI reasoning_effort: 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'default'

        マッピング:
        - budget_tokens >= 10000 -> 'high'
        - budget_tokens >= 5000  -> 'medium'
        - budget_tokens >= 2000  -> 'low'
        - budget_tokens < 2000   -> 'minimal'
        """
        if not isinstance(thinking, dict):
            return None

        thinking_type = thinking.get("type", "disabled")

        if thinking_type == "disabled":
            return None
        elif thinking_type == "enabled":
            budget_tokens = thinking.get("budget_tokens", 0)
            if budget_tokens >= 10000:
                return "high"
            elif budget_tokens >= 5000:
                return "medium"
            elif budget_tokens >= 2000:
                return "low"
            else:
                return "minimal"
        elif thinking_type == "adaptive":
            # Adaptive thinking: effort は budget_tokens ではなく output_config.effort で制御される。
            # デフォルト値を返す。利用可能な場合、呼び出し元は output_config.effort で上書きすること。
            return "medium"

        return None

    @staticmethod
    def build_reasoning_effort_param(
        thinking: Dict[str, Any], output_config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Anthropic thinking から OpenAI の ``reasoning_effort`` 値を構築する。

        summary が要求されている場合（``thinking.summary`` で明示的に指定、または
        ``BLUELLM_REASONING_AUTO_SUMMARY`` オプトインによる）は ``{"effort", "summary"}``
        の dict を返す（M1: 古い effort のみの変換で失われていた summary を保持する）。
        Adaptive thinking は ``output_config.effort`` から effort を取得する。
        thinking が effort にマッピングされない場合は ``None`` を返す。
        """
        reasoning_effort = (
            BlueLLMMessagesAdapter.translate_anthropic_thinking_to_reasoning_effort(
                thinking
            )
        )
        if not reasoning_effort:
            return None
        if (
            isinstance(thinking, dict)
            and thinking.get("type") == "adaptive"
            and isinstance(output_config, dict)
            and output_config.get("effort")
        ):
            reasoning_effort = output_config["effort"]
        summary = thinking.get("summary") if isinstance(thinking, dict) else None
        if summary:
            return {"effort": reasoning_effort, "summary": summary}
        if is_reasoning_auto_summary_enabled():
            return {"effort": reasoning_effort, "summary": "detailed"}
        return reasoning_effort

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
    def translate_anthropic_tool_choice_to_openai(
        tool_choice: AnthropicMessagesToolChoice,
    ) -> ChatCompletionToolChoiceValues:
        """Anthropic の ``tool_choice`` を OpenAI の同等形式に変換する。"""
        if tool_choice["type"] == "any":
            return "required"
        elif tool_choice["type"] == "auto":
            return "auto"
        elif tool_choice["type"] == "tool":
            # OpenAI の64文字制限を超える場合は tool 名を切り詰める
            original_name = tool_choice.get("name", "")
            truncated_name = truncate_tool_name(original_name)
            tc_function_param = ChatCompletionToolChoiceFunctionParam(
                name=truncated_name
            )
            return ChatCompletionToolChoiceObjectParam(
                type="function", function=tc_function_param
            )
        elif tool_choice["type"] == "none":
            return "none"
        else:
            raise ValueError(
                "Incompatible tool choice param submitted - {}".format(tool_choice)
            )

    def translate_anthropic_tools_to_openai(
        self, tools: List[AllAnthropicToolsValues], model: Optional[str] = None
    ) -> Tuple[List[ChatCompletionToolParam], Dict[str, str]]:
        """
        Anthropic の tools を OpenAI 形式に変換する。

        Returns:
            (translated_tools, tool_name_mapping) のタプル
            - tool_name_mapping は OpenAI の64文字制限を超えた tool の
              切り詰めた名前を元の名前にマッピングする
        """
        new_tools: List[ChatCompletionToolParam] = []
        tool_name_mapping: Dict[str, str] = {}
        mapped_tool_params = ["name", "input_schema", "description", "cache_control"]

        for idx, tool in enumerate(tools):
            # Anthropic ネイティブの tool はそのまま保持すべきかチェックする
            tool_type = tool.get("type", "")
            if any(tool_type.startswith(t.value) for t in ANTHROPIC_HOSTED_TOOLS):
                # Anthropic ネイティブの tool は元の形式のまま保持する
                new_tools.append(tool)  # type: ignore[arg-type]
                continue

            raw_name = tool.get("name")
            if raw_name is None or (
                isinstance(raw_name, str) and not str(raw_name).strip()
            ):
                original_name = f"unnamed_tool_{idx}"
            else:
                original_name = str(raw_name)
            truncated_name = truncate_tool_name(original_name)

            # 名前が切り詰められた場合はマッピングを保存する
            if truncated_name != original_name:
                tool_name_mapping[truncated_name] = original_name

            function_chunk = ChatCompletionToolParamFunctionChunk(
                name=truncated_name,
            )
            if "input_schema" in tool:
                function_chunk["parameters"] = tool["input_schema"]  # type: ignore
            if "description" in tool:
                function_chunk["description"] = tool["description"]  # type: ignore

            for k, v in tool.items():
                if k not in mapped_tool_params:  # 追加の computer kwargs をそのまま渡す
                    function_chunk.setdefault("parameters", {}).update({k: v})
            tool_param = ChatCompletionToolParam(
                type="function", function=function_chunk
            )
            self._add_cache_control_if_applicable(tool, tool_param, model)
            new_tools.append(tool_param)  # type: ignore[arg-type]

        return new_tools, tool_name_mapping  # type: ignore[return-value]

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
                BlueLLMMessagesAdapter._add_additional_properties_false(prop)

        # array の items を処理する
        if "items" in schema:
            BlueLLMMessagesAdapter._add_additional_properties_false(
                schema["items"]
            )

        # anyOf/oneOf/allOf を処理する
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                for sub_schema in schema[key]:
                    BlueLLMMessagesAdapter._add_additional_properties_false(
                        sub_schema
                    )

        # $defs / definitions を処理する
        for key in ("$defs", "definitions"):
            if key in schema:
                for def_schema in schema[key].values():
                    BlueLLMMessagesAdapter._add_additional_properties_false(
                        def_schema
                    )

    def _add_system_message_to_messages(
        self,
        new_messages: List[AllMessageValues],
        anthropic_message_request: AnthropicMessagesRequest,
    ) -> None:
        """リクエストに system メッセージが存在する場合、メッセージリストに追加する。"""
        if "system" not in anthropic_message_request:
            return
        system_content = anthropic_message_request["system"]
        if not system_content:
            return
        # system を文字列または content block の配列として処理する
        if isinstance(system_content, str):
            new_messages.insert(
                0,
                ChatCompletionSystemMessage(role="system", content=system_content),
            )
        elif isinstance(system_content, list):
            # Anthropic の system content block を OpenAI 形式に変換する
            openai_system_content: List[Dict[str, Any]] = []
            model_name = anthropic_message_request.get("model", "")
            for block in system_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_block: Dict[str, Any] = {
                        "type": "text",
                        "text": block.get("text", ""),
                    }
                    self._add_cache_control_if_applicable(block, text_block, model_name)
                    openai_system_content.append(text_block)
            if openai_system_content:
                new_messages.insert(
                    0,
                    ChatCompletionSystemMessage(role="system", content=openai_system_content),  # type: ignore
                )

    @staticmethod
    def _translate_metadata_to_openai(
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Anthropic リクエストから OpenAI リクエストへ metadata フィールドを変換する。"""
        if "metadata" in anthropic_message_request:
            metadata = anthropic_message_request["metadata"]
            if metadata and "user_id" in metadata:
                new_kwargs["user"] = metadata["user_id"]

        if "bluellm_metadata" in anthropic_message_request:
            new_kwargs["metadata"] = anthropic_message_request.pop("bluellm_metadata")

    def _translate_tool_choice_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Anthropic の tool_choice を OpenAI 形式に変換する。"""
        if "tool_choice" not in anthropic_message_request:
            return
        tool_choice = anthropic_message_request["tool_choice"]
        if not tool_choice:
            return
        new_kwargs["tool_choice"] = self.translate_anthropic_tool_choice_to_openai(
            tool_choice=cast(AnthropicMessagesToolChoice, tool_choice)
        )

    def _translate_tools_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> Dict[str, str]:
        """tools を変換し、必要に応じて web_search_options を取得する。"""
        if "tools" not in anthropic_message_request:
            return {}

        tools = anthropic_message_request["tools"]
        if not tools:
            return {}

        web_search_tools: List[AllAnthropicToolsValues] = []
        regular_tools: List[AllAnthropicToolsValues] = []
        for tool in tools:
            cast_tool = cast(Dict[str, Any], tool)
            if self._is_web_search_tool(cast_tool):
                web_search_tools.append(cast(AllAnthropicToolsValues, tool))
            else:
                regular_tools.append(cast(AllAnthropicToolsValues, tool))

        if web_search_tools:
            new_kwargs["web_search_options"] = {}  # type: ignore

        if not regular_tools:
            return {}

        translated_tools, tool_name_mapping = self.translate_anthropic_tools_to_openai(
            tools=regular_tools,
            model=new_kwargs.get("model"),
        )
        new_kwargs["tools"] = translated_tools
        return tool_name_mapping

    def _translate_thinking_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Anthropic の thinking を thinking または reasoning_effort に変換する。"""
        if "thinking" not in anthropic_message_request:
            return

        thinking = anthropic_message_request["thinking"]
        if not thinking:
            return

        model = new_kwargs.get("model", "")
        if self.is_anthropic_claude_model(model):
            new_kwargs["thinking"] = thinking  # type: ignore
            return

        reasoning_effort = self.build_reasoning_effort_param(
            cast(Dict[str, Any], thinking),
            anthropic_message_request.get("output_config"),
        )
        if reasoning_effort is not None:
            new_kwargs["reasoning_effort"] = cast(Any, reasoning_effort)

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

    def _copy_untranslated_anthropic_params(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """変換不要な Anthropic パラメーターをそのままコピーする。"""
        translatable_params = self.translatable_anthropic_params()
        for k, v in anthropic_message_request.items():
            if k not in translatable_params:  # 残りのパラメーターはそのまま渡す
                new_kwargs[k] = v  # type: ignore

    def translate_anthropic_to_openai(
        self, anthropic_message_request: AnthropicMessagesRequest
    ) -> Tuple[ChatCompletionRequest, Dict[str, str]]:
        """
        Anthropic の ``/v1/messages`` リクエストを OpenAI 形式に変換する beta Anthropic アダプター用メソッド。

        Returns:
            (openai_request, tool_name_mapping) のタプル
            - tool_name_mapping は OpenAI の64文字制限を超えた tool の
              切り詰めた名前を元の名前にマッピングする
        """
        new_messages: List[AllMessageValues] = []
        tool_name_mapping: Dict[str, str] = {}

        ## ANTHROPIC メッセージを OPENAI に変換
        messages_list: List[
            Union[
                AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam
            ]
        ] = cast(
            List[
                Union[
                    AnthropicMessagesUserMessageParam,
                    AnthopicMessagesAssistantMessageParam,
                ]
            ],
            anthropic_message_request["messages"],
        )
        new_messages = self.translate_anthropic_messages_to_openai(
            messages=messages_list,
            model=anthropic_message_request.get("model"),
        )
        ## メッセージに SYSTEM メッセージを追加
        self._add_system_message_to_messages(new_messages, anthropic_message_request)

        new_kwargs: ChatCompletionRequest = {
            "model": anthropic_message_request["model"],
            "messages": new_messages,
        }
        ## METADATA を変換
        self._translate_metadata_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## TOOL CHOICE を変換
        self._translate_tool_choice_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## TOOLS を変換
        tool_name_mapping = self._translate_tools_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## THINKING を変換
        self._translate_thinking_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        ## OUTPUT_FORMAT を RESPONSE_FORMAT に変換
        self._translate_output_format_to_openai(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )
        self._copy_untranslated_anthropic_params(
            anthropic_message_request=anthropic_message_request,
            new_kwargs=new_kwargs,
        )

        return new_kwargs, tool_name_mapping

    @staticmethod
    def _apply_tool_result_error_marker(
        tool_message: ChatCompletionToolMessage,
    ) -> None:
        """tool_result のコンテンツの先頭にエラーマーカーをインプレースで追加する。

        Anthropic の ``tool_result`` には OpenAI tool メッセージに対応するフィールドがない
        ``is_error`` フラグがある。リストコンテンツには先頭のテキストパーツとして、
        それ以外には文字列プレフィックスとしてマーカーを追加することで、
        モデルは tool 呼び出しが失敗したことを認識できる。
        """
        marker = "[tool_result is_error=true]"
        existing = tool_message.get("content")
        if isinstance(existing, list):
            tool_message["content"] = [
                ChatCompletionTextObject(type="text", text=marker),
                *existing,
            ]
        else:
            separator = "\n" if existing else ""
            tool_message["content"] = f"{marker}{separator}{existing or ''}"

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
        uncached_input_tokens = usage.prompt_tokens or 0
        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = (
                getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            )
            uncached_input_tokens -= cached_tokens

        anthropic_usage = AnthropicUsage(
            input_tokens=uncached_input_tokens,
            output_tokens=usage.completion_tokens or 0,
        )
        if (
            hasattr(usage, "_cache_creation_input_tokens")
            and usage._cache_creation_input_tokens > 0
        ):
            anthropic_usage["cache_creation_input_tokens"] = (
                usage._cache_creation_input_tokens
            )
        if cached_tokens > 0:
            anthropic_usage["cache_read_input_tokens"] = cached_tokens

        if polyfill_result is not None and polyfill_result.iterations_usage is not None:
            message_iteration: UsageIteration = {
                "type": "message",
                "input_tokens": uncached_input_tokens,
                "output_tokens": usage.completion_tokens or 0,
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
        from bluellm.types.anthropic import TextBlock

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
                uncached_input_tokens = usage_chunk.prompt_tokens or 0
                cached_tokens = 0
                if (
                    hasattr(usage_chunk, "prompt_tokens_details")
                    and usage_chunk.prompt_tokens_details
                ):
                    cached_tokens = (
                        getattr(
                            usage_chunk.prompt_tokens_details,
                            "cached_tokens",
                            0,
                        )
                        or 0
                    )
                    uncached_input_tokens -= cached_tokens

                usage_delta = UsageDelta(
                    input_tokens=uncached_input_tokens,
                    output_tokens=usage_chunk.completion_tokens or 0,
                )
                if (
                    hasattr(usage_chunk, "_cache_creation_input_tokens")
                    and usage_chunk._cache_creation_input_tokens > 0
                ):
                    usage_delta["cache_creation_input_tokens"] = (
                        usage_chunk._cache_creation_input_tokens
                    )
                if cached_tokens > 0:
                    usage_delta["cache_read_input_tokens"] = cached_tokens
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
