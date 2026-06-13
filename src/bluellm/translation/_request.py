import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from bluellm.translation._tools import (
    truncate_tool_name,
)
from bluellm.types.anthropic_request import (
    AnthopicMessagesAssistantMessageParam,
    AnthropicMessagesRequest,
    AnthropicMessagesUserMessageParam,
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
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
)


class _RequestMixin:
    """Anthropic Messages リクエストを OpenAI Chat Completions 形式へ変換する変換群。"""

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
        ) = self.translate_anthropic_to_openai(
            anthropic_message_request=request_body
        )

        return translated_body, tool_name_mapping

    ### [BETA] `/v1/messages` エンドポイントサポート用

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
