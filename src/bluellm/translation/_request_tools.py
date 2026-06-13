from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

from bluellm.translation._tools import (
    truncate_tool_name,
)
from bluellm.types.anthropic_common import (
    ANTHROPIC_HOSTED_TOOLS,
)
from bluellm.types.anthropic_request import (
    AnthropicMessagesRequest,
)
from bluellm.types.anthropic_tools import (
    AllAnthropicToolsValues,
    AnthropicMessagesToolChoice,
)
from bluellm.types.openai import (
    ChatCompletionRequest,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
    ChatCompletionToolChoiceValues,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)


class _RequestToolsMixin:
    """Anthropic の tools / tool_choice を OpenAI 形式へ変換する変換群。"""

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
