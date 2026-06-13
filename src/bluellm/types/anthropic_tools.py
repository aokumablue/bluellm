from typing import Any, Dict, List, Optional, Union

from typing_extensions import Literal, Required, TypedDict

from .openai import ChatCompletionCachedContent


class AnthropicMessagesToolChoice(TypedDict, total=False):
    type: Required[Literal["auto", "any", "tool", "none"]]
    name: str
    disable_parallel_tool_use: bool  # デフォルトはfalse


AnthropicInputSchema = TypedDict(
    "AnthropicInputSchema",
    {
        "type": Optional[str],
        "properties": Optional[dict],
        "additionalProperties": Optional[bool],
        "required": Optional[List[str]],
        "$defs": Optional[Dict],
        "strict": Optional[bool],
    },
    total=False,
)


class AnthropicOutputSchema(TypedDict, total=False):
    type: Required[Literal["json_schema"]]
    schema: Required[dict]


class AnthropicOutputConfig(TypedDict, total=False):
    """Claudeの出力動作を制御するための設定。"""

    effort: Literal["high", "medium", "low", "xhigh", "max"]
    format: AnthropicOutputSchema


class AnthropicMessagesTool(TypedDict, total=False):
    name: Required[str]
    description: str
    input_schema: Optional[AnthropicInputSchema]
    type: Literal["custom"]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    defer_loading: bool
    allowed_callers: Optional[List[str]]
    input_examples: Optional[List[Dict[str, Any]]]


class AnthropicComputerTool(TypedDict, total=False):
    display_width_px: Required[int]
    display_height_px: Required[int]
    display_number: int
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    type: Required[str]
    name: Required[str]


class AnthropicWebSearchUserLocation(TypedDict, total=False):
    city: Optional[str]
    country: Optional[str]
    region: Optional[str]
    timezone: Optional[str]
    type: Required[Literal["approximate"]]


class AnthropicWebSearchTool(TypedDict, total=False):
    name: Required[Literal["web_search"]]
    type: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    max_uses: Optional[int]
    user_location: Optional[AnthropicWebSearchUserLocation]
    defer_loading: Optional[bool]
    allowed_callers: Optional[List[str]]
    input_examples: Optional[List[Dict[str, Any]]]


class AnthropicHostedTools(TypedDict, total=False):  # bash_toolおよびtext_editor用
    type: Required[str]
    name: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    defer_loading: Optional[bool]
    allowed_callers: Optional[List[str]]
    input_examples: Optional[List[Dict[str, Any]]]


class AnthropicCodeExecutionTool(TypedDict, total=False):
    type: Required[str]
    name: Required[Literal["code_execution"]]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    defer_loading: Optional[bool]
    allowed_callers: Optional[List[str]]
    input_examples: Optional[List[Dict[str, Any]]]


class AnthropicMemoryTool(TypedDict, total=False):
    type: Required[str]
    name: Required[Literal["memory"]]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    defer_loading: Optional[bool]
    allowed_callers: Optional[List[str]]
    input_examples: Optional[List[Dict[str, Any]]]


class AnthropicToolSearchToolRegex(TypedDict, total=False):
    """ツール検出にregexパターンを使用するtool searchツール。"""

    type: Required[Literal["tool_search_tool_regex_20251119"]]
    name: Required[str]


class AnthropicToolSearchToolBM25(TypedDict, total=False):
    """ツール検出にBM25アルゴリズムを使用するtool searchツール。"""

    type: Required[Literal["tool_search_tool_bm25_20251119"]]
    name: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    defer_loading: Optional[bool]
    allowed_callers: Optional[List[str]]
    input_examples: Optional[List[Dict[str, Any]]]


class AnthropicAdvisorTool(TypedDict, total=False):
    """Advisorツール — 高速なexecutorモデルと高知能なadvisorモデルを組み合わせる。"""

    type: Required[Literal["advisor_20260301"]]
    name: Required[Literal["advisor"]]
    model: Required[str]
    max_uses: Optional[int]
    caching: Optional[dict]


AllAnthropicToolsValues = Union[
    AnthropicComputerTool,
    AnthropicHostedTools,
    AnthropicMessagesTool,
    AnthropicWebSearchTool,
    AnthropicCodeExecutionTool,
    AnthropicMemoryTool,
    AnthropicToolSearchToolRegex,
    AnthropicToolSearchToolBM25,
    AnthropicAdvisorTool,
]
