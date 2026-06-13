from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal, NotRequired, TypeAlias, TypedDict

from .anthropic_context import ContextManagementResponse


class AnthropicResponseContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicResponseContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict
    provider_specific_fields: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")  # provider_specific_fieldsを許可


class AnthropicResponseContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str]


class AnthropicResponseContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: str


class AnthropicResponseTextBlock(TypedDict, total=False):
    """
    Anthropic Response Text Block: https://docs.anthropic.com/en/api/messages
    """

    citations: Optional[List[Dict[str, Any]]]
    text: str
    type: Literal["text"]


class AnthropicResponseToolUseBlock(TypedDict, total=False):
    """
    Anthropic Response Tool Use Block: https://docs.anthropic.com/en/api/messages
    """

    id: Optional[str]
    input: Optional[str]
    name: Optional[str]
    type: Literal["tool_use"]


class AnthropicResponseThinkingBlock(TypedDict, total=False):
    """
    Anthropic Response Thinking Block: https://docs.anthropic.com/en/api/messages
    """

    signature: Optional[str]
    thinking: Optional[str]
    type: Literal["thinking"]


class AnthropicResponseRedactedThinkingBlock(TypedDict, total=False):
    """
    Anthropic Response Redacted Thinking Block: https://docs.anthropic.com/en/api/messages
    """

    data: Optional[str]
    type: Literal["redacted_thinking"]


AnthropicResponseContentBlock: TypeAlias = Union[
    AnthropicResponseTextBlock,
    AnthropicResponseToolUseBlock,
    AnthropicResponseThinkingBlock,
    AnthropicResponseRedactedThinkingBlock,
]


class AnthropicUsage(TypedDict, total=False):
    """リクエストで使用されたtoken数。"""

    input_tokens: int
    output_tokens: int
    # 使用されたキャッシュtoken数
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class AnthropicMessagesResponse(TypedDict, total=False):
    """
    Anthropic Messages API Response: https://docs.anthropic.com/en/api/messages
    """

    content: Optional[
        List[
            Union[
                AnthropicResponseContentBlock,
                AnthropicResponseContentBlockText,
                AnthropicResponseContentBlockToolUse,
            ]
        ]
    ]
    id: str
    model: Optional[str]  # AnthropicのModelタイプを表す
    role: Optional[Literal["assistant"]]
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ]
    stop_sequence: Optional[str]
    type: Optional[Literal["message"]]
    usage: Optional[AnthropicUsage]
    context_management: NotRequired[ContextManagementResponse]
