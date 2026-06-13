from typing import List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict

from .anthropic_common import ToolCaller
from .anthropic_context import ContextManagementResponse
from .openai import ChatCompletionThinkingBlock


class ContentTextBlockDelta(TypedDict):
    """
    'delta': {'type': 'text_delta', 'text': 'Hello'}
    """

    type: str
    text: str


class ContentCitationsBlockDelta(TypedDict):
    type: Literal["citations"]
    citation: dict


class ContentJsonBlockDelta(TypedDict):
    """
    "delta": {"type": "input_json_delta","partial_json": "{\"location\": \"San Fra"}}
    """

    type: str
    partial_json: str


class ContentThinkingBlockDelta(TypedDict):
    """
    "delta": {"type": "thinking_delta", "thinking": "Let me solve this step by step:"}}
    """

    type: Literal["thinking_delta"]
    thinking: str


class ContentThinkingSignatureBlockDelta(TypedDict):
    """
    "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}
    """

    type: Literal["signature_delta"]
    signature: str


class ContentBlockDelta(TypedDict):
    type: Literal["content_block_delta"]
    index: int
    delta: Union[
        ContentTextBlockDelta,
        ContentJsonBlockDelta,
        ContentCitationsBlockDelta,
        ContentThinkingBlockDelta,
        ContentThinkingSignatureBlockDelta,
    ]


class ContentBlockStop(TypedDict):
    type: Literal["content_block_stop"]
    index: int


class ToolUseBlock(TypedDict):
    """
    "content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}
    """

    id: str

    input: dict

    name: str

    type: Literal["tool_use"]
    caller: Optional[ToolCaller]


class TextBlock(TypedDict):
    text: str

    type: Literal["text"]


class ContentBlockStartToolUse(TypedDict):
    type: Literal["content_block_start"]
    id: str
    name: str
    input: dict
    content_block: ToolUseBlock


class ContentBlockStartText(TypedDict):
    type: Literal["content_block_start"]
    index: int
    content_block: TextBlock


ContentBlockContentBlockDict = Union[
    ToolUseBlock, TextBlock, ChatCompletionThinkingBlock
]

ContentBlockStart = Union[ContentBlockStartToolUse, ContentBlockStartText]


class MessageDelta(TypedDict, total=False):
    stop_reason: Optional[str]


class UsageDelta(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class MessageBlockDelta(TypedDict):
    """
    Anthropic
    chunk = {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens', 'stop_sequence': None}, 'usage': {'output_tokens': 10}}
    """

    type: Literal["message_delta"]
    delta: MessageDelta
    usage: UsageDelta
    context_management: NotRequired[ContextManagementResponse]


class MessageChunk(TypedDict, total=False):
    id: str
    type: str
    role: str
    model: str
    content: List
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    usage: UsageDelta


class MessageStartBlock(TypedDict):
    """
        Anthropic
        chunk = {
        "type": "message_start",
        "message": {
            "id": "msg_vrtx_011PqREFEMzd3REdCoUFAmdG",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-sonnet-20240229",
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {
                "input_tokens": 270,
                "output_tokens": 1
            }
        }
    }
    """

    type: Literal["message_start"]
    message: MessageChunk
