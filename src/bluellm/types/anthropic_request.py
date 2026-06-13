from typing import Any, Dict, Iterable, List, Optional, Union

from typing_extensions import Literal, Required, TypedDict

from .anthropic_common import (
    AnthropicMcpServerTool,
    AnthropicMetadata,
    ToolCaller,
)
from .anthropic_tools import (
    AllAnthropicToolsValues,
    AnthropicMessagesToolChoice,
    AnthropicOutputConfig,
    AnthropicOutputSchema,
)
from .openai import (
    ChatCompletionCachedContent,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)


class AnthropicMessagesTextParam(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class AnthropicMessagesToolUseParam(TypedDict, total=False):
    type: Required[Literal["tool_use"]]
    id: str
    name: str
    input: dict
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    caller: Optional[ToolCaller]


AnthropicMessagesAssistantMessageValues = Union[
    AnthropicMessagesTextParam,
    AnthropicMessagesToolUseParam,
    ChatCompletionThinkingBlock,
    ChatCompletionRedactedThinkingBlock,
]


class AnthopicMessagesAssistantMessageParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[AnthropicMessagesAssistantMessageValues]]]
    """システムメッセージの内容。"""

    role: Required[Literal["assistant"]]
    """メッセージ作成者のロール。この場合は `author`。"""

    name: str
    """参加者の任意の名前。

    同じロールの参加者を区別するためのモデル情報を提供する。
    """


class AnthropicContentParamSource(TypedDict):
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicContentParamSourceUrl(TypedDict):
    type: Literal["url"]
    url: str


class AnthropicContentParamSourceFileId(TypedDict):
    type: Literal["file"]
    file_id: str


class AnthropicMessagesContainerUploadParam(TypedDict, total=False):
    type: Required[Literal["container_upload"]]
    file_id: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class AnthropicMessagesImageParam(TypedDict, total=False):
    type: Required[Literal["image"]]
    source: Required[
        Union[
            AnthropicContentParamSource,
            AnthropicContentParamSourceFileId,
            AnthropicContentParamSourceUrl,
        ]
    ]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class CitationsObject(TypedDict):
    enabled: bool


class AnthropicMessagesDocumentParam(TypedDict, total=False):
    type: Required[Literal["document"]]
    source: Required[
        Union[
            AnthropicContentParamSource,
            AnthropicContentParamSourceFileId,
            AnthropicContentParamSourceUrl,
        ]
    ]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]
    title: str
    context: str
    citations: Optional[CitationsObject]


class AnthropicMessagesToolResultContent(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class AnthropicMessagesToolResultParam(TypedDict, total=False):
    type: Required[Literal["tool_result"]]
    tool_use_id: Required[str]
    is_error: bool
    content: Union[
        str,
        Iterable[
            Union[
                AnthropicMessagesToolResultContent,
                AnthropicMessagesImageParam,
                AnthropicMessagesDocumentParam,
            ]
        ],
    ]
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


AnthropicMessagesUserMessageValues = Union[
    AnthropicMessagesTextParam,
    AnthropicMessagesImageParam,
    AnthropicMessagesToolResultParam,
    AnthropicMessagesDocumentParam,
    AnthropicMessagesContainerUploadParam,
]


class AnthropicMessagesUserMessageParam(TypedDict, total=False):
    role: Required[Literal["user"]]
    content: Required[Union[str, Iterable[AnthropicMessagesUserMessageValues]]]


AllAnthropicMessageValues = Union[
    AnthropicMessagesUserMessageParam, AnthopicMessagesAssistantMessageParam
]


class AnthropicMessagesRequestOptionalParams(TypedDict, total=False):
    max_tokens: Optional[int]
    metadata: Optional[Union[AnthropicMetadata, Dict]]
    stop_sequences: Optional[List[str]]
    stream: Optional[bool]
    system: Optional[Union[str, List]]
    temperature: Optional[float]
    thinking: Optional[Dict]
    tool_choice: Optional[Union[AnthropicMessagesToolChoice, Dict]]
    tools: Optional[List[Union[AllAnthropicToolsValues, Dict]]]
    top_k: Optional[int]
    inference_geo: Optional[str]
    top_p: Optional[float]
    mcp_servers: Optional[List[AnthropicMcpServerTool]]
    context_management: Optional[Dict[str, Any]]
    container: Optional[
        Dict[str, Any]
    ]  # コード実行のスキルを持つコンテナ設定
    output_format: Optional[AnthropicOutputSchema]  # 構造化出力のサポート
    speed: Optional[str]  # Opusモデル向けファストモードのサポート
    output_config: Optional[
        AnthropicOutputConfig
    ]  # Claudeの出力動作の設定
    cache_control: Optional[Dict[str, Any]]  # 自動プロンプトキャッシュ
    reasoning_effort: Optional[str]


class AnthropicMessagesRequest(AnthropicMessagesRequestOptionalParams, total=False):
    model: Required[str]
    messages: Required[Union[List[AllAnthropicMessageValues], List[Dict]]]
    bluellm_metadata: dict
