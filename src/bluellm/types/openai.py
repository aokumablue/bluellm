from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from typing_extensions import Required, TypedDict

from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam,
)


class ChatCompletionAudioDelta(TypedDict, total=False):
    data: str
    transcript: str
    expires_at: int
    id: str


class ChatCompletionToolCallFunctionChunk(TypedDict, total=False):
    name: Optional[str]
    arguments: str
    provider_specific_fields: Optional[Dict[str, Any]]


class ChatCompletionAssistantToolCall(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk


class ChatCompletionToolCallChunk(TypedDict):  # /chat/completions 呼び出しの結果
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk
    index: int


class ChatCompletionDeltaToolCallChunk(TypedDict, total=False):
    id: str
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk
    index: int


class ChatCompletionCachedContent(TypedDict):
    type: Literal["ephemeral"]


class ChatCompletionThinkingBlock(TypedDict, total=False):
    type: Required[Literal["thinking"]]
    thinking: str
    signature: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class ChatCompletionRedactedThinkingBlock(TypedDict, total=False):
    type: Required[Literal["redacted_thinking"]]
    data: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


class ChatCompletionReasoningSummaryTextBlock(TypedDict, total=False):
    type: Required[Literal["summary_text"]]
    text: str


class ChatCompletionReasoningItem(TypedDict, total=False):
    """会話履歴のラウンドトリップ用 OpenAI Responses API reasoning アイテムを表す。"""

    type: Required[Literal["reasoning"]]
    id: str
    encrypted_content: Optional[str]
    summary: List["ChatCompletionReasoningSummaryTextBlock"]


class WebSearchOptionsUserLocationApproximate(TypedDict, total=False):
    city: str
    """ユーザーの都市を示す自由テキスト入力（例: `San Francisco`）。"""

    country: str
    """
    ユーザーの2文字の [ISO 国コード](https://en.wikipedia.org/wiki/ISO_3166-1)
    （例: `US`）。
    """

    region: str
    """ユーザーの地域を示す自由テキスト入力（例: `California`）。"""

    timezone: str
    """
    ユーザーの [IANA タイムゾーン](https://timeapi.io/documentation/iana-timezones)
    （例: `America/Los_Angeles`）。
    """


class WebSearchOptionsUserLocation(TypedDict, total=False):
    approximate: Required[WebSearchOptionsUserLocationApproximate]
    """検索用のおおよその位置情報パラメータ。"""

    type: Required[Literal["approximate"]]
    """位置情報の近似タイプ。常に `approximate`。"""


class WebSearchOptions(TypedDict, total=False):
    search_context_size: Literal["low", "medium", "high"]
    """
    検索に使用するコンテキストウィンドウの量を示す高レベルのガイダンス。
    `low`、`medium`、`high` のいずれか。デフォルトは `medium`。
    """

    user_location: Optional[WebSearchOptionsUserLocation]
    """検索用のおおよその位置情報パラメータ。"""


class FileSearchTool(TypedDict, total=False):
    type: Literal["file_search"]
    """定義するツールのタイプ: `file_search`"""

    vector_store_ids: Optional[List[str]]
    """検索対象の vector store の ID 一覧。"""


class ChatCompletionAnnotationURLCitation(TypedDict, total=False):
    end_index: int
    """メッセージ内の URL 引用の最後の文字のインデックス。"""

    start_index: int
    """メッセージ内の URL 引用の最初の文字のインデックス。"""

    title: str
    """Webリソースのタイトル。"""

    url: str
    """Webリソースの URL。"""


class ChatCompletionAnnotation(TypedDict, total=False):
    type: Literal["url_citation"]
    """URL 引用のタイプ。常に `url_citation`。"""

    url_citation: ChatCompletionAnnotationURLCitation
    """Web 検索を使用した際の URL 引用。"""


class OpenAIChatCompletionTextObject(TypedDict):
    type: Literal["text"]
    text: str


class ChatCompletionTextObject(
    OpenAIChatCompletionTextObject, total=False
):
    cache_control: ChatCompletionCachedContent


class ChatCompletionImageUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str
    format: str


class ChatCompletionImageObject(TypedDict):
    type: Literal["image_url"]
    image_url: Union[str, ChatCompletionImageUrlObject]


class ChatCompletionVideoUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str


class ChatCompletionVideoObject(TypedDict):
    type: Literal["video_url"]
    video_url: Union[str, ChatCompletionVideoUrlObject]


class ChatCompletionAudioObject(ChatCompletionContentPartInputAudioParam):
    pass


class DocumentObject(TypedDict):
    type: Literal["text"]
    media_type: str
    data: str


class CitationsObject(TypedDict):
    enabled: bool


class ChatCompletionDocumentObject(TypedDict):
    type: Literal["document"]
    source: DocumentObject
    title: str
    context: str
    citations: Optional[CitationsObject]


class ChatCompletionFileObjectFile(TypedDict, total=False):
    file_data: str
    file_id: str
    filename: str
    format: str
    detail: str  # 動画・画像の解像度制御用（low, medium, high, ultra_high）
    video_metadata: Dict[
        str, Any
    ]  # 動画固有のメタデータ用（fps, start_offset, end_offset）


class ChatCompletionFileObject(TypedDict):
    type: Literal["file"]
    file: ChatCompletionFileObjectFile


OpenAIMessageContentListBlock = Union[
    ChatCompletionTextObject,
    ChatCompletionImageObject,
    ChatCompletionAudioObject,
    ChatCompletionDocumentObject,
    ChatCompletionVideoObject,
    ChatCompletionFileObject,
]

OpenAIMessageContent = Union[
    str,
    Iterable[OpenAIMessageContentListBlock],
]

# 補完を生成するためのプロンプト。文字列、文字列配列、token 配列、token 配列の配列としてエンコードされる。
AllPromptValues = Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None]


class OpenAIChatCompletionUserMessage(TypedDict):
    role: Literal["user"]
    content: OpenAIMessageContent


class OpenAITextCompletionUserMessage(TypedDict):
    role: Literal["user"]
    content: AllPromptValues


class ChatCompletionUserMessage(OpenAIChatCompletionUserMessage, total=False):
    cache_control: ChatCompletionCachedContent


class OpenAIChatCompletionAssistantMessage(TypedDict, total=False):
    role: Required[Literal["assistant"]]
    content: Optional[
        Union[
            str,
            Iterable[
                Union[
                    ChatCompletionTextObject,
                    ChatCompletionThinkingBlock,
                    ChatCompletionRedactedThinkingBlock,
                    ChatCompletionImageObject,
                ]
            ],
        ]
    ]
    name: Optional[str]
    tool_calls: Optional[List[ChatCompletionAssistantToolCall]]
    function_call: Optional[ChatCompletionToolCallFunctionChunk]
    reasoning_content: Optional[str]


class ChatCompletionAssistantMessage(OpenAIChatCompletionAssistantMessage, total=False):
    cache_control: ChatCompletionCachedContent
    thinking_blocks: Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ]
    reasoning_items: Optional[List[ChatCompletionReasoningItem]]


class ChatCompletionToolMessage(TypedDict):
    role: Literal["tool"]
    content: Union[str, Iterable[ChatCompletionTextObject]]
    tool_call_id: str


class ChatCompletionFunctionMessage(TypedDict):
    role: Literal["function"]
    content: Optional[Union[str, Iterable[ChatCompletionTextObject]]]
    name: str
    tool_call_id: Optional[str]


class OpenAIChatCompletionSystemMessage(TypedDict, total=False):
    role: Required[Literal["system"]]
    content: Required[Union[str, List]]
    name: str


class OpenAIChatCompletionDeveloperMessage(TypedDict, total=False):
    role: Required[Literal["developer"]]
    content: Required[Union[str, List]]
    name: str


class ChatCompletionSystemMessage(OpenAIChatCompletionSystemMessage, total=False):
    cache_control: ChatCompletionCachedContent


class ChatCompletionDeveloperMessage(OpenAIChatCompletionDeveloperMessage, total=False):
    cache_control: ChatCompletionCachedContent


class GenericChatCompletionMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[Union[str, List]]


ValidUserMessageContentTypes = [
    "text",
    "image_url",
    "input_audio",
    "audio_url",
    "document",
    "guarded_text",
    "video_url",
    "file",
]  # ユーザーメッセージのバリデーション用。Anthropic メッセージの誤送信を防ぐ。

ValidUserMessageContentTypesLiteral = Literal[
    "text",
    "image_url",
    "input_audio",
    "audio_url",
    "document",
    "guarded_text",
    "video_url",
    "file",
]

# アシスタントメッセージのコンテンツタイプ（text, thinking, redacted_thinking, image_url）
ValidAssistantMessageContentTypesLiteral = Literal[
    "text",
    "thinking",
    "redacted_thinking",
    "image_url",
]

ValidAssistantMessageContentTypes = [
    "text",
    "thinking",
    "redacted_thinking",
    "image_url",
]

# チャット補完メッセージの有効なコンテンツタイプの結合
ValidChatCompletionMessageContentTypesLiteral = Literal[
    "text",
    "image_url",
    "input_audio",
    "audio_url",
    "document",
    "guarded_text",
    "video_url",
    "file",
    "thinking",
    "redacted_thinking",
]

ValidChatCompletionMessageContentTypes = [
    "text",
    "image_url",
    "input_audio",
    "audio_url",
    "document",
    "guarded_text",
    "video_url",
    "file",
    "thinking",
    "redacted_thinking",
]

AllMessageValues = Union[
    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionSystemMessage,
    ChatCompletionFunctionMessage,
    ChatCompletionDeveloperMessage,
]


class ChatCompletionToolChoiceFunctionParam(TypedDict):
    name: str


class ChatCompletionToolChoiceObjectParam(TypedDict):
    type: Literal["function"]
    function: ChatCompletionToolChoiceFunctionParam


ChatCompletionToolChoiceStringValues = Literal["none", "auto", "required"]

ChatCompletionToolChoiceValues = Union[
    ChatCompletionToolChoiceStringValues, ChatCompletionToolChoiceObjectParam
]


class ChatCompletionToolParamFunctionChunk(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: dict
    strict: bool


class OpenAIChatCompletionToolParam(TypedDict):
    type: Union[Literal["function"], str]
    function: ChatCompletionToolParamFunctionChunk


class ChatCompletionToolParam(OpenAIChatCompletionToolParam, total=False):
    cache_control: ChatCompletionCachedContent


class Function(TypedDict, total=False):
    name: Required[str]
    """呼び出す関数の名前。"""


class ChatCompletionNamedToolChoiceParam(TypedDict, total=False):
    function: Required[Function]

    type: Required[Literal["function"]]
    """ツールのタイプ。現在は `function` のみサポート。"""


class ChatCompletionRequest(TypedDict, total=False):
    model: Required[str]
    messages: Required[List[AllMessageValues]]
    frequency_penalty: float
    logit_bias: dict
    logprobs: bool
    top_logprobs: int
    max_tokens: int
    n: int
    presence_penalty: float
    response_format: dict
    seed: int
    service_tier: str
    safety_identifier: str
    stop: Union[str, List[str]]
    stream_options: dict
    temperature: float
    top_p: float
    tools: List[ChatCompletionToolParam]
    tool_choice: ChatCompletionToolChoiceValues
    parallel_tool_calls: bool
    function_call: Union[str, dict]
    functions: List
    user: str
    metadata: dict
    reasoning_effort: str  # OpenAI o1/o3 の reasoning パラメータ


class ChatCompletionDeltaChunk(TypedDict, total=False):
    content: Optional[str]
    tool_calls: List[ChatCompletionDeltaToolCallChunk]
    role: str


ChatCompletionAssistantContentValue = (
    str  # 変数として保持、stream_chunk_builder でも使用
)


class ChatCompletionResponseMessage(TypedDict, total=False):
    content: Optional[ChatCompletionAssistantContentValue]
    annotations: Optional[List[ChatCompletionAnnotation]]
    tool_calls: Optional[List[ChatCompletionToolCallChunk]]
    role: Literal["assistant"]
    function_call: Optional[ChatCompletionToolCallFunctionChunk]
    provider_specific_fields: Optional[dict]
    reasoning_content: Optional[str]
    thinking_blocks: Optional[
        List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]
    ]


class ChatCompletionUsageBlock(TypedDict, total=False):
    prompt_tokens: Required[int]
    completion_tokens: Required[int]
    total_tokens: Required[int]
    prompt_tokens_details: Optional[dict]
    completion_tokens_details: Optional[dict]
