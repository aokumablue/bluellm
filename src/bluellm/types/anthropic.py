from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal, NotRequired, Required, TypeAlias, TypedDict

from .openai import (
    ChatCompletionCachedContent,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionUsageBlock,
)


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


ANTHROPIC_ADVISOR_TOOL_TYPE: Literal["advisor_20260301"] = "advisor_20260301"


class AnthropicAdvisorTool(TypedDict, total=False):
    """Advisorツール — 高速なexecutorモデルと高知能なadvisorモデルを組み合わせる。"""

    type: Required[Literal["advisor_20260301"]]
    name: Required[Literal["advisor"]]
    model: Required[str]
    max_uses: Optional[int]
    caching: Optional[dict]


class ToolReference(TypedDict, total=False):
    """遅延ロードされたツールから展開されるべきツールへの参照。"""

    type: Required[Literal["tool_reference"]]
    tool_name: Required[str]


class DirectToolCaller(TypedDict, total=False):
    """Claudeが直接ツールを呼び出したことを示す。"""

    type: Required[Literal["direct"]]


class CodeExecutionToolCaller(TypedDict, total=False):
    """コード実行からプログラム的にツールが呼び出されたことを示す。"""

    type: Required[Literal["code_execution_20250825"]]
    tool_id: Required[str]  # 呼び出しを行ったコード実行ツールのID


ToolCaller = Union[DirectToolCaller, CodeExecutionToolCaller]


class AnthropicContainer(TypedDict, total=False):
    """コード実行のコンテナメタデータ。"""

    id: Required[str]
    expires_at: Optional[str]  # ISO 8601タイムスタンプ


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


class AnthropicMcpServerToolConfiguration(TypedDict, total=False):
    allowed_tools: Optional[List[str]]


class AnthropicMcpServerTool(TypedDict, total=False):
    type: Required[Literal["url"]]
    url: Required[str]
    name: Required[str]
    tool_configuration: AnthropicMcpServerToolConfiguration
    authorization_token: str


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


class AnthropicCitationPageLocation(TypedDict, total=False):
    """
    ページベースの参照に対するAnthropicの引用。
    ページ番号を持つドキュメントから引用する際に使用する。
    """

    type: Literal["page_location"]
    cited_text: str  # 引用される正確なテキスト（出力tokenにはカウントされない）
    document_index: int  # 引用されたドキュメントを参照するインデックス
    document_title: Optional[str]  # 引用されたドキュメントのタイトル
    start_page_number: int  # 1始まりの開始ページ
    end_page_number: int  # 終了ページ（排他的）


class AnthropicCitationCharLocation(TypedDict, total=False):
    """
    文字ベースの参照に対するAnthropicの引用。
    文字位置を持つテキストから引用する際に使用する。
    """

    type: Literal["char_location"]
    cited_text: str  # 引用される正確なテキスト（出力tokenにはカウントされない）
    document_index: int  # 引用されたドキュメントを参照するインデックス
    document_title: Optional[str]  # 引用されたドキュメントのタイトル
    start_char_index: int  # 引用の開始文字インデックス
    end_char_index: int  # 引用の終了文字インデックス


# すべての引用フォーマットのUnion型
AnthropicCitation = Union[AnthropicCitationPageLocation, AnthropicCitationCharLocation]


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


class AnthropicMetadata(TypedDict, total=False):
    user_id: str


class AnthropicSystemMessageContent(TypedDict, total=False):
    type: str
    text: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


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


class AppliedEdit(TypedDict, total=False):
    """適用済みのcontext_management編集の1件（Anthropicレスポンス形式）。"""

    type: str
    cleared_input_tokens: int
    cleared_tool_uses: int
    cleared_thinking_turns: int
    # compact_20260112のフィールド
    summary_input_tokens: int
    summary_output_tokens: int
    error: str
    warnings: List[str]


class ContextManagementResponse(TypedDict, total=False):
    """``applied_edits`` を持つ ``context_management`` レスポンス。"""

    applied_edits: List[AppliedEdit]


class CompactionBlock(TypedDict, total=False):
    """合成された ``compaction`` コンテンツブロック（compact_20260112）。"""

    type: Required[Literal["compaction"]]
    content: Optional[str]


class UsageIteration(TypedDict, total=False):
    """1回のサンプリングイテレーションのtoken使用量（compact_20260112）。"""

    type: Required[Literal["compaction", "message"]]
    input_tokens: int
    output_tokens: int


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


AnthropicFinishReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"
]


class AnthropicChatCompletionUsageBlock(ChatCompletionUsageBlock, total=False):
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


ANTHROPIC_API_HEADERS = {
    "anthropic-version",
    "anthropic-beta",
}

ANTHROPIC_API_ONLY_HEADERS = {  # Vertex AIやBedrockでAnthropicを呼び出す場合は失敗する
    "anthropic-beta",
}


class AnthropicThinkingParam(TypedDict, total=False):
    type: Literal["enabled", "adaptive"]
    budget_tokens: int


class ANTHROPIC_HOSTED_TOOLS(str, Enum):
    WEB_SEARCH = "web_search"
    BASH = "bash"
    TEXT_EDITOR = "text_editor"
    CODE_EXECUTION = "code_execution"
    WEB_FETCH = "web_fetch"
    MEMORY = "memory"
    TOOL_SEARCH_TOOL = "tool_search_tool"


class ANTHROPIC_BETA_HEADER_VALUES(str, Enum):
    """
    Anthropicの既知のbetaヘッダー値。
    """

    WEB_FETCH_2025_09_10 = "web-fetch-2025-09-10"
    WEB_SEARCH_2025_03_05 = "web-search-2025-03-05"
    CONTEXT_MANAGEMENT_2025_06_27 = "context-management-2025-06-27"
    COMPACT_2026_01_12 = "compact-2026-01-12"
    STRUCTURED_OUTPUT_2025_09_25 = "structured-outputs-2025-11-13"
    ADVANCED_TOOL_USE_2025_11_20 = "advanced-tool-use-2025-11-20"
    FAST_MODE_2026_02_01 = "fast-mode-2026-02-01"
    ADVISOR_TOOL_2026_03_01 = "advisor-tool-2026-03-01"




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
    """
    リクエストで使用された入出力token数
    """

    input_tokens: int
    output_tokens: int

    """
    使用されたキャッシュtoken数
    """
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
