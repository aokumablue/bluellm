from enum import Enum
from typing import List, Optional, Union

from typing_extensions import Literal, Required, TypedDict

from .openai import ChatCompletionCachedContent


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


class AnthropicMcpServerToolConfiguration(TypedDict, total=False):
    allowed_tools: Optional[List[str]]


class AnthropicMcpServerTool(TypedDict, total=False):
    type: Required[Literal["url"]]
    url: Required[str]
    name: Required[str]
    tool_configuration: AnthropicMcpServerToolConfiguration
    authorization_token: str


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


class AnthropicMetadata(TypedDict, total=False):
    user_id: str


class AnthropicSystemMessageContent(TypedDict, total=False):
    type: str
    text: str
    cache_control: Optional[Union[dict, ChatCompletionCachedContent]]


AnthropicFinishReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"
]


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
