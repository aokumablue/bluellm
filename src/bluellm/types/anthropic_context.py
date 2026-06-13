from typing import List, Optional

from typing_extensions import Literal, Required, TypedDict


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
