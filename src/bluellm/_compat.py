"""translation モジュールと streaming モジュールがインポートする内部型の最小限のスタブ。

これらの型は openai SDK の `ChatCompletion` / `ChatCompletionChunk` と
属性互換なオブジェクトを対象に動作する。重い内部型
（`ModelResponse`、`Usage` など）はアノテーションや `isinstance`
ガード内にのみ現れ、OpenAI/Azure パスでは有効にならないため、
ここでは軽量な Protocol スタブとして提供する。

`StreamingChoices` は translation.py 内の `isinstance` ガードで使用されるため
（常に False だが挙動を維持する）空クラスのまま温存する。
他のクラス（`ModelResponse`、`Usage`、`Choices`）はアノテーション専用 Protocol
として定義し、mypy の `attr-defined` エラーを削減する。
`@runtime_checkable` は付与しない（isinstance 禁止）。
"""

from __future__ import annotations

import logging
import os
import uuid as uuid  # 再エクスポート用；移植コードが `uuid.uuid4()` を呼び出す
from typing import Any, List, Optional

from typing_extensions import Protocol

verbose_logger = logging.getLogger("bluellm")

THOUGHT_SIGNATURE_SEPARATOR = "__thought__"


def is_reasoning_auto_summary_enabled() -> bool:
    """デフォルトの 'summary: detailed' 注入が有効かどうかを返す（オプトイン）。"""
    return os.getenv("BLUELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"


# --- 内部型 Protocol スタブ -----------------------------------------------
# `@runtime_checkable` は付与しない（isinstance での使用禁止）。
# 実際の実行時オブジェクトは openai SDK の ChatCompletion / ChatCompletionChunk
# であり、これらのクラスのインスタンスにはならない。
class Usage(Protocol):
    """openai SDK の Usage オブジェクトと属性互換な Protocol。"""

    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: Any


class Choices(Protocol):
    """openai SDK の Choice オブジェクトと属性互換な Protocol。"""

    message: Any
    finish_reason: Any


class ModelResponse(Protocol):
    """openai SDK の ChatCompletion オブジェクトと属性互換な Protocol。"""

    id: str
    model: Optional[str]
    choices: List[Choices]
    usage: Usage


class ModelResponseStream(Protocol):
    """openai SDK の ChatCompletionChunk ストリームと属性互換な Protocol。"""


class StreamingChoices:  # noqa: D401 - isinstance ガード用；空クラスを温存（挙動は常に False）
    """translation.py の isinstance ガードで参照される空クラス（runtime 挙動維持）。"""


# コンテキスト管理ポリフィルは実行されない；`polyfill_result` は常に None。
# アノテーションが解決できるように存在するだけで、メンバーに到達することはない。
class PolyfillResult:  # noqa: D401 - placeholder
    compaction_block: Any = None
    iterations_usage: Any = None

    def applied_edits_for_response(self) -> Optional[list]:
        return None


# コンストラクタとイテレータプロトコルのエントリポイントのみを保持する：唯一のサブクラス
# （BlueLLMStreamWrapper）は常に独自の __next__/__anext__ を提供するため、
# 基底クラスの実装は到達不能だった（かつバグあり：エラーを暗黙の None yield に
# 飲み込み、async コルーチン内で StopIteration を発生させていた）。
# デッドコードとして持ち続けるのではなく、削除する。
class AdapterCompletionStreamWrapper:
    """上流 stream とイテレータエントリポイントを保持する基底 stream ラッパー。

    サブクラス（BlueLLMStreamWrapper）が __next__/__anext__ を提供する；この基底クラスは
    stream を保持し、__iter__/__aiter__ から ``self`` を返すだけである。
    """

    def __init__(self, completion_stream):
        """ラップした上流 completion stream を保存する。"""
        self.completion_stream = completion_stream

    def __iter__(self):
        """同期イテレータとして self を返す。"""
        return self

    def __aiter__(self):
        """非同期イテレータとして self を返す。"""
        return self


def _attempt_json_repair(s: str) -> Optional[Any]:
    """LLM tool call が生成した切り詰められた JSON の修復を試みる。"""
    import json

    stripped = s.rstrip()
    if not stripped:
        return None

    opener_stack: list = []
    in_string = False
    escape_next = False

    for ch in stripped:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            opener_stack.append("}")
        elif ch == "[":
            opener_stack.append("]")
        elif ch in ("}", "]"):
            if opener_stack and opener_stack[-1] == ch:
                opener_stack.pop()

    if not opener_stack:
        return None

    candidate = stripped.rstrip(",")
    candidate += "".join(reversed(opener_stack))

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    return None


def parse_tool_call_arguments(
    arguments: Optional[str],
    tool_name: Optional[str] = None,
    context: Optional[str] = None,
) -> Any:
    """JSON 文字列から tool call の引数を解析し、切り詰めを修復する。"""
    import json

    if not arguments or not arguments.strip():
        return {}

    try:
        return json.loads(arguments)
    except json.JSONDecodeError as original_error:
        repaired = _attempt_json_repair(arguments)
        if repaired is not None:
            verbose_logger.warning(
                "Repaired truncated tool call arguments for tool '%s' (%s).",
                tool_name or "<unknown>",
                context or "unknown context",
            )
            return repaired

        error_parts = ["Failed to parse tool call arguments"]
        if tool_name:
            error_parts.append(f"for tool '{tool_name}'")
        if context:
            error_parts.append(f"({context})")
        error_message = (
            " ".join(error_parts)
            + f". Error: {str(original_error)}. Arguments: {arguments}"
        )
        raise ValueError(error_message) from original_error
