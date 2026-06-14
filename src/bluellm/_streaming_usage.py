"""ストリーミング経路の usage ペイロード構築・付与ロジックを集約する Mixin。

``BlueLLMStreamWrapper`` のステートマシン本体（チャンク処理・content block
遷移・SSE 発行）から、usage dict の構築と ``context_management`` /
compaction iteration usage の付与という凝集した関心事を切り出したもの。

各メソッドは ``self.holding_stop_reason_chunk`` / ``self.applied_edits`` /
``self.iterations_usage`` を参照する。これらは合成先の
``BlueLLMStreamWrapper.__init__`` で初期化される。
"""

from typing import Any, Dict, List

from bluellm.types.anthropic_context import (
    ContextManagementResponse,
    UsageIteration,
)
from bluellm.types.anthropic_streaming import (
    UsageDelta,
)


class _StreamingUsageMixin:
    """usage dict の構築・クランプと message_delta への付与ロジックを提供する Mixin。"""

    def _merge_usage_into_held_stop_reason_chunk(self, chunk: Any) -> Dict[str, Any]:
        """``chunk`` の usage データを保持中の ``message_delta`` チャンクにマージする。

        sync の ``__next__`` と async の ``__anext__`` の両パスで共有されるため、
        微妙な保持＆マージロジック（キャッシュトークン、``context_management``
        の付与、``UsageDelta`` の形状）が一箇所に集約される。

        呼び出し元は ``self.holding_stop_reason_chunk`` と
        ``self.queued_usage_chunk`` の状態管理、および返されたマージ済みチャンクの
        キューイングを担当する。
        """
        if self.holding_stop_reason_chunk is None:  # pragma: no cover - 呼び出し元の will_merge_into_held チェックにより保証済み
            raise RuntimeError("_merge_usage called without a held stop_reason chunk")
        merged_chunk = self.holding_stop_reason_chunk.copy()
        if "delta" not in merged_chunk:
            merged_chunk["delta"] = {}

        uncached_input_tokens = chunk.usage.prompt_tokens or 0
        if (
            hasattr(chunk.usage, "prompt_tokens_details")
            and chunk.usage.prompt_tokens_details
        ):
            cached_tokens = (
                getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0) or 0
            )
            uncached_input_tokens -= cached_tokens

        # 非ストリーミング経路（translation/_response.py の _build_usage_dict）と
        # 同様に、cached_tokens > prompt_tokens や負の completion_tokens による
        # 負数 usage が SSE でクライアントへ流出するのを防ぐためクランプする。
        usage_dict: UsageDelta = {
            "input_tokens": max(0, uncached_input_tokens),
            "output_tokens": max(0, chunk.usage.completion_tokens or 0),
        }
        if (
            hasattr(chunk.usage, "_cache_creation_input_tokens")
            and chunk.usage._cache_creation_input_tokens > 0
        ):
            usage_dict["cache_creation_input_tokens"] = (
                chunk.usage._cache_creation_input_tokens
            )
        if (
            hasattr(chunk.usage, "_cache_read_input_tokens")
            and chunk.usage._cache_read_input_tokens > 0
        ):
            usage_dict["cache_read_input_tokens"] = chunk.usage._cache_read_input_tokens
        merged_chunk["usage"] = usage_dict
        if self.applied_edits and "context_management" not in merged_chunk:
            merged_chunk["context_management"] = ContextManagementResponse(
                applied_edits=list(self.applied_edits)
            )
        return self._augment_message_delta_usage(merged_chunk)

    def _ensure_context_management_attached(
        self, message_delta_chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """``self.applied_edits`` が空でなく、チャンクがまだ ``context_management`` を
        持っていない場合、``message_delta`` チャンクに ``context_management`` を付与する。
        （場合によっては新しい）チャンク dict を返す。

        このガードを一箇所に集約することで、すべての ``message_delta`` 発行パス
        （usage とのマージ、保持中チャンクの直接フラッシュ）が一貫して
        ``applied_edits`` をクライアントに届けることが保証される。
        """
        if not self.applied_edits or "context_management" in message_delta_chunk:
            return message_delta_chunk
        augmented = message_delta_chunk.copy()
        augmented["context_management"] = ContextManagementResponse(
            applied_edits=list(self.applied_edits)
        )
        return augmented

    def _augment_message_delta_usage(
        self, message_delta_chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ポリフィルの compaction iteration usage を最終 message_delta に付与する。

        また、``self.applied_edits`` が空でない場合に直接の保持チャンク
        フラッシュパスがマージパスの保証と同期するよう、
        ``context_management`` を防御的に再付与する。
        """
        message_delta_chunk = self._ensure_context_management_attached(
            message_delta_chunk
        )
        if self.iterations_usage is None:
            return message_delta_chunk
        usage = message_delta_chunk.get("usage")
        if not isinstance(usage, dict) or "iterations" in usage:
            return message_delta_chunk

        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        augmented = message_delta_chunk.copy()
        augmented_usage = dict(usage)
        iterations: List[UsageIteration] = list(self.iterations_usage)
        # ``message`` iteration は実際のトークンデータがある場合にのみ発行する。
        # usage チャンクが別途存在しない場合（例：プロバイダーが finish_reason だけを送信した場合）、
        # 保持中の ``message_delta`` には translate ステップからのプレースホルダー 0 が含まれる。
        # ゼロトークンの iteration を報告するのは誤解を招き、非ストリーミングパスとの
        # 一貫性も損なうため行わない。
        if input_tokens > 0 or output_tokens > 0:
            message_iteration: UsageIteration = {
                "type": "message",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            iterations.append(message_iteration)
        augmented_usage["iterations"] = iterations  # type: ignore[typeddict-unknown-key]
        augmented["usage"] = augmented_usage
        return augmented

    def _create_initial_usage_delta(self) -> UsageDelta:
        """
        message_start イベント用の初期 UsageDelta を作成する。

        prompt caching がサポートされていることをクライアント（Claude Code など）に
        示すため、キャッシュトークンフィールド（cache_creation_input_tokens、
        cache_read_input_tokens）を 0 で初期化する。

        実際のキャッシュトークン値はストリーム末尾の message_delta イベントで
        提供される。Bedrock Converse API は usage データを最終レスポンスチャンクに
        のみ返すため。

        Returns:
            すべてのトークン数が 0 で初期化された UsageDelta。
        """
        return UsageDelta(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
