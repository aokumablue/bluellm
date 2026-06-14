"""OpenAI の呼び出しを Anthropic の `/v1/messages` フォーマットに変換する"""

import json
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
)

from bluellm._compat import verbose_logger
from bluellm._compat import uuid
from bluellm.types.anthropic_context import (
    AppliedEdit,
    CompactionBlock,
    ContextManagementResponse,
    UsageIteration,
)
from bluellm.types.anthropic_streaming import (
    UsageDelta,
)
from bluellm._compat import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from bluellm._compat import ModelResponseStream


class BlueLLMStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - 最初のチャンクは 'message_start' を返す
    - content block は開始と終了が必要
    - finish_reason は Anthropic の理由コードに正確にマッピングする必要がある（そうしないと Anthropic クライアントがパースできない）
    """

    from bluellm.types.anthropic_streaming import (
        ContentBlockContentBlockDict,
        ContentBlockStart,
        ContentBlockStartText,
        TextBlock,
    )

    sent_first_chunk: bool
    sent_content_block_start: bool
    sent_content_block_finish: bool
    current_content_block_type: Literal["text", "tool_use", "thinking"]
    sent_last_message: bool
    holding_stop_reason_chunk: Optional[Any]
    queued_usage_chunk: bool
    current_content_block_index: int

    def __init__(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
        applied_edits: Optional[List[AppliedEdit]] = None,
        compaction_block: Optional[CompactionBlock] = None,
        iterations_usage: Optional[List[UsageIteration]] = None,
    ):
        """OpenAI の completion stream をラップし、ストリームごとの状態を初期化する。

        ``tool_name_mapping`` は切り詰められた tool 名を復元する。``applied_edits`` /
        ``compaction_block`` / ``iterations_usage`` は context-management の
        ポリフィルデータを発行イベントに付与する。
        """
        super().__init__(completion_stream)
        self.sent_first_chunk: bool = False
        self.sent_content_block_start: bool = False
        self.sent_content_block_finish: bool = False
        self.current_content_block_type: Literal["text", "tool_use", "thinking"] = (
            "text"
        )
        self.sent_last_message: bool = False
        self.holding_stop_reason_chunk: Optional[Any] = None
        self.queued_usage_chunk: bool = False
        self.current_content_block_index: int = 0
        self.model = model
        # 切り詰められた tool 名から元の名前へのマッピング（OpenAI の 64 文字制限対応）
        self.tool_name_mapping = tool_name_mapping or {}
        # 最終 message_delta に applied_edits を付与するポリフィル。
        self.applied_edits: List[AppliedEdit] = list(applied_edits or [])
        # compact_20260112 ポリフィル（ストリーミング）から生成された compaction block。
        self.compaction_block = compaction_block
        self.iterations_usage = iterations_usage
        self.sent_compaction_block: bool = False
        # compaction block の start/delta/stop イベントが呼び出し元に実際に
        # 消費されるタイミングと同期して（公開ステートマシンを進めながら）
        # 正確に発行されるようにするフェーズ別フラグ。3つのイベントを事前に
        # キューイングすると、クライアントが ``content_block_stop`` を受信する前に
        # ``sent_content_block_finish=True`` がセットされてしまい、ドレイン期間中に
        # 観測可能な状態が矛盾する。
        self.sent_compaction_block_start: bool = False
        self.sent_compaction_block_delta: bool = False
        # 複数のチャンクをバッファリングするためのインスタンスごとのキュー。
        # クラスレベルではなくここで初期化する必要がある。同一の deque を
        # 並行ストリームが共有すると SSE イベント順序が壊れる。
        self.chunk_queue: deque = deque()
        # インスタンスごとのデフォルト content block。クラスレベルではなくここで
        # 初期化する必要がある。`_should_start_new_content_block` は
        # `tool_block["name"]` をインプレースで書き換えるため、クラスレベルで
        # 共有するとストリーム間でデータが漏れる。
        self.current_content_block_start: (
            "BlueLLMStreamWrapper.ContentBlockContentBlockDict"
        ) = self.TextBlock(
            type="text",
            text="",
        )

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

    def _next_compaction_event(self) -> Optional[Dict[str, Any]]:
        """次の compaction content-block SSE イベントを返す。なければ ``None``。

        Anthropic はコンパクションを単一デルタ（トークンごとのストリーミングなし）
        で配信するが、適切な start → delta → stop の3イベントとして公開する。
        各呼び出しが正確に1イベントを返すことで、ステートマシン
        （``sent_content_block_finish``、``current_content_block_index``）は
        終端の stop イベントが実際に呼び出し元に返されたときのみ進められる。
        これにより、フラグがブロック完了を示している間に stop イベントがまだ
        バッファリングされているという観測可能なウィンドウを防ぐ。
        """
        if self.compaction_block is None or self.sent_compaction_block:
            return None

        compaction_index = self.current_content_block_index

        if not self.sent_compaction_block_start:
            self.sent_compaction_block_start = True
            return {
                "type": "content_block_start",
                "index": compaction_index,
                # テキストブロックの形状（{"type": "text", "text": ""}）をミラーリング:
                # ``content_block_start`` を検査するクライアントがフルブロックスキーマを
                # 確認できるよう空の ``content`` フィールドを送信する。
                # 実際のサマリーテキストは下記の ``content_block_delta`` で届く。
                "content_block": {"type": "compaction", "content": ""},
            }

        if not self.sent_compaction_block_delta:
            self.sent_compaction_block_delta = True
            summary_content = self.compaction_block.get("content") or ""
            return {
                "type": "content_block_delta",
                "index": compaction_index,
                "delta": {"type": "compaction_delta", "content": summary_content},
            }

        stop_event = {
            "type": "content_block_stop",
            "index": compaction_index,
        }
        # ここでは ``sent_content_block_finish`` に触れない: このフラグは通常の
        # text/tool_use/thinking ブロックのステートマシン用であり、
        # 合成 compaction ブロックのライフサイクルとは独立している。
        # 両者を混同すると、通常の content block が一度も開始されていないのに
        # ``sent_content_block_finish=True`` が外部の観察者（サブクラスのオーバーライド、
        # イントロスペクションフック、例外パス）に見えてしまう。
        self._increment_content_block_index()
        self.sent_compaction_block = True
        return stop_event

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

    def __next__(self):
        """次の Anthropic SSE イベント dict を返す（同期イテレーション）。

        プリアンブルを発行し、上流の各チャンクを処理し、保持チャンクをフラッシュする。
        上流のエラーはログに記録され、クリーンな停止に変換される。
        """
        try:
            pre = self._emit_preamble_chunk()
            if pre is not None:
                return pre
            for chunk in self.completion_stream:
                out = self._process_upstream_chunk(chunk)
                if out is not None:
                    return out
            end = self._flush_stream_end()
            if end is not None:
                return end
            raise StopIteration
        except StopIteration:
            # ループ終了時点で _flush_stream_end がバッファリングされた全チャンクを
            # ドレイン済みのため、ここでフラッシュするものは何もない。
            raise StopIteration
        except Exception as e:
            # type と status のみをログに記録する（str(e) はエンドポイント/ボディの
            # 詳細を含む可能性があるため除外）。完全なトレースバックは
            # exc_info=True を通じてリダクションフィルターに渡される
            # （下記の SSE ラッパーと同じ方式）。sync イテレーターは上流の
            # エラーをクリーンな停止に変換する。async パスはエラーを伝播させ、
            # SSE ラッパーがエラーイベントを発行できるようにする。
            verbose_logger.error(
                "Anthropic Adapter stream aborted: %s (status=%s)",
                type(e).__name__,
                getattr(e, "status_code", "N/A"),
                exc_info=True,
            )
            raise StopIteration

    async def __anext__(self):
        """次の Anthropic SSE イベント dict を返す（async イテレーション）。

        :meth:`__next__` と同じフローだが、上流のエラーは伝播させることで
        async_anthropic_sse_wrapper が終端エラーイベントを発行できるようにする。
        """
        try:
            pre = self._emit_preamble_chunk()
            if pre is not None:
                return pre
            async for chunk in self.completion_stream:
                out = self._process_upstream_chunk(chunk)
                if out is not None:
                    return out
            end = self._flush_stream_end()
            if end is not None:
                return end
            # StopIteration を発生させてローカルでキャッチする（コルーチンの外には
            # 伝播しないため PEP 479 は適用されない）、そして StopAsyncIteration に
            # 変換する。_flush_stream_end はキューをドレイン済み。
            raise StopIteration
        except StopIteration:
            raise StopAsyncIteration
        # 外側の `except Exception` なし: sync パスと異なり、上流エラーは伝播させることで
        # async_anthropic_sse_wrapper がエラーイベントを発行できるようにする。

    def _emit_preamble_chunk(self) -> Optional[Dict[str, Any]]:
        """次のプリストリームイベントを返す。プリアンブル完了後は None を返す。

        最初にキューイングされたチャンクをドレインし、次に（1回だけ）
        message_start、compaction イベント、初期 content_block_start を発行する。
        None は呼び出し元に上流ストリームの消費を開始するよう通知する。
        __next__ と __anext__ で共有される（ロジックは上流イテレーション非依存）。
        """
        if self.chunk_queue:
            return self.chunk_queue.popleft()

        if self.sent_first_chunk is False:
            self.sent_first_chunk = True
            self.chunk_queue.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_{}".format(uuid.uuid4()),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": self._create_initial_usage_delta(),
                    },
                }
            )
            return self.chunk_queue.popleft()

        if self.sent_compaction_block is False and self.compaction_block is not None:
            compaction_event = self._next_compaction_event()
            if compaction_event is not None:
                return compaction_event

        if self.sent_content_block_start is False:
            self.sent_content_block_start = True
            self.sent_content_block_finish = False
            self.chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": self.current_content_block_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            return self.chunk_queue.popleft()

        return None

    def _process_upstream_chunk(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """上流の1チャンクを変換し、次のイベントを返す。なければ None。

        None は「このチャンクにはイベントなし、次に進む」を意味する（usage が
        マージされた後のトレーリングチャンク破棄）。__next__ と __anext__ で
        共有される。両者の違いは for/async-for イテレーションのみ。
        ``None``/``"None"`` センチネルチャンクは例外を発生させる（元のループと同じ動作）。
        """
        from bluellm.translation import BlueLLMMessagesAdapter

        if chunk == "None" or chunk is None:
            raise ValueError("upstream stream yielded a None chunk")

        choices = getattr(chunk, "choices", None) or []
        has_choices = bool(choices)
        should_start_new_block = self._should_start_new_content_block(chunk)
        if should_start_new_block:
            self._increment_content_block_index()

        # applied_edits は finish_reason が設定された最終 message_delta にのみ
        # 付与すればよい。中間チャンクには不要。hold-and-merge パスでは
        # context_management はマージ済みチャンクに直接付与されるため、
        # 変換後の ``processed_chunk`` は破棄される。そのため applied_edits の
        # 付与をスキップし、不要な ``MessageBlockDelta`` のアロケーションを避ける。
        will_merge_into_held = (
            self.holding_stop_reason_chunk is not None
            and getattr(chunk, "usage", None) is not None
        )
        if not has_choices:
            return self._handle_no_choices_chunk(chunk, will_merge_into_held)
        is_final_chunk = choices[0].finish_reason is not None
        processed_chunk = BlueLLMMessagesAdapter().translate_streaming_openai_response_to_anthropic(
            response=chunk,
            current_content_block_index=self.current_content_block_index,
            applied_edits=(
                self.applied_edits
                if is_final_chunk and not will_merge_into_held
                else None
            ),
        )

        # usage チャンクかつ保持中の stop_reason チャンクがある場合
        if will_merge_into_held:
            return self._merge_usage_and_dequeue(chunk)

        if self.queued_usage_chunk:
            # usage はすでにマージ＆発行済み。以降のプロバイダーイベントは
            # Anthropic SSE 順序に違反する（最終 ``message_delta`` の後に
            # チャンクは続けられない）。None を返して暗黙的に破棄する
            # （呼び出し元は次の上流チャンクに進む）。
            return None

        if should_start_new_block and not self.sent_content_block_finish:
            return self._queue_content_block_transition(processed_chunk)

        if (
            processed_chunk["type"] == "message_delta"
            and self.sent_content_block_finish is False
        ):
            return self._queue_message_delta_with_block_stop(processed_chunk)
        if (  # pragma: no cover - message_delta は常に上記の分岐で処理される
            processed_chunk.get("type") == "message_delta"
        ):
            processed_chunk = self._augment_message_delta_usage(processed_chunk)
        self.chunk_queue.append(processed_chunk)
        return self.chunk_queue.popleft()

    def _merge_usage_and_dequeue(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """usage チャンクを保持中の stop_reason チャンクへマージしてキューを進める。

        マージ済みチャンクをキューに積み、``queued_usage_chunk`` を立て、
        保持中チャンクをクリアしてから先頭イベントを返す。
        """
        merged_chunk = self._merge_usage_into_held_stop_reason_chunk(chunk)
        self.chunk_queue.append(merged_chunk)
        self.queued_usage_chunk = True
        self.holding_stop_reason_chunk = None
        return self.chunk_queue.popleft()

    def _handle_no_choices_chunk(
        self, chunk: Any, will_merge_into_held: bool
    ) -> Optional[Dict[str, Any]]:
        """choices を持たないチャンク（usage 専用/空チャンク）を処理する。

        保持中の stop_reason チャンクがあれば usage をマージして発行し、
        それ以外（usage 発行済み、または寄与しない空チャンク）は None を返して
        破棄する。
        """
        if will_merge_into_held:
            return self._merge_usage_and_dequeue(chunk)
        if self.queued_usage_chunk:
            # usage はすでにマージ＆発行済み。以降の空チャンクは
            # Anthropic SSE 順序に寄与しないため破棄する。
            return None
        return None

    def _queue_content_block_transition(
        self, processed_chunk: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """新しい content block への遷移シーケンスをキューイングする。

        ``content_block_stop`` -> ``content_block_start`` をキューに積む。
        トリガーチャンクが tool 引数（input_json_delta）を運ぶ場合は、
        暗黙的に破棄されないようそのデルタもキューに積む。先頭イベントを返す。
        """
        # シーケンスをキューイング: content_block_stop -> content_block_start
        # テキストブロックの場合、トリガーチャンクは別のデルタとして発行しない。
        # content_block_start がその情報を運ぶため。
        # tool_use ブロックの場合、トリガーチャンクが input_json_delta データを
        # 持つ場合はデルタも発行しなければならない。一部のプロバイダー
        # （xAI、Gemini など）は関数名/id と同じストリーミングチャンクに
        # tool 引数を含めるため。

        # 1. 現在の content block を停止
        self.chunk_queue.append(
            {
                "type": "content_block_stop",
                "index": max(self.current_content_block_index - 1, 0),
            }
        )

        # 2. 新しい content block を開始
        self.chunk_queue.append(
            {
                "type": "content_block_start",
                "index": self.current_content_block_index,
                "content_block": self.current_content_block_start,
            }
        )

        # 3. トリガーチャンクが tool 引数データを持つ場合、
        # input_json_delta が暗黙的に破棄されないようキューイングする。
        if (
            processed_chunk.get("type") == "content_block_delta"
            and isinstance(processed_chunk.get("delta"), dict)
            and processed_chunk["delta"].get("type") == "input_json_delta"
            and processed_chunk["delta"].get("partial_json")
        ):
            self.chunk_queue.append(processed_chunk)

        self.sent_content_block_finish = False
        return self.chunk_queue.popleft()

    def _queue_message_delta_with_block_stop(
        self, processed_chunk: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """まだ閉じていないブロックの ``content_block_stop`` と message_delta を処理する。

        先に ``content_block_stop`` をキューに積む。stop_reason を持つ
        message_delta は usage マージのため保持し、持たない場合は usage を
        付与してキューに積む。先頭イベントを返す。
        """
        # content_block_stop と message_delta の両方をキューイング
        self.chunk_queue.append(
            {
                "type": "content_block_stop",
                "index": self.current_content_block_index,
            }
        )
        self.sent_content_block_finish = True
        if processed_chunk.get("delta", {}).get("stop_reason") is not None:
            self.holding_stop_reason_chunk = processed_chunk
        else:  # pragma: no cover - message_delta は常に stop_reason を持つ
            processed_chunk = self._augment_message_delta_usage(processed_chunk)
            self.chunk_queue.append(processed_chunk)
        return self.chunk_queue.popleft()

    def _flush_stream_end(self) -> Optional[Dict[str, Any]]:
        """上流ストリームが正常終了した後、保持中のチャンクをフラッシュする。

        次のイベントを返す。何も残っていなければ None。保持中の ``message_delta``
        （finish_reason は確認済みだが usage はまだ）はフラッシュされ、その前に
        Anthropic SSE 順序を保つため ``content_block_stop`` が発行される。
        usage がマージ済みの場合は代わりに破棄される。__next__ と __anext__ で共有。
        """
        if not self.queued_usage_chunk and self.holding_stop_reason_chunk is not None:
            # 最終 ``message_delta`` の前には ``content_block_stop`` が必要。
            # 有効な Anthropic 順序の SSE を維持するため、アクティブブロックが
            # 閉じられていなければここで ``content_block_stop`` を発行する。
            if not self.sent_content_block_finish:  # pragma: no cover - 保持中の stop_reason は常に送信済みの content_block_stop と共存する
                self.chunk_queue.append(
                    {
                        "type": "content_block_stop",
                        "index": self.current_content_block_index,
                    }
                )
                self.sent_content_block_finish = True
            self.chunk_queue.append(
                self._augment_message_delta_usage(self.holding_stop_reason_chunk)
            )
            self.holding_stop_reason_chunk = None

        if not self.sent_last_message:
            self.sent_last_message = True
            self.chunk_queue.append({"type": "message_stop"})

        if self.chunk_queue:
            return self.chunk_queue.popleft()
        return None

    @staticmethod
    def _sse_error_event() -> bytes:
        """ストリーム中断時のサニタイズ済み Anthropic ``error`` SSE イベント。

        上流の詳細（エンドポイント/キーが含まれる可能性あり）は含まない。
        完全な原因はサーバー側でログに記録される。これを発行することで、
        クライアントはサイレントに切り詰められたストリームの代わりに
        終端エラーイベントを受信できる。
        """
        err = {
            "type": "error",
            "error": {"type": "api_error", "message": "upstream stream error"},
        }
        return f"event: error\ndata: {json.dumps(err)}\n\n".encode()

    def anthropic_sse_wrapper(self) -> Iterator[bytes]:
        """
        BlueLLMStreamWrapper の dict チャンクを Server-Sent Events フォーマットに変換する。
        Bedrock の bedrock_sse_wrapper 実装と同様。

        このラッパーは dict チャンクが event 行と data 行の両方を持つ SSE フォーマットで
        出力されることを保証する。
        """
        try:
            for chunk in self:
                if isinstance(chunk, dict):
                    event_type: str = str(chunk.get("type", "message"))
                    payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                    yield payload.encode()
                else:  # pragma: no cover - __next__ は常に dict イベントのみを yield する
                    yield chunk
        except Exception as e:  # pragma: no cover - sync の __next__ は上流エラーをクリーンな停止に変換するため、ここには何も伝播しない
            # async ラッパーのガードをミラーリング。対称性/堅牢性のために保持。
            verbose_logger.error(
                "Anthropic SSE stream aborted: %s (status=%s)",
                type(e).__name__,
                getattr(e, "status_code", "N/A"),
                exc_info=True,
            )
            yield self._sse_error_event()

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        anthropic_sse_wrapper の async 版。
        BlueLLMStreamWrapper の dict チャンクを Server-Sent Events フォーマットに変換する。
        """
        try:
            async for chunk in self:
                if isinstance(chunk, dict):
                    event_type: str = str(chunk.get("type", "message"))
                    payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                    yield payload.encode()
                else:  # pragma: no cover - __anext__ は常に dict イベントのみを yield する
                    yield chunk
        except Exception as e:
            # type と status のみをログに記録する（str(e) はエンドポイント/ボディの
            # 詳細を含む可能性があるため除外）。完全なトレースバックは
            # exc_info=True を通じてリダクションフィルターに渡される。
            verbose_logger.error(
                "Anthropic SSE stream aborted: %s (status=%s)",
                type(e).__name__,
                getattr(e, "status_code", "N/A"),
                exc_info=True,
            )
            yield self._sse_error_event()

    def _increment_content_block_index(self):
        """次の Anthropic content block index に進める。"""
        self.current_content_block_index += 1

    def _should_start_new_content_block(self, chunk: "ModelResponseStream") -> bool:
        """
        処理済みチャンクに基づいて新しい content block を開始すべきか判定する。
        新しい content block を検出するための具体的なロジックでこのメソッドをオーバーライドする。

        新しい content block を開始したい場合の例:
        - テキストから tool 呼び出しへの切り替え
        - レスポンス内の異なる content type
        - コンテンツ内の特定のマーカー
        """
        from bluellm.translation import BlueLLMMessagesAdapter

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return False

        # ロジック例 - 必要に応じてカスタマイズ:
        # チャンクが tool 呼び出しを示す場合
        if choices[0].finish_reason is not None:
            return False

        (
            block_type,
            content_block_start,
        ) = BlueLLMMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices  # type: ignore
        )

        # OpenAI の 64 文字制限により切り詰められた場合、元の tool 名を復元する
        if block_type == "tool_use":
            # 型の絞り込み: block_type が "tool_use" の場合、content_block_start は ToolUseBlock
            from typing import cast

            from bluellm.types.anthropic_streaming import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)

            if tool_block.get("name"):
                truncated_name = tool_block["name"]
                original_name = self.tool_name_mapping.get(
                    truncated_name, truncated_name
                )
                tool_block["name"] = original_name

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        # 並列 tool 呼び出しの場合、関数名を受信したときに必ず新しい content block が必要。
        # 関数名は新しい tool 呼び出しを示すシグナルとなる。
        if block_type == "tool_use":
            tool_block = cast(ToolUseBlock, content_block_start)
            if tool_block.get("name"):
                self.current_content_block_type = block_type
                self.current_content_block_start = content_block_start
                return True

        return False
