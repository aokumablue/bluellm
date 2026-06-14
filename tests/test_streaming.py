"""Coverage for the streaming hold/merge/compaction helpers and edge paths."""

import asyncio
from types import SimpleNamespace

from helpers import stream_chunk, usage, usage_only_chunk

from bluellm.providers.openai_like import OpenAILikeProvider
from bluellm.streaming import BlueLLMStreamWrapper


def _wrap(**kw):
    kw.setdefault("tool_name_mapping", {})
    return BlueLLMStreamWrapper(completion_stream=iter([]), model="m", **kw)


def _async_collect(chunks, **wrap_kw):
    async def agen():
        for c in chunks:
            yield c

    async def run():
        shaped = OpenAILikeProvider._shape_stream(agen())
        wrapper = BlueLLMStreamWrapper(
            completion_stream=shaped, model="m", tool_name_mapping={}, **wrap_kw
        )
        out = b""
        async for sse in wrapper.async_anthropic_sse_wrapper():
            out += sse if isinstance(sse, (bytes, bytearray)) else sse.encode()
        return out.decode()

    return asyncio.run(run())


def _async_collect_raw(chunks, **wrap_kw):
    async def agen():
        for c in chunks:
            yield c

    async def run():
        wrapper = BlueLLMStreamWrapper(
            completion_stream=agen(), model="m", tool_name_mapping={}, **wrap_kw
        )
        out = b""
        async for sse in wrapper.async_anthropic_sse_wrapper():
            out += sse if isinstance(sse, (bytes, bytearray)) else sse.encode()
        return out.decode()

    return asyncio.run(run())


# ---- _merge_usage_into_held_stop_reason_chunk ----
def test_merge_usage_with_cache_tokens_and_context_management():
    w = _wrap(applied_edits=[{"type": "clear_tool_uses_20250919"}])
    w.holding_stop_reason_chunk = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
    }
    u = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=3,
        prompt_tokens_details=SimpleNamespace(cached_tokens=4),
    )
    u._cache_creation_input_tokens = 2
    u._cache_read_input_tokens = 4
    merged = w._merge_usage_into_held_stop_reason_chunk(SimpleNamespace(usage=u))
    assert merged["usage"]["input_tokens"] == 6
    assert merged["usage"]["cache_creation_input_tokens"] == 2
    assert merged["usage"]["cache_read_input_tokens"] == 4
    assert "context_management" in merged


def test_merge_usage_without_delta_key():
    w = _wrap()
    w.holding_stop_reason_chunk = {"type": "message_delta"}  # no "delta"
    u = SimpleNamespace(prompt_tokens=5, completion_tokens=1, prompt_tokens_details=None)
    merged = w._merge_usage_into_held_stop_reason_chunk(SimpleNamespace(usage=u))
    assert merged["delta"] == {}
    assert merged["usage"]["input_tokens"] == 5


def test_merge_usage_clamps_negative_input_tokens():
    """cached_tokens > prompt_tokens でも input_tokens が負にならず 0 にクランプされる。"""
    w = _wrap()
    w.holding_stop_reason_chunk = {"type": "message_delta", "delta": {}}
    u = SimpleNamespace(
        prompt_tokens=5,
        completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cached_tokens=10),
    )
    merged = w._merge_usage_into_held_stop_reason_chunk(SimpleNamespace(usage=u))
    assert merged["usage"]["input_tokens"] == 0
    assert merged["usage"]["output_tokens"] == 10


def test_merge_usage_clamps_negative_output_tokens():
    """負の completion_tokens でも output_tokens が負にならず 0 にクランプされる。"""
    w = _wrap()
    w.holding_stop_reason_chunk = {"type": "message_delta", "delta": {}}
    u = SimpleNamespace(prompt_tokens=10, completion_tokens=-5, prompt_tokens_details=None)
    merged = w._merge_usage_into_held_stop_reason_chunk(SimpleNamespace(usage=u))
    assert merged["usage"]["input_tokens"] == 10
    assert merged["usage"]["output_tokens"] == 0


# ---- _augment_message_delta_usage ----
def test_augment_adds_iterations():
    w = _wrap(iterations_usage=[{"type": "message", "input_tokens": 1, "output_tokens": 1}])
    out = w._augment_message_delta_usage(
        {"type": "message_delta", "usage": {"input_tokens": 5, "output_tokens": 2}}
    )
    assert out["usage"]["iterations"][-1] == {
        "type": "message",
        "input_tokens": 5,
        "output_tokens": 2,
    }


def test_augment_skips_zero_token_iteration():
    w = _wrap(iterations_usage=[{"type": "message", "input_tokens": 1, "output_tokens": 1}])
    out = w._augment_message_delta_usage(
        {"type": "message_delta", "usage": {"input_tokens": 0, "output_tokens": 0}}
    )
    # the seed iteration stays, but no zero-token message iteration is appended
    assert len(out["usage"]["iterations"]) == 1


def test_augment_attaches_context_management():
    w = _wrap(applied_edits=[{"type": "clear_tool_uses_20250919"}])
    out = w._augment_message_delta_usage(
        {"type": "message_delta", "usage": {"input_tokens": 1, "output_tokens": 1}}
    )
    assert "context_management" in out


def test_augment_no_iterations_usage_is_noop():
    w = _wrap()
    chunk = {"type": "message_delta", "usage": {"input_tokens": 1, "output_tokens": 1}}
    assert w._augment_message_delta_usage(chunk) == chunk


def test_augment_non_dict_usage_returns_unchanged():
    w = _wrap(iterations_usage=[{"type": "message", "input_tokens": 1, "output_tokens": 1}])
    chunk = {"type": "message_delta"}  # no usage
    assert w._augment_message_delta_usage(chunk) == chunk


# ---- _next_compaction_event ----
def test_next_compaction_event_sequence():
    w = _wrap(compaction_block={"content": "summary text"})
    e1 = w._next_compaction_event()
    e2 = w._next_compaction_event()
    e3 = w._next_compaction_event()
    assert e1["type"] == "content_block_start"
    assert e2["delta"]["content"] == "summary text"
    assert e3["type"] == "content_block_stop"
    assert w._next_compaction_event() is None  # exhausted


def test_next_compaction_event_none_without_block():
    assert _wrap()._next_compaction_event() is None


def test_create_initial_usage_delta():
    assert _wrap()._create_initial_usage_delta()["input_tokens"] == 0


# ---- integration: compaction block in the live stream ----
def test_stream_emits_compaction_events():
    body = _async_collect(
        [stream_chunk(content="hi"), stream_chunk(finish_reason="stop"), usage_only_chunk(usage(3, 1))],
        compaction_block={"content": "compacted"},
    )
    assert "compaction" in body


# ---- integration: usage merge then trailing chunk dropped ----
def test_stream_drops_trailing_chunk_after_usage_merge():
    body = _async_collect(
        [
            stream_chunk(content="hi"),
            stream_chunk(finish_reason="stop"),
            usage_only_chunk(usage(8, 2)),
            stream_chunk(content="trailing should be dropped"),
        ]
    )
    assert "trailing" not in body
    assert body.count("event: message_stop") == 1


def test_stream_raw_empty_choices_usage_only_does_not_crash():
    body = _async_collect_raw(
        [
            stream_chunk(content="hi"),
            stream_chunk(finish_reason="stop"),
            usage_only_chunk(usage(8, 2)),
        ]
    )
    assert "event: message_stop" in body
    assert '"input_tokens": 8' in body
    assert '"output_tokens": 2' in body


def test_stream_empty_chunk_after_usage_merge_is_dropped():
    # streaming.py:430,433 — a no-choices/no-usage chunk arriving after the usage
    # has already been merged & queued contributes nothing to the Anthropic SSE
    # order and is dropped (queued_usage_chunk True branch).
    body = _async_collect_raw(
        [
            stream_chunk(content="hi"),
            stream_chunk(finish_reason="stop"),
            usage_only_chunk(usage(8, 2)),
            SimpleNamespace(id="c1", model="gpt-5.4", choices=[], usage=None),
        ]
    )
    assert body.count("event: message_stop") == 1
    assert '"input_tokens": 8' in body


def test_stream_empty_chunk_without_pending_usage_is_dropped():
    # streaming.py:434 — a no-choices/no-usage chunk with nothing held and no
    # queued usage yields no event and is skipped without affecting the stream.
    body = _async_collect_raw(
        [
            SimpleNamespace(id="c1", model="gpt-5.4", choices=[], usage=None),
            stream_chunk(content="hi"),
            stream_chunk(finish_reason="stop"),
            usage_only_chunk(usage(8, 2)),
        ]
    )
    assert body.count("event: message_stop") == 1
    assert "hi" in body


def test_should_start_new_block_empty_choices_returns_false():
    assert BlueLLMStreamWrapper(
        completion_stream=iter([]), model="m", tool_name_mapping={}
    )._should_start_new_content_block(usage_only_chunk(usage(1, 1))) is False


# ---- _should_start_new_content_block: parallel tool calls + name restoration ----
def _tool_chunk(name):
    from bluellm._compat import StreamingChoices

    delta = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="c", function=SimpleNamespace(name=name, arguments=None))],
        role=None,
        function_call=None,
    )
    choice = StreamingChoices()
    choice.delta = delta
    choice.finish_reason = None
    return SimpleNamespace(choices=[choice])


def test_should_start_new_block_parallel_tools_and_name_restore():
    w = _wrap(tool_name_mapping={"sh": "LongToolName"})
    # First tool chunk: text -> tool_use switch (also restores the truncated name).
    assert w._should_start_new_content_block(_tool_chunk("sh")) is True
    assert w.current_content_block_start["name"] == "LongToolName"
    # Second tool chunk while already in a tool_use block: a new parallel call.
    assert w._should_start_new_content_block(_tool_chunk("sh")) is True


# ---- integration: None sentinel chunk -> async error event ----
def test_stream_none_chunk_emits_error_event():
    async def agen():
        yield stream_chunk(content="hi")
        yield None

    async def run():
        wrapper = BlueLLMStreamWrapper(
            completion_stream=agen(), model="m", tool_name_mapping={}
        )
        out = b""
        async for sse in wrapper.async_anthropic_sse_wrapper():
            out += sse if isinstance(sse, (bytes, bytearray)) else sse.encode()
        return out.decode()

    body = asyncio.run(run())
    assert "event: error" in body
