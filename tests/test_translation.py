import asyncio
import re

import pytest
from helpers import stream_chunk, text_completion, tool_call, usage, usage_only_chunk

from bluellm.providers.openai_like import OpenAILikeProvider
from bluellm.streaming import BlueLLMStreamWrapper
from bluellm.translation import BlueLLMMessagesAdapter
from bluellm.translation._errors import UnsupportedContentError


def test_input_translation_roles_tools_toolchoice():
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 1024,
        "system": "be helpful",
        "messages": [
            {"role": "user", "content": "read foo.py"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"path": "foo.py"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "x=1"}],
            },
        ],
        "tools": [
            {"name": "Read", "description": "d", "input_schema": {"type": "object", "properties": {}}}
        ],
        "tool_choice": {"type": "auto"},
    }
    req, mapping = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert req["tool_choice"] == "auto"
    assistant = req["messages"][2]
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert mapping == {}


def test_output_tool_use_block():
    resp = text_completion(finish_reason="tool_calls", tool_calls=[tool_call()])
    out = BlueLLMMessagesAdapter().translate_completion_output_params(
        response=resp, tool_name_mapping={}
    )
    assert out["stop_reason"] == "tool_use"
    block = next(b for b in out["content"] if b["type"] == "tool_use")
    assert block["name"] == "Read"
    assert block["input"] == {"path": "foo.py"}


def test_output_text_and_usage():
    out = BlueLLMMessagesAdapter().translate_completion_output_params(
        response=text_completion(text="hi"), tool_name_mapping={}
    )
    assert out["content"][0] == {"type": "text", "text": "hi"}
    assert out["usage"] == {"input_tokens": 12, "output_tokens": 7}
    assert out["stop_reason"] == "end_turn"


def test_content_filter_maps_to_refusal():
    # H1: Azure frequently stops on content_filter. Mapping it to end_turn made
    # Claude Code treat a filtered stop as a normal completion; it must map to
    # the Anthropic `refusal` stop_reason instead.
    out = BlueLLMMessagesAdapter().translate_completion_output_params(
        response=text_completion(text="", finish_reason="content_filter"),
        tool_name_mapping={},
    )
    assert out["stop_reason"] == "refusal"


def test_base64_image_still_translates():
    # Don't break: a valid base64 image source must still become an OpenAI
    # image_url data URI (success path).
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    }
                ],
            }
        ],
    }
    req, _ = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    user_msg = next(m for m in req["messages"] if m["role"] == "user")
    img = user_msg["content"][0]
    assert img["type"] == "image_url"
    assert img["image_url"]["url"] == "data:image/png;base64,aGVsbG8="


def test_image_source_translation_branches():
    # Cover every branch of the image-source translator: the url success path
    # and each unrepresentable source (non-object, empty base64, empty url).
    adapter = BlueLLMMessagesAdapter()
    assert (
        adapter._translate_anthropic_image_to_openai(
            {"type": "url", "url": "https://img/x.png"}
        )
        == "https://img/x.png"
    )
    for bad in (
        "not-a-dict",
        {"type": "base64", "data": ""},
        {"type": "url", "url": ""},
    ):
        with pytest.raises(UnsupportedContentError):
            adapter._translate_anthropic_image_to_openai(bad)


def test_tool_result_is_error_with_empty_content():
    # M9: a failing tool_result with no content body still gets the marker
    # (string prefix with no separator).
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "is_error": True}
                ],
            }
        ],
    }
    req, _ = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    tool_msg = next(m for m in req["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == "[tool_result is_error=true]"


def test_unsupported_image_source_raises_explicit_error():
    # M8/L5: an image block whose source the proxy cannot represent (e.g. a
    # `file`/file_id reference) used to be silently dropped, so the model never
    # saw the image and answered as if it were absent. It must instead raise an
    # explicit error the server can turn into a 400.
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "file", "file_id": "file_123"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(UnsupportedContentError):
        BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(dict(body))


def test_unsupported_document_source_raises_explicit_error():
    # M8/L5: same for a document block (PDF etc.) with an untranslatable source.
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": "file_456"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(UnsupportedContentError):
        BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(dict(body))


def test_tool_result_without_is_error_is_unchanged():
    # Don't break: a normal (successful) tool_result keeps plain string content.
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}
                ],
            }
        ],
    }
    req, _ = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    tool_msg = next(m for m in req["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == "ok"


def test_tool_result_is_error_marks_string_content():
    # M9: a tool_result with is_error=true must signal failure to the model.
    # OpenAI tool messages have no is_error field, so the content is prefixed
    # with an explicit error marker rather than being indistinguishable from a
    # successful result.
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "is_error": True,
                        "content": "boom",
                    }
                ],
            }
        ],
    }
    req, _ = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    tool_msg = next(m for m in req["messages"] if m["role"] == "tool")
    assert tool_msg["content"].startswith("[tool_result is_error=true]")
    assert "boom" in tool_msg["content"]


def test_is_error_marker_scoped_to_its_own_tool_result():
    # M9 invariant: with two tool_results in one turn where only the second is
    # an error, the marker must decorate ONLY the second's message (the
    # tool_result_start snapshot must not leak across tool_results).
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "ok", "content": "fine"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "bad",
                        "is_error": True,
                        "content": "boom",
                    },
                ],
            }
        ],
    }
    req, _ = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    tools = [m for m in req["messages"] if m["role"] == "tool"]
    ok_msg = next(m for m in tools if m["tool_call_id"] == "ok")
    bad_msg = next(m for m in tools if m["tool_call_id"] == "bad")
    assert ok_msg["content"] == "fine"  # untouched
    assert bad_msg["content"].startswith("[tool_result is_error=true]")


def test_tool_result_is_error_marks_list_content():
    # M9: list-shaped tool_result content gets a leading error text part so the
    # signal survives without corrupting the existing parts.
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "is_error": True,
                        "content": [
                            {"type": "text", "text": "line one"},
                            {"type": "text", "text": "line two"},
                        ],
                    }
                ],
            }
        ],
    }
    req, _ = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    tool_msg = next(m for m in req["messages"] if m["role"] == "tool")
    parts = tool_msg["content"]
    assert isinstance(parts, list)
    assert parts[0]["text"] == "[tool_result is_error=true]"
    assert parts[1]["text"] == "line one"


def test_empty_content_yields_empty_text_block():
    # M11: when the upstream returns no content and no tool calls (e.g. a
    # content_filter refusal), the Anthropic response must still carry a
    # non-empty content array; emit an empty text block rather than [].
    out = BlueLLMMessagesAdapter().translate_completion_output_params(
        response=text_completion(text=None, finish_reason="content_filter"),
        tool_name_mapping={},
    )
    assert out["content"] == [{"type": "text", "text": ""}]


def test_streaming_sse_order_and_usage_merge():
    async def fake_stream():
        yield stream_chunk(content="Hel")
        yield stream_chunk(content="lo")
        yield stream_chunk(finish_reason="stop")
        yield usage_only_chunk(usage(10, 3))  # include_usage final chunk (empty choices)

    async def run():
        shaped = OpenAILikeProvider._shape_stream(fake_stream())
        wrapper = BlueLLMStreamWrapper(
            completion_stream=shaped, model="gpt-5.4", tool_name_mapping={}
        )
        out = b""
        async for sse in wrapper.async_anthropic_sse_wrapper():
            out += sse if isinstance(sse, (bytes, bytearray)) else sse.encode()
        return out.decode()

    body = asyncio.run(run())
    events = [
        line.split("event: ")[1]
        for line in body.splitlines()
        if line.startswith("event: ")
    ]
    assert events == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert '"input_tokens": 10' in body and '"output_tokens": 3' in body


def test_streaming_thinking_block_nonstr_raises_typeerror():
    # M5: the production isinstance asserts (stripped under -O) are now explicit
    # TypeErrors. A streaming thinking block whose 'thinking' is not a string
    # must raise TypeError, not silently pass or AssertionError.
    from types import SimpleNamespace

    from bluellm._compat import StreamingChoices

    choice = StreamingChoices()
    choice.delta = SimpleNamespace(
        tool_calls=None,
        content=None,
        thinking_blocks=[{"type": "thinking", "thinking": 123, "signature": ""}],
    )
    with pytest.raises(TypeError):
        BlueLLMMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=[choice]
        )


def _translate(messages, **extra):
    body = {"model": "claude-sonnet-4", "max_tokens": 16, "messages": messages, **extra}
    req, mapping = BlueLLMMessagesAdapter().translate_completion_input_params_with_tool_mapping(
        dict(body)
    )
    return req, mapping


# ---------------------------------------------------------------------------
# Characterization tests for translate_anthropic_messages_to_openai (H5): lock
# the current behavior of every user/assistant/tool_result branch so the split
# into _translate_user_message/_translate_tool_result/_translate_assistant_message
# cannot change observable output.
# ---------------------------------------------------------------------------


def test_charz_user_mixed_content():
    # text + url image + base64 document -> three OpenAI content parts.
    req, _ = _translate(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "Zm9v",
                        },
                    },
                ],
            }
        ]
    )
    user = next(m for m in req["messages"] if m["role"] == "user")
    assert [c["type"] for c in user["content"]] == ["text", "image_url", "image_url"]
    assert user["content"][1]["image_url"]["url"] == "https://x/y.png"
    assert user["content"][2]["image_url"]["url"] == "data:application/pdf;base64,Zm9v"


def test_charz_tool_result_list_variants():
    # tool_result list content: single text, single image, and multi-item.
    req, _ = _translate(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "a",
                        "content": [{"type": "text", "text": "one"}],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "b",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://i/i.png"},
                            }
                        ],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "content": [
                            {"type": "text", "text": "x"},
                            {"type": "text", "text": "y"},
                        ],
                    },
                ],
            }
        ]
    )
    tools = [m for m in req["messages"] if m["role"] == "tool"]
    assert tools[0]["content"] == "one"
    assert tools[1]["content"] == "https://i/i.png"
    assert isinstance(tools[2]["content"], list)
    assert [p["text"] for p in tools[2]["content"]] == ["x", "y"]


def test_charz_assistant_text_tool_use_thinking():
    # assistant with text + thinking + redacted_thinking + tool_use.
    req, _ = _translate(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "xx"},
                    {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"path": "f"}},
                ],
            }
        ]
    )
    asst = next(m for m in req["messages"] if m["role"] == "assistant")
    assert asst["content"] == "ok"
    assert asst["tool_calls"][0]["function"]["name"] == "Read"
    assert asst["tool_calls"][0]["function"]["arguments"] == '{"path": "f"}'
    assert "thinking_blocks" in asst


def test_charz_assistant_cache_control_text_uses_list_form():
    # A text block carrying cache_control forces the list content form (rather
    # than the concatenated string form).
    req, _ = _translate(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "cached",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
    )
    asst = next(m for m in req["messages"] if m["role"] == "assistant")
    assert isinstance(asst["content"], list)
    assert asst["content"][0]["text"] == "cached"


def test_charz_assistant_plain_text_concatenated():
    # Without cache_control, multiple text blocks collapse to a single string.
    req, _ = _translate(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ]
    )
    asst = next(m for m in req["messages"] if m["role"] == "assistant")
    assert asst["content"] == "ab"


# ---------------------------------------------------------------------------
# Characterization (golden) tests: lock the current streaming behavior so the
# H4 __next__/__anext__ dedup refactor cannot change observable output.
#
# Two invariants are locked:
#  1. Shared core: for a "shaperless" stream (no empty-choices usage chunk),
#     the sync `anthropic_sse_wrapper` and async `async_anthropic_sse_wrapper`
#     emit identical SSE (modulo the random message id). This guards the
#     __next__/__anext__ logic the dedup merges. The production async path runs
#     chunks through OpenAILikeProvider._shape_stream first (which reshapes the
#     include_usage empty-choices chunk); the sync path has no such shaper, so
#     equality is only meaningful for shaperless inputs.
#  2. Live async: the production pipeline (_shape_stream -> async wrapper) emits
#     well-ordered SSE with merged usage for the usual scenarios.
# The sync/async error paths are intentionally asymmetric and are covered by the
# dedicated error tests, not here.
# ---------------------------------------------------------------------------


def _strip_msg_id(body):
    """Mask the random `msg_<uuid>` id so two runs are byte-comparable."""
    return re.sub(r"msg_[A-Za-z0-9-]+", "msg_X", body)


def _event_names(body):
    return [
        line.split("event: ")[1]
        for line in body.splitlines()
        if line.startswith("event: ")
    ]


def _sync_sse(make_chunks):
    """Drive the sync wrapper over a fresh chunk list and return decoded SSE."""
    wrapper = BlueLLMStreamWrapper(
        completion_stream=iter(make_chunks()), model="gpt-5.4", tool_name_mapping={}
    )
    out = b""
    for sse in wrapper.anthropic_sse_wrapper():
        out += sse if isinstance(sse, (bytes, bytearray)) else sse.encode()
    return out.decode()


def _async_sse(make_chunks, shape=False):
    """Drive the async wrapper over a fresh chunk list and return decoded SSE.

    When ``shape`` is True the chunks pass through the production
    ``_shape_stream`` (needed for include_usage empty-choices chunks).
    """

    async def agen():
        for chunk in make_chunks():
            yield chunk

    async def run():
        stream = OpenAILikeProvider._shape_stream(agen()) if shape else agen()
        wrapper = BlueLLMStreamWrapper(
            completion_stream=stream, model="gpt-5.4", tool_name_mapping={}
        )
        out = b""
        async for sse in wrapper.async_anthropic_sse_wrapper():
            out += sse if isinstance(sse, (bytes, bytearray)) else sse.encode()
        return out.decode()

    return asyncio.run(run())


# Shaperless scenarios (no empty-choices usage chunk) — sync and async share the
# exact same code path, so their output must match after masking the message id.
_SHAPERLESS_SCENARIOS = {
    "text_no_usage": lambda: [
        stream_chunk(content="Hi"),
        stream_chunk(finish_reason="stop"),
    ],
    "two_text_chunks": lambda: [
        stream_chunk(content="Hel"),
        stream_chunk(content="lo"),
        stream_chunk(finish_reason="stop"),
    ],
    "usage_on_finish_chunk": lambda: [
        stream_chunk(content="Hi"),
        stream_chunk(finish_reason="stop", use=usage(7, 2)),
    ],
    "empty_content_then_finish": lambda: [
        stream_chunk(content=""),
        stream_chunk(finish_reason="stop"),
    ],
    "text_then_tool_use": lambda: [
        stream_chunk(content="Let me read it"),
        stream_chunk(tool_calls=[tool_call(name="Read", arguments='{"path":"f"}')]),
        stream_chunk(finish_reason="tool_calls"),
    ],
}


@pytest.mark.parametrize("name", sorted(_SHAPERLESS_SCENARIOS))
def test_streaming_sync_async_identical(name):
    # H4 safety net: the shared __next__/__anext__ core must emit identical SSE
    # (modulo the random message id) for a shaperless clean stream.
    make = _SHAPERLESS_SCENARIOS[name]
    assert _strip_msg_id(_sync_sse(make)) == _strip_msg_id(_async_sse(make))


@pytest.mark.parametrize("name", sorted(_SHAPERLESS_SCENARIOS))
def test_streaming_shaperless_event_order(name):
    body = _async_sse(_SHAPERLESS_SCENARIOS[name])
    events = _event_names(body)
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert "message_delta" in events
    last_delta = len(events) - 1 - events[::-1].index("message_delta")
    assert events.index("content_block_stop") < last_delta


# Live scenarios go through the production _shape_stream (include_usage chunks).
_LIVE_SCENARIOS = {
    "text_then_usage": (
        lambda: [
            stream_chunk(content="Hel"),
            stream_chunk(content="lo"),
            stream_chunk(finish_reason="stop"),
            usage_only_chunk(usage(10, 3)),
        ],
        (10, 3),
    ),
    "finish_in_last_content_chunk": (
        lambda: [
            stream_chunk(content="Hi"),
            stream_chunk(content="!", finish_reason="stop"),
            usage_only_chunk(usage(5, 1)),
        ],
        (5, 1),
    ),
    "finish_then_separate_usage": (
        lambda: [
            stream_chunk(content="Hi"),
            stream_chunk(finish_reason="stop"),
            usage_only_chunk(usage(8, 2)),
        ],
        (8, 2),
    ),
}


@pytest.mark.parametrize("name", sorted(_LIVE_SCENARIOS))
def test_streaming_live_pipeline(name):
    # Production pipeline (_shape_stream -> async wrapper): well-ordered SSE with
    # the include_usage values merged into the terminal message_delta.
    make, (in_tok, out_tok) = _LIVE_SCENARIOS[name]
    body = _async_sse(make, shape=True)
    events = _event_names(body)
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    last_delta = len(events) - 1 - events[::-1].index("message_delta")
    assert events.index("content_block_stop") < last_delta
    assert f'"input_tokens": {in_tok}' in body
    assert f'"output_tokens": {out_tok}' in body


def test_streaming_upstream_error_emits_error_event():
    # H2: an exception raised mid-stream must surface as an Anthropic `error`
    # SSE event (not abort the response with no terminal event), and must not
    # leak raw upstream detail to the client.
    async def boom_stream():
        yield stream_chunk(content="Hi")
        raise RuntimeError("upstream exploded mid-stream")

    async def run():
        wrapper = BlueLLMStreamWrapper(
            completion_stream=boom_stream(), model="gpt-5.4", tool_name_mapping={}
        )
        out = b""
        async for sse in wrapper.async_anthropic_sse_wrapper():
            out += sse if isinstance(sse, (bytes, bytearray)) else sse.encode()
        return out.decode()

    body = asyncio.run(run())
    events = [
        line.split("event: ")[1]
        for line in body.splitlines()
        if line.startswith("event: ")
    ]
    assert "error" in events
    assert "exploded" not in body


def test_sync_next_error_log_omits_raw_exception(caplog):
    # M6: the sync __next__ error log must carry the exception type/status, not
    # str(e) (which may include endpoint/body detail) or an inlined traceback.
    def boom():
        raise RuntimeError("RAWSYNCDETAIL")
        yield  # pragma: no cover - makes boom a generator

    wrapper = BlueLLMStreamWrapper(
        completion_stream=boom(), model="gpt-5.4", tool_name_mapping={}
    )
    with caplog.at_level("ERROR"):
        list(wrapper.anthropic_sse_wrapper())
    rec = next(r for r in caplog.records if "stream aborted" in r.getMessage())
    assert "RAWSYNCDETAIL" not in rec.getMessage()


def test_streaming_error_log_message_omits_raw_exception(caplog):
    # H-4: the error log message line must carry exception type/status, not the
    # raw str(e) (which may include endpoint/body detail).
    async def boom_stream():
        yield stream_chunk(content="Hi")
        raise RuntimeError("RAWUPSTREAMDETAIL")

    async def run():
        wrapper = BlueLLMStreamWrapper(
            completion_stream=boom_stream(), model="gpt-5.4", tool_name_mapping={}
        )
        async for _ in wrapper.async_anthropic_sse_wrapper():
            pass

    with caplog.at_level("ERROR"):
        asyncio.run(run())
    rec = next(r for r in caplog.records if "stream aborted" in r.getMessage())
    assert "RAWUPSTREAMDETAIL" not in rec.getMessage()
