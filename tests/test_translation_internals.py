"""Branch coverage for the Anthropic<->OpenAI input/output/stream translators."""

import pytest
from helpers import stream_chunk

from bluellm.translation import BlueLLMMessagesAdapter, BlueLLMMessagesAdapter

A = BlueLLMMessagesAdapter()
M = BlueLLMMessagesAdapter()
_MSGS = [{"role": "user", "content": "hi"}]


def _in(model="claude-x", **extra):
    body = {"model": model, "max_tokens": 16, "messages": _MSGS, **extra}
    req, mapping = A.translate_completion_input_params_with_tool_mapping(dict(body))
    return req, mapping


# ---- required-param validation ----
def test_missing_model_raises():
    with pytest.raises(ValueError):
        A.translate_completion_input_params_with_tool_mapping(
            {"model": "", "messages": _MSGS}
        )


def test_missing_messages_raises():
    with pytest.raises(ValueError):
        A.translate_completion_input_params_with_tool_mapping(
            {"model": "m", "messages": []}
        )


# ---- system message ----
def test_system_string_prepended():
    req, _ = _in(system="be nice")
    assert req["messages"][0] == {"role": "system", "content": "be nice"}


def test_system_list_with_cache_control():
    req, _ = _in(
        model="claude-x",
        system=[{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
    )
    sys_msg = req["messages"][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"][0]["text"] == "s"


def test_system_empty_is_skipped():
    req, _ = _in(system="")
    assert all(m["role"] != "system" for m in req["messages"])


# ---- tool_choice ----
@pytest.mark.parametrize(
    "tc,expected",
    [
        ({"type": "any"}, "required"),
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
    ],
)
def test_tool_choice_simple(tc, expected):
    req, _ = _in(tool_choice=tc)
    assert req["tool_choice"] == expected


def test_tool_choice_specific_tool():
    req, _ = _in(tool_choice={"type": "tool", "name": "Read"})
    assert req["tool_choice"]["function"]["name"] == "Read"


def test_tool_choice_invalid_raises():
    with pytest.raises(ValueError):
        M.translate_anthropic_tool_choice_to_openai({"type": "bogus"})


# ---- tools ----
def test_tools_long_name_truncated_and_mapped():
    long = "x" * 80
    _, mapping = _in(
        tools=[{"name": long, "input_schema": {"type": "object", "properties": {}}}]
    )
    assert long in mapping.values()


def test_tools_unnamed_gets_placeholder():
    req, _ = _in(tools=[{"input_schema": {"type": "object", "properties": {}}}])
    names = [t["function"]["name"] for t in req["tools"]]
    assert any("unnamed_tool" in n for n in names)


def test_tools_extra_kwargs_merged_into_parameters():
    req, _ = _in(tools=[{"name": "T", "display_width_px": 100}])
    fn = req["tools"][0]["function"]
    assert fn["parameters"]["display_width_px"] == 100


# ---- thinking (non-claude model -> reasoning_effort) ----
@pytest.mark.parametrize(
    "budget,effort",
    [(10000, "high"), (5000, "medium"), (2000, "low"), (500, "minimal")],
)
def test_thinking_budget_tiers(budget, effort):
    req, _ = _in(model="gpt-x", thinking={"type": "enabled", "budget_tokens": budget})
    assert req["reasoning_effort"] == effort


def test_thinking_disabled_no_effort():
    req, _ = _in(model="gpt-x", thinking={"type": "disabled"})
    assert "reasoning_effort" not in req


def test_thinking_claude_passthrough():
    req, _ = _in(model="claude-x", thinking={"type": "enabled", "budget_tokens": 9000})
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 9000}


def test_thinking_adaptive_uses_output_config_effort():
    req, _ = _in(
        model="gpt-x",
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
    )
    assert req["reasoning_effort"] == "high"


def test_thinking_summary_preserved():
    req, _ = _in(
        model="gpt-x",
        thinking={"type": "enabled", "budget_tokens": 10000, "summary": "detailed"},
    )
    assert req["reasoning_effort"] == {"effort": "high", "summary": "detailed"}


def test_thinking_auto_summary_env(monkeypatch):
    monkeypatch.setenv("BLUELLM_REASONING_AUTO_SUMMARY", "true")
    req, _ = _in(model="gpt-x", thinking={"type": "enabled", "budget_tokens": 10000})
    assert req["reasoning_effort"] == {"effort": "high", "summary": "detailed"}


# ---- output_format -> response_format ----
def test_output_format_json_schema():
    req, _ = _in(
        output_format={
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "object", "properties": {"b": {"type": "string"}}}},
                },
            },
        }
    )
    rf = req["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


def test_output_format_invalid_ignored():
    req, _ = _in(output_format={"type": "text"})
    assert "response_format" not in req


# ---- metadata ----
def test_metadata_user_id_mapped():
    req, _ = _in(metadata={"user_id": "u-1"})
    assert req["user"] == "u-1"


# ===========================================================================
# Output translation (translate_completion_output_params)
# ===========================================================================
from types import SimpleNamespace  # noqa: E402

from bluellm._compat import StreamingChoices  # noqa: E402


def _usage(prompt=10, completion=3, cached=0, cache_creation=0):
    u = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=(
            SimpleNamespace(cached_tokens=cached) if cached else None
        ),
    )
    if cache_creation:
        u._cache_creation_input_tokens = cache_creation
    return u


def _resp(message, finish_reason="stop", usage=None):
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id="cmpl", model="gpt", choices=[choice], usage=usage or _usage()
    )


def _msg(**kw):
    base = dict(role="assistant", content=None, tool_calls=None, function_call=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_output_thinking_blocks():
    msg = _msg(
        content="hi",
        thinking_blocks=[
            {"type": "thinking", "thinking": "t", "signature": "s"},
            {"type": "redacted_thinking", "data": "d"},
        ],
    )
    out = A.translate_completion_output_params(response=_resp(msg), tool_name_mapping={})
    types = [b["type"] for b in out["content"]]
    assert "thinking" in types and "redacted_thinking" in types


def test_output_reasoning_content_fallback():
    msg = _msg(content="hi", reasoning_content="because")
    out = A.translate_completion_output_params(response=_resp(msg), tool_name_mapping={})
    think = next(b for b in out["content"] if b["type"] == "thinking")
    assert think["thinking"] == "because"


def test_output_tool_use_signature_and_thought_id():
    tc = SimpleNamespace(
        id="call_1__thought__SIGTOKEN",
        type="function",
        function=SimpleNamespace(name="Read", arguments='{"path":"f"}'),
        provider_specific_fields={"thought_signature": "sig-x"},
    )
    msg = _msg(content=None, tool_calls=[tc])
    out = A.translate_completion_output_params(
        response=_resp(msg, finish_reason="tool_calls"), tool_name_mapping={}
    )
    block = next(b for b in out["content"] if b["type"] == "tool_use")
    assert block["id"] == "call_1"  # thought-signature suffix stripped
    assert block["provider_specific_fields"]["signature"] == "sig-x"


def test_output_finish_reason_length_to_max_tokens():
    out = A.translate_completion_output_params(
        response=_resp(_msg(content="x"), finish_reason="length"), tool_name_mapping={}
    )
    assert out["stop_reason"] == "max_tokens"


def test_output_usage_cached_and_cache_creation():
    usage = _usage(prompt=10, completion=3, cached=4, cache_creation=2)
    out = A.translate_completion_output_params(
        response=_resp(_msg(content="x"), usage=usage), tool_name_mapping={}
    )
    assert out["usage"]["input_tokens"] == 6  # 10 - 4 cached
    assert out["usage"]["cache_read_input_tokens"] == 4
    assert out["usage"]["cache_creation_input_tokens"] == 2


def _legacy_usage_dict(usage):
    """共通化前の usage 抽出ロジック（バイト等価性検証用の基準実装）。"""
    uncached_input_tokens = usage.prompt_tokens or 0
    cached_tokens = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        uncached_input_tokens -= cached_tokens
    result = {
        "input_tokens": uncached_input_tokens,
        "output_tokens": usage.completion_tokens or 0,
    }
    if (
        hasattr(usage, "_cache_creation_input_tokens")
        and usage._cache_creation_input_tokens > 0
    ):
        result["cache_creation_input_tokens"] = usage._cache_creation_input_tokens
    if cached_tokens > 0:
        result["cache_read_input_tokens"] = cached_tokens
    return result


@pytest.mark.parametrize(
    "prompt,completion,cached,cache_creation",
    [
        (10, 3, 0, 0),  # キャッシュなし: 任意キー欠落
        (10, 3, 4, 0),  # cache_read のみ
        (10, 3, 0, 2),  # cache_creation のみ
        (10, 3, 4, 2),  # 双方
        (None, None, 0, 0),  # prompt/completion None -> 0 フォールバック
    ],
)
def test_build_usage_dict_matches_legacy(prompt, completion, cached, cache_creation):
    usage = _usage(
        prompt=prompt, completion=completion, cached=cached, cache_creation=cache_creation
    )
    result = M._build_usage_dict(usage)
    legacy = _legacy_usage_dict(usage)
    assert result == legacy
    assert set(result.keys()) == set(legacy.keys())  # キー集合の完全一致


# ===========================================================================
# Streaming chunk translation (lower-level helpers)
# ===========================================================================
def _delta(**kw):
    base = dict(content=None, tool_calls=None, role=None, function_call=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _schoice(**delta_kw):
    c = StreamingChoices()
    c.delta = _delta(**delta_kw)
    c.finish_reason = None
    return c


def _tool_delta(name=None, arguments=None, call_id="call_1"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_stream_content_block_tool_use():
    block_type, block = M._translate_streaming_openai_chunk_to_anthropic_content_block(
        choices=[_schoice(tool_calls=[_tool_delta(name="Read")])]
    )
    assert block_type == "tool_use"
    assert block["name"] == "Read"


def test_stream_content_block_thinking():
    block_type, _ = M._translate_streaming_openai_chunk_to_anthropic_content_block(
        choices=[
            _schoice(thinking_blocks=[{"type": "thinking", "thinking": "t", "signature": ""}])
        ]
    )
    assert block_type == "thinking"


def test_stream_content_block_reasoning_content():
    block_type, _ = M._translate_streaming_openai_chunk_to_anthropic_content_block(
        choices=[_schoice(reasoning_content="r")]
    )
    assert block_type == "thinking"


def test_stream_content_block_text_default():
    block_type, _ = M._translate_streaming_openai_chunk_to_anthropic_content_block(
        choices=[_schoice(content="hello")]
    )
    assert block_type == "text"


def test_stream_delta_text():
    dtype, delta = M._translate_streaming_openai_chunk_to_anthropic(
        choices=[_schoice(content="hi")]
    )
    assert dtype == "text_delta" and delta["text"] == "hi"


def test_stream_delta_input_json():
    dtype, delta = M._translate_streaming_openai_chunk_to_anthropic(
        choices=[_schoice(tool_calls=[_tool_delta(arguments='{"a":1}')])]
    )
    assert dtype == "input_json_delta" and delta["partial_json"] == '{"a":1}'


def test_stream_delta_thinking():
    dtype, delta = M._translate_streaming_openai_chunk_to_anthropic(
        choices=[
            _schoice(thinking_blocks=[{"type": "thinking", "thinking": "t", "signature": ""}])
        ]
    )
    assert dtype == "thinking_delta" and delta["thinking"] == "t"


def test_stream_delta_signature():
    dtype, delta = M._translate_streaming_openai_chunk_to_anthropic(
        choices=[
            _schoice(thinking_blocks=[{"type": "thinking", "thinking": "", "signature": "s"}])
        ]
    )
    assert dtype == "signature_delta" and delta["signature"] == "s"


def test_stream_delta_reasoning_content():
    dtype, delta = M._translate_streaming_openai_chunk_to_anthropic(
        choices=[_schoice(reasoning_content="r")]
    )
    assert dtype == "thinking_delta" and delta["thinking"] == "r"


# ---- streaming final-chunk usage extraction ----
def _final_chunk(finish="stop", usage=None):
    choice = SimpleNamespace(index=0, delta=_delta(), finish_reason=finish)
    return SimpleNamespace(id="c", model="m", choices=[choice], usage=usage)


def test_stream_final_usage_cached_and_cache_creation():
    usage = _usage(prompt=10, completion=3, cached=4, cache_creation=2)
    out = M.translate_streaming_openai_response_to_anthropic(
        response=_final_chunk(usage=usage),
        current_content_block_index=0,
        applied_edits=[{"type": "clear_tool_uses_20250919"}],
    )
    assert out["type"] == "message_delta"
    assert out["usage"]["input_tokens"] == 6
    assert out["usage"]["cache_read_input_tokens"] == 4
    assert out["usage"]["cache_creation_input_tokens"] == 2
    assert "context_management" in out


def test_stream_final_usage_absent():
    out = M.translate_streaming_openai_response_to_anthropic(
        response=_final_chunk(usage=None), current_content_block_index=0
    )
    assert out["usage"]["input_tokens"] == 0


# ===========================================================================
# Remaining input/assistant/tool_result/output edges
# ===========================================================================
def test_metadata_field_mapped_to_metadata():
    req, _ = _in(bluellm_metadata={"trace": "1"})
    assert req["metadata"] == {"trace": "1"}


def test_falsy_tool_choice_skipped():
    req, _ = _in(tool_choice={})
    assert "tool_choice" not in req


def test_empty_tools_skipped():
    req, _ = _in(tools=[])
    assert "tools" not in req


def test_falsy_thinking_skipped():
    req, _ = _in(model="gpt-x", thinking={})
    assert "reasoning_effort" not in req and "thinking" not in req


def test_web_search_tool_sets_options_and_no_regular_tools():
    req, _ = _in(tools=[{"type": "web_search_20250305", "name": "web_search"}])
    assert req["web_search_options"] == {}
    assert "tools" not in req  # only web-search tools -> no regular tools list


def test_web_search_plus_regular_tool():
    req, _ = _in(
        tools=[
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "Read", "input_schema": {"type": "object", "properties": {}}},
        ]
    )
    assert req["web_search_options"] == {}
    assert any(t["function"]["name"] == "Read" for t in req["tools"])


def test_hosted_tool_kept_as_is():
    req, _ = _in(tools=[{"type": "bash_20250124", "name": "bash"}])
    assert req["tools"][0]["type"] == "bash_20250124"


def test_output_format_non_dict_ignored():
    assert M.translate_anthropic_output_format_to_openai("nope") is None


def test_output_format_missing_schema_ignored():
    assert M.translate_anthropic_output_format_to_openai({"type": "json_schema"}) is None


def test_output_format_anyof_and_defs():
    rf = M.translate_anthropic_output_format_to_openai(
        {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"x": {"anyOf": [{"type": "object", "properties": {"y": {"type": "string"}}}]}},
                "$defs": {"D": {"type": "object", "properties": {"z": {"type": "string"}}}},
            },
        }
    )
    defs = rf["json_schema"]["schema"]["$defs"]["D"]
    assert defs["additionalProperties"] is False


def test_assistant_string_content():
    req, _ = _translate_one({"role": "assistant", "content": "plain"})
    asst = next(m for m in req["messages"] if m["role"] == "assistant")
    assert asst["content"] == "plain"


def test_assistant_list_with_bare_string_item():
    req, _ = _translate_one({"role": "assistant", "content": ["bare"]})
    asst = next(m for m in req["messages"] if m["role"] == "assistant")
    assert asst["content"] == "bare"


def test_assistant_tool_use_with_signature():
    req, _ = _translate_one(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t",
                    "name": "R",
                    "input": {},
                    "provider_specific_fields": {"signature": "sig"},
                }
            ],
        }
    )
    asst = next(m for m in req["messages"] if m["role"] == "assistant")
    fn = asst["tool_calls"][0]["function"]
    assert fn["provider_specific_fields"]["thought_signature"] == "sig"


def test_tool_result_single_string_in_list():
    req, _ = _translate_one(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": ["just text"]}
            ],
        }
    )
    tool = next(m for m in req["messages"] if m["role"] == "tool")
    assert tool["content"] == "just text"


def _translate_one(message):
    body = {"model": "claude-x", "max_tokens": 16, "messages": [message]}
    return A.translate_completion_input_params_with_tool_mapping(dict(body))


def test_output_tool_name_restored_from_mapping():
    tc = SimpleNamespace(
        id="c", type="function", function=SimpleNamespace(name="SHORT", arguments="{}")
    )
    out = A.translate_completion_output_params(
        response=_resp(_msg(tool_calls=[tc]), finish_reason="tool_calls"),
        tool_name_mapping={"SHORT": "TheRealLongName"},
    )
    block = next(b for b in out["content"] if b["type"] == "tool_use")
    assert block["name"] == "TheRealLongName"


def test_output_tool_use_function_level_signature():
    tc = SimpleNamespace(
        id="c",
        type="function",
        function=SimpleNamespace(
            name="R", arguments="{}", provider_specific_fields={"thought_signature": "fsig"}
        ),
    )
    out = A.translate_completion_output_params(
        response=_resp(_msg(tool_calls=[tc]), finish_reason="tool_calls"),
        tool_name_mapping={},
    )
    block = next(b for b in out["content"] if b["type"] == "tool_use")
    assert block["provider_specific_fields"]["signature"] == "fsig"


# ---- streaming thinking error branches ----
def test_stream_content_block_signature_nonstr_raises():
    with pytest.raises(TypeError):
        M._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=[
                _schoice(thinking_blocks=[{"type": "thinking", "thinking": "", "signature": 1}])
            ]
        )


def test_stream_content_block_thinking_and_signature_raises():
    with pytest.raises(ValueError):
        M._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=[
                _schoice(thinking_blocks=[{"type": "thinking", "thinking": "t", "signature": "s"}])
            ]
        )


def test_stream_delta_signature_nonstr_raises():
    with pytest.raises(TypeError):
        M._translate_streaming_openai_chunk_to_anthropic(
            choices=[
                _schoice(thinking_blocks=[{"type": "thinking", "thinking": "", "signature": 1}])
            ]
        )


def test_stream_delta_reasoning_and_signature_raises():
    with pytest.raises(ValueError):
        M._translate_streaming_openai_chunk_to_anthropic(
            choices=[
                _schoice(thinking_blocks=[{"type": "thinking", "thinking": "t", "signature": "s"}])
            ]
        )


def test_stream_delta_thinking_nonstr_raises():
    with pytest.raises(TypeError):
        M._translate_streaming_openai_chunk_to_anthropic(
            choices=[
                _schoice(thinking_blocks=[{"type": "thinking", "thinking": 99, "signature": ""}])
            ]
        )


def test_stream_content_block_tool_use_thought_signature_id():
    _, block = M._translate_streaming_openai_chunk_to_anthropic_content_block(
        choices=[_schoice(tool_calls=[_tool_delta(name="R", call_id="c__thought__SIG")])]
    )
    assert block["id"] == "c"
    assert block["provider_specific_fields"]["signature"] == "SIG"


# ---- remaining misc edges ----
def test_thinking_to_effort_non_dict_and_unknown_type():
    assert M.translate_anthropic_thinking_to_reasoning_effort("nope") is None
    assert M.translate_anthropic_thinking_to_reasoning_effort({"type": "weird"}) is None


def test_add_additional_properties_false_ignores_non_dict():
    # Direct call with a non-dict schema is a no-op (returns without error).
    M._add_additional_properties_false("not-a-dict")


def test_output_format_object_with_items():
    rf = M.translate_anthropic_output_format_to_openai(
        {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "items": {"type": "object", "properties": {"b": {"type": "string"}}}},
        }
    )
    assert rf["json_schema"]["schema"]["items"]["additionalProperties"] is False


def test_finish_reason_unknown_maps_to_end_turn():
    out = A.translate_completion_output_params(
        response=_resp(_msg(content="x"), finish_reason="function_call"),
        tool_name_mapping={},
    )
    assert out["stop_reason"] == "end_turn"


def test_tool_result_multi_with_string_and_image():
    req, _ = _translate_one(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": [
                        "plain",
                        {"type": "text", "text": "txt"},
                        {"type": "image", "source": {"type": "url", "url": "https://i/i.png"}},
                    ],
                }
            ],
        }
    )
    tool = next(m for m in req["messages"] if m["role"] == "tool")
    types = [p["type"] for p in tool["content"]]
    assert types == ["text", "text", "image_url"]


def test_output_with_polyfill_result():
    pf = SimpleNamespace(
        compaction_block={"type": "text", "text": "compacted"},
        iterations_usage=[{"type": "message", "input_tokens": 1, "output_tokens": 1}],
        applied_edits_for_response=lambda: [{"type": "clear_tool_uses_20250919"}],
    )
    out = A.translate_completion_output_params(
        response=_resp(_msg(content="x")), tool_name_mapping={}, polyfill_result=pf
    )
    assert out["content"][0] == {"type": "text", "text": "compacted"}
    assert "iterations" in out["usage"]
    assert "context_management" in out


def test_sync_streaming_wrapper_is_async_false():
    def gen():
        yield stream_chunk(content="hi")
        yield stream_chunk(finish_reason="stop")

    it = A.translate_completion_output_params_streaming(gen(), model="m", is_async=False)
    body = b"".join(c if isinstance(c, bytes) else c.encode() for c in it).decode()
    assert "message_start" in body and "message_stop" in body
