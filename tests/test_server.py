import pytest
from helpers import (
    AZURE_KEY,
    CONFIG_TEMPLATE,
    MASTER_KEY,
    install_fake_client,
    stream_chunk,
    text_completion,
    usage,
    usage_only_chunk,
)
from openai import BadRequestError
from starlette.testclient import TestClient

from bluellm.config import load_config
from bluellm.server import create_app

AUTH = {"x-api-key": MASTER_KEY}
MSG = {"model": "claude-x", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def _make(create_fn):
        monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
        monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
        cfg = tmp_path / "config.yml"
        cfg.write_text(CONFIG_TEMPLATE)
        install_fake_client(monkeypatch, create_fn)
        return TestClient(create_app(load_config(str(cfg))))

    return _make


def _bad_request(param, code):
    e = BadRequestError.__new__(BadRequestError)
    e.param = param
    e.code = code
    e.args = (f"Unsupported value: '{param}' is not supported with this model.",)
    return e


def test_health_no_auth(make_client):
    client = make_client(lambda **kw: None)
    assert client.get("/health").json() == {"status": "ok"}



def test_root_no_auth(make_client):
    client = make_client(lambda **kw: None)
    assert client.get("/").json() == {"status": "ok"}
    assert client.head("/").status_code == 200


def test_auth_required(make_client):
    async def create(**kw):
        return text_completion()

    client = make_client(create)
    assert client.post("/v1/messages", json=MSG).status_code == 401
    assert client.post("/v1/messages", headers={"x-api-key": "wrong"}, json=MSG).status_code == 401
    assert client.post("/v1/messages", headers=AUTH, json=MSG).status_code == 200


def test_nonstream_response(make_client):
    async def create(**kw):
        return text_completion(text="Hello there")

    client = make_client(create)
    body = client.post("/v1/messages", headers=AUTH, json=MSG).json()
    assert body["content"] == [{"type": "text", "text": "Hello there"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 7}


def test_streaming_response(make_client):
    async def create(**kw):
        async def gen():
            yield stream_chunk(content="Hi")
            yield stream_chunk(finish_reason="stop")
            yield usage_only_chunk(usage(8, 2))

        assert kw.get("stream") is True
        assert kw.get("stream_options") == {"include_usage": True}
        return gen()

    client = make_client(create)
    with client.stream("POST", "/v1/messages", headers=AUTH, json={**MSG, "stream": True}) as r:
        body = b"".join(r.iter_bytes()).decode()
    events = [
        line.split("event: ")[1]
        for line in body.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "message_start" and events[-1] == "message_stop"
    assert "content_block_delta" in events


def test_auto_drop_unsupported_params(make_client):
    calls = []

    async def create(**kw):
        calls.append(dict(kw))
        if "temperature" in kw:
            raise _bad_request("temperature", "unsupported_value")
        if "top_p" in kw:
            raise _bad_request("top_p", "unsupported_parameter")
        return text_completion()

    client = make_client(create)
    r = client.post(
        "/v1/messages",
        headers=AUTH,
        json={**MSG, "temperature": 0.7, "top_p": 0.9},
    )
    assert r.status_code == 200
    assert len(calls) == 3  # temperature drop, top_p drop, success
    assert "temperature" not in calls[-1] and "top_p" not in calls[-1]
    # max_tokens was rewritten to max_completion_tokens and never dropped
    assert calls[-1]["max_completion_tokens"] == 50


def test_context_management_stripped_before_upstream_call(make_client):
    calls = []

    async def create(**kw):
        calls.append(dict(kw))
        return text_completion(text="Hello there")

    client = make_client(create)
    r = client.post(
        "/v1/messages",
        headers=AUTH,
        json={
            **MSG,
            "context_management": {"clear_function_results": True},
        },
    )
    assert r.status_code == 200
    assert calls
    assert "context_management" not in calls[-1]


def test_upstream_error_sanitized(make_client):
    async def create(**kw):
        e = BadRequestError.__new__(BadRequestError)
        e.param = None
        e.code = "content_policy_violation"
        e.status_code = 400
        e.args = ("internal upstream detail with https://example.openai.azure.com",)
        raise e

    client = make_client(create)
    r = client.post("/v1/messages", headers=AUTH, json=MSG)
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert "azure.com" not in body["error"]["message"]


def test_unsupported_content_returns_400(make_client):
    # M8/L5: a request carrying an image source the proxy can't translate must
    # be rejected with a 400 invalid_request_error rather than silently dropping
    # the image (or returning a 500).
    async def create(**kw):
        return text_completion()

    client = make_client(create)
    r = client.post(
        "/v1/messages",
        headers=AUTH,
        json={
            "model": "claude-x",
            "max_tokens": 50,
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
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_count_tokens(make_client):
    client = make_client(lambda **kw: None)
    r = client.post(
        "/v1/messages/count_tokens",
        headers=AUTH,
        json={"model": "x", "messages": [{"role": "user", "content": "hello world"}]},
    )
    assert r.status_code == 200
    assert r.json()["input_tokens"] >= 1


def test_unknown_model_returns_404(tmp_path, monkeypatch):
    # No "*" catch-all configured -> an unmatched model name is a 404.
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    no_wildcard = """
models:
  - name: "gpt-only"
    params:
      model: azure/gpt-5.4
      endpoint: https://example.openai.azure.com
      key: os.environ/AZURE_API_KEY
      version: "2025-01-01-preview"
generals:
  key: os.environ/BLUELLM_MASTER_KEY
"""
    cfg = tmp_path / "config.yml"
    cfg.write_text(no_wildcard)
    install_fake_client(monkeypatch, lambda **kw: text_completion())
    client = TestClient(create_app(load_config(str(cfg))))
    r = client.post("/v1/messages", headers=AUTH, json={**MSG, "model": "claude-x"})
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


def test_count_tokens_invalid_json(make_client):
    client = make_client(lambda **kw: None)
    r = client.post("/v1/messages/count_tokens", headers=AUTH, content="not json{")
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_non_status_upstream_error_is_500(make_client):
    # A non-APIStatusError from the provider is sanitized to a generic 500.
    async def create(**kw):
        raise RuntimeError("internal boom with secret detail")

    client = make_client(create)
    r = client.post("/v1/messages", headers=AUTH, json=MSG)
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["type"] == "api_error"
    assert "boom" not in body["error"]["message"]


def test_invalid_json_body_returns_anthropic_error(make_client):
    # M4: malformed JSON must surface as an Anthropic-shaped error, not a 500.
    client = make_client(lambda **kw: None)
    r = client.post("/v1/messages", headers=AUTH, content="not json{")
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_unknown_model_404_body_omits_internal_detail(tmp_path, monkeypatch):
    # S2-a: the 404 body must not leak routing internals (presence/absence of the
    # '*' catch-all) nor echo back the client-sent model name. Status/type stay 404.
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    no_wildcard = """
models:
  - name: "gpt-only"
    params:
      model: azure/gpt-5.4
      endpoint: https://example.openai.azure.com
      key: os.environ/AZURE_API_KEY
      version: "2025-01-01-preview"
generals:
  key: os.environ/BLUELLM_MASTER_KEY
"""
    cfg = tmp_path / "config.yml"
    cfg.write_text(no_wildcard)
    install_fake_client(monkeypatch, lambda **kw: text_completion())
    client = TestClient(create_app(load_config(str(cfg))))
    r = client.post("/v1/messages", headers=AUTH, json={**MSG, "model": "claude-secret-xyz"})
    assert r.status_code == 404
    msg = r.json()["error"]["message"]
    assert "catch-all" not in msg
    assert "model_list" not in msg
    assert "claude-secret-xyz" not in msg


def test_unsupported_content_400_body_is_client_facing_only(make_client):
    # S2-b: unsupported content is an explicit 400 describing the client's own
    # input; it carries no internal config/path/upstream detail, so it is NOT a
    # leak. This guards the current (correct) behavior against accidental change.
    client = make_client(lambda **kw: text_completion())
    bad = {
        "model": "claude-x",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": [{"type": "image", "source": "not-an-object"}]}
        ],
    }
    r = client.post("/v1/messages", headers=AUTH, json=bad)
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    # client-facing description of the offending input is retained
    assert "image source" in body["error"]["message"]
    # no internal leakage
    assert "azure.com" not in body["error"]["message"]
    assert "/home/" not in body["error"]["message"]
    assert "Traceback" not in body["error"]["message"]


def test_large_body_returns_413(make_client):
    # M-4: Content-Length exceeding max_request_body_mb (default 10 MB) returns 413.
    client = make_client(lambda **kw: None)
    oversized = 11 * 1024 * 1024  # 11 MB > default 10 MB
    r = client.post(
        "/v1/messages",
        headers={**AUTH, "content-length": str(oversized)},
        content=b"x",
    )
    assert r.status_code == 413
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_body_within_limit_passes_through(make_client):
    # M-4: Content-Length within the limit allows normal processing.
    async def create(**kw):
        return text_completion()

    client = make_client(create)
    r = client.post("/v1/messages", headers={**AUTH, "content-length": "100"}, json=MSG)
    assert r.status_code == 200


def test_badrequest_debug_log_omits_raw_body(make_client, caplog):
    # S3: the provider's BadRequestError debug log must carry type/param/code, not
    # the raw str(e) body, which can contain dynamic secrets absent from the
    # redaction set. Diagnostic param/code must still be present.
    async def create(**kw):
        if "temperature" in kw:
            e = BadRequestError.__new__(BadRequestError)
            e.param = "temperature"
            e.code = "unsupported_value"
            e.args = ("Unsupported value SECRETBODYMARKER token=sk-leaked",)
            raise e
        return text_completion()

    client = make_client(create)
    with caplog.at_level("DEBUG", logger="bluellm"):
        r = client.post("/v1/messages", headers=AUTH, json={**MSG, "temperature": 0.7})
    assert r.status_code == 200
    debug_text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "SECRETBODYMARKER" not in debug_text
    assert "sk-leaked" not in debug_text
    # diagnostic param/code retained for unsupported-param identification
    assert "temperature" in debug_text
