"""Targeted unit tests filling coverage gaps in the small core modules."""

import asyncio

import pytest
from openai import BadRequestError

from bluellm.config import Config, GeneralSettings, ModelConfig, _split_provider
from bluellm.providers import openai_like
from bluellm.providers.base import BaseProvider
from bluellm.providers.openai_like import OpenAILikeProvider, get_provider
from bluellm.router import Router


def _mc(**kw):
    base = dict(model_name="m", provider="openai", deployment="gpt", api_key="k")
    base.update(kw)
    return ModelConfig(**base)


# ---- router ----
def test_router_exact_named_match():
    cfg = Config(
        model_list=[_mc(model_name="claude-3", deployment="dep-a")],
        general_settings=GeneralSettings(),
    )
    assert Router(cfg).resolve("claude-3").deployment == "dep-a"


def test_router_raises_without_match_or_wildcard():
    cfg = Config(model_list=[_mc(model_name="only")], general_settings=GeneralSettings())
    with pytest.raises(KeyError):
        Router(cfg).resolve("missing")


# ---- config helpers ----
def test_split_provider_defaults_to_openai():
    assert _split_provider("gpt-5.4") == ("openai", "gpt-5.4")
    assert _split_provider("azure/dep") == ("azure", "dep")


def test_load_config_empty_models_raises(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("models: []\ngenerals: {}\n")
    from bluellm.config import load_config

    with pytest.raises(ValueError):
        load_config(str(p))


# ---- base provider ----
def test_base_provider_acreate_not_implemented():
    with pytest.raises(NotImplementedError):
        asyncio.run(BaseProvider().acreate({}, stream=False))


# ---- OpenAILikeProvider client construction ----
def test_build_client_openai():
    prov = OpenAILikeProvider(_mc(provider="openai", api_base=None))
    assert prov._client is not None


def test_build_client_azure_endpoint():
    prov = OpenAILikeProvider(
        _mc(
            provider="azure",
            api_base="https://x.openai.azure.com",
            api_version="2025-01-01-preview",
        )
    )
    assert prov._client is not None


def test_build_client_azure_full_base_url():
    prov = OpenAILikeProvider(
        _mc(
            provider="azure",
            api_base="https://x.openai.azure.com/openai/deployments/dep",
            api_version="2025-01-01-preview",
        )
    )
    assert prov._client is not None


# ---- _prepare extra_params + _sanitize_message ----
def test_prepare_merges_extra_params_and_sanitizes_messages():
    prov = OpenAILikeProvider(_mc(extra_params={"reasoning_effort": "high"}))
    out = prov._prepare(
        {
            "messages": [
                "not-a-dict",
                {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}},
            ]
        }
    )
    assert out["reasoning_effort"] == "high"
    # non-dict message passes through untouched; dict loses cache_control
    assert out["messages"][0] == "not-a-dict"
    assert "cache_control" not in out["messages"][1]


# ---- _droppable_param ----
def _bad_request(param=None, code=None, message="oops"):
    e = BadRequestError.__new__(BadRequestError)
    e.param = param
    e.code = code
    e.args = (message,)
    return e


def test_droppable_param_message_fallback():
    assert (
        OpenAILikeProvider._droppable_param(
            _bad_request(param="top_p", message="top_p is unsupported here")
        )
        == "top_p"
    )


def test_droppable_param_none_when_no_param():
    assert OpenAILikeProvider._droppable_param(_bad_request(param=None)) is None


def test_droppable_param_none_when_unrelated_message():
    assert (
        OpenAILikeProvider._droppable_param(
            _bad_request(param="top_p", message="please contact support")
        )
        is None
    )


# ---- get_provider ----
def test_get_provider_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_provider(_mc(provider="bedrock"))


def test_get_provider_caches_by_fingerprint():
    openai_like._PROVIDER_CACHE.clear()
    a = get_provider(_mc())
    b = get_provider(_mc())
    assert a is b
