from types import SimpleNamespace

from bluellm.auth import Authenticator, _extract_key, hash_token


def test_authenticator_enabled_flag():
    assert Authenticator("k").enabled is True
    assert Authenticator(None).enabled is False


def test_verify_noop_when_unauthenticated():
    # No master key configured -> verify accepts anything (open local mode).
    Authenticator(None).verify("whatever")
    Authenticator(None).verify(None)


def test_hash_token_is_sha256_hex():
    import hashlib

    assert hash_token("abc") == hashlib.sha256(b"abc").hexdigest()


def test_extract_key_bearer_header():
    req = SimpleNamespace(headers={"authorization": "Bearer tok-123"})
    assert _extract_key(req) == "tok-123"


def test_extract_key_prefers_x_api_key():
    req = SimpleNamespace(headers={"x-api-key": "xk", "authorization": "Bearer b"})
    assert _extract_key(req) == "xk"


def test_extract_key_none_when_absent():
    assert _extract_key(SimpleNamespace(headers={})) is None
