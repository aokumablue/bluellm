import pytest
from helpers import AZURE_KEY, MASTER_KEY, SALT_KEY

from bluellm import crypto
from bluellm.config import load_config
from bluellm.router import Router

SCHEMA = """
models:
  - name: "*"
    params:
      model: azure/gpt-5.4
      endpoint: https://example.openai.azure.com
      key: {key}
      version: "2025-01-01-preview"
generals:
  key: os.environ/BLUELLM_MASTER_KEY
  salt: os.environ/BLUELLM_SALT_KEY
  port: 4242
"""


def _write(tmp_path, key_value):
    p = tmp_path / "config.yml"
    p.write_text(SCHEMA.format(key=key_value))
    return str(p)


def test_new_schema_and_env_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    cfg = load_config(_write(tmp_path, "os.environ/AZURE_API_KEY"))

    mc = cfg.model_list[0]
    assert mc.model_name == "*"
    assert mc.provider == "azure"
    assert mc.deployment == "gpt-5.4"
    assert mc.api_base == "https://example.openai.azure.com"
    assert mc.api_key == AZURE_KEY
    assert mc.api_version == "2025-01-01-preview"
    assert cfg.general_settings.master_key == MASTER_KEY
    assert cfg.general_settings.port == 4242


def test_encrypted_value_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    monkeypatch.setenv("BLUELLM_SALT_KEY", SALT_KEY)
    token = crypto.encrypt_value("secret-azure-key", SALT_KEY)
    cfg = load_config(_write(tmp_path, f"encrypted:{token}"))
    assert cfg.model_list[0].api_key == "secret-azure-key"


def test_wildcard_routing(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    cfg = load_config(_write(tmp_path, "os.environ/AZURE_API_KEY"))
    router = Router(cfg)
    # Claude Code's model name routes through the "*" catch-all.
    assert router.resolve("claude-sonnet-4-20250514").deployment == "gpt-5.4"


def test_encrypted_value_requires_dedicated_salt_key(tmp_path, monkeypatch):
    # H8: the master key must not silently double as the encryption salt key.
    # With no BLUELLM_SALT_KEY configured, an encrypted: value cannot be
    # decrypted and config load must fail loudly (key separation), rather than
    # falling back to decrypting with the master key.
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    monkeypatch.delenv("BLUELLM_SALT_KEY", raising=False)
    token = crypto.encrypt_value("secret-azure-key", MASTER_KEY)
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, f"encrypted:{token}"))


def test_short_master_key_rejected(tmp_path, monkeypatch):
    # H9: a too-short master key is brute-forceable; reject it at load.
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", "short")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "os.environ/AZURE_API_KEY"))


def test_short_salt_key_rejected(tmp_path, monkeypatch):
    # H9: a too-short salt key (the only guard on encrypted values) is rejected.
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    monkeypatch.setenv("BLUELLM_SALT_KEY", "short")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "os.environ/AZURE_API_KEY"))


def test_scrypt_encrypt_decrypt_roundtrip():
    # H9: scrypt-stretched key still round-trips encrypt/decrypt.
    token = crypto.encrypt_value("a-secret", SALT_KEY)
    assert crypto.decrypt_value(token, SALT_KEY) == "a-secret"


def test_endpoint_optional_passes_through(tmp_path, monkeypatch):
    # M10: a config with no endpoint is valid (the OpenAI SDK then uses its
    # default base URL); _validate_api_base must let None through.
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    no_endpoint = """
models:
  - name: "*"
    params:
      model: azure/gpt-5.4
      key: plaintext-key
      version: "2025-01-01-preview"
generals:
  key: os.environ/BLUELLM_MASTER_KEY
"""
    p = tmp_path / "config.yml"
    p.write_text(no_endpoint)
    cfg = load_config(str(p))
    assert cfg.model_list[0].api_base is None


def test_endpoint_metadata_ip_rejected(tmp_path, monkeypatch):
    # M10: a link-local / metadata endpoint (169.254.169.254) is an SSRF gadget.
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    bad = SCHEMA.replace(
        "https://example.openai.azure.com", "http://169.254.169.254/latest/meta-data"
    ).format(key="plaintext-key")
    p = tmp_path / "config.yml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_config(str(p))


def test_endpoint_loopback_ip_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    bad = SCHEMA.replace(
        "https://example.openai.azure.com", "http://127.0.0.1:8080"
    ).format(key="plaintext-key")
    p = tmp_path / "config.yml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_config(str(p))


def test_ollama_loopback_endpoint_allowed(tmp_path, monkeypatch):
    # Ollama runs locally; an ollama/* model may point at a loopback endpoint.
    # The SSRF guard's loopback rejection is carved out only for ollama.
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    cfg_text = """
models:
  - name: "*"
    params:
      model: ollama/llama3.3
      endpoint: http://localhost:11434/v1
generals:
  key: os.environ/BLUELLM_MASTER_KEY
"""
    p = tmp_path / "config.yml"
    p.write_text(cfg_text)
    cfg = load_config(str(p))
    assert cfg.model_list[0].provider == "ollama"
    assert cfg.model_list[0].api_base == "http://localhost:11434/v1"


def test_non_ollama_loopback_endpoint_still_rejected(tmp_path, monkeypatch):
    # The loopback carve-out must NOT relax the SSRF guard for non-ollama
    # providers (e.g. an openai entry pointed at loopback stays rejected).
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    cfg_text = """
models:
  - name: "*"
    params:
      model: openai/gpt-5.4
      endpoint: http://127.0.0.1:11434/v1
      key: plaintext-key
generals:
  key: os.environ/BLUELLM_MASTER_KEY
"""
    p = tmp_path / "config.yml"
    p.write_text(cfg_text)
    with pytest.raises(ValueError):
        load_config(str(p))


def test_malformed_config_schema_rejected(tmp_path, monkeypatch):
    # M10: an unknown top-level key (e.g. a typo) fails loudly at load.
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    p = tmp_path / "config.yml"
    p.write_text("modelz:\n  - name: x\ngenerals: {}\n")  # "modelz" typo
    with pytest.raises(ValueError):
        load_config(str(p))


def test_config_path_is_resolved(tmp_path, monkeypatch):
    # M10: a non-normalized path (with ..) still loads via Path.resolve().
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    sub = tmp_path / "sub"
    sub.mkdir()
    real = _write(tmp_path, "os.environ/AZURE_API_KEY")
    weird = str(sub / ".." / "config.yml")
    cfg = load_config(weird)
    assert cfg.model_list[0].deployment == "gpt-5.4"
    assert real  # the canonical path was written


def test_endpoint_must_use_http_scheme(tmp_path, monkeypatch):
    # M10: a tampered config must not point the proxy at a non-http(s) URL
    # (SSRF / local-resource access defense).
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    bad = SCHEMA.replace(
        "https://example.openai.azure.com", "file:///etc/passwd"
    ).format(key="plaintext-key")
    p = tmp_path / "config.yml"
    p.write_text(bad)
    with pytest.raises(ValueError):
        load_config(str(p))


def test_max_request_body_mb_default(tmp_path, monkeypatch):
    # M-4: generals.max_request_body_mb defaults to 10 when not specified in YAML.
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    cfg = load_config(_write(tmp_path, "os.environ/AZURE_API_KEY"))
    assert cfg.general_settings.max_request_body_mb == 10


def test_max_request_body_mb_custom(tmp_path, monkeypatch):
    # M-4: generals.max_request_body_mb is resolved from YAML when specified.
    monkeypatch.setenv("AZURE_API_KEY", AZURE_KEY)
    monkeypatch.setenv("BLUELLM_MASTER_KEY", MASTER_KEY)
    custom = SCHEMA.rstrip() + "\n  max_request_body_mb: 5\n"
    p = tmp_path / "config.yml"
    p.write_text(custom.format(key="os.environ/AZURE_API_KEY"))
    cfg = load_config(str(p))
    assert cfg.general_settings.max_request_body_mb == 5
