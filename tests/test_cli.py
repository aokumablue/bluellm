"""CLI (`python -m bluellm`) tests covering serve/encrypt/main and helpers."""

from types import SimpleNamespace

import pytest
import uvicorn
from helpers import CONFIG_TEMPLATE, SALT_KEY

import bluellm.__main__ as cli
from bluellm import crypto


def test_default_config_path_returns_config_yml():
    assert cli._default_config_path().endswith("config.yml")


def test_default_config_path_fallback_without_pyproject(monkeypatch):
    # No ancestor has pyproject.toml -> fall back to the package-dir default.
    monkeypatch.setattr(cli.Path, "is_file", lambda self: False)
    assert cli._default_config_path().endswith("config.yml")


def test_cmd_serve_starts_uvicorn_and_warns_when_open(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("AZURE_API_KEY", "azkey")
    monkeypatch.delenv("BLUELLM_MASTER_KEY", raising=False)  # -> unauthenticated warning
    cfg = tmp_path / "config.yml"
    cfg.write_text(CONFIG_TEMPLATE)
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))
    args = SimpleNamespace(config=str(cfg), host=None, port=None)
    with caplog.at_level("WARNING"):
        assert cli._cmd_serve(args) == 0
    assert captured["host"] == "127.0.0.1"
    assert any("UNAUTHENTICATED" in r.getMessage() for r in caplog.records)


def test_is_loopback_host():
    assert cli._is_loopback_host("127.0.0.1") is True
    assert cli._is_loopback_host("localhost") is True
    assert cli._is_loopback_host("::1") is True
    # case / trailing-dot normalization (HIGH-2): these must not bypass the check
    assert cli._is_loopback_host("LOCALHOST") is True
    assert cli._is_loopback_host("Localhost.") is True
    assert cli._is_loopback_host("127.0.0.1.") is True
    assert cli._is_loopback_host("0.0.0.0") is False
    assert cli._is_loopback_host("10.0.0.5") is False
    assert cli._is_loopback_host("example.com") is False


def test_cmd_serve_refuses_unauthenticated_non_loopback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AZURE_API_KEY", "azkey")
    monkeypatch.delenv("BLUELLM_MASTER_KEY", raising=False)  # unauthenticated
    cfg = tmp_path / "config.yml"
    cfg.write_text(CONFIG_TEMPLATE)
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(kw))
    args = SimpleNamespace(config=str(cfg), host="0.0.0.0", port=None)
    assert cli._cmd_serve(args) == 1
    assert "Refusing to start" in capsys.readouterr().err
    assert started == {}  # uvicorn never started


def test_cmd_serve_allows_unauthenticated_loopback(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "azkey")
    monkeypatch.delenv("BLUELLM_MASTER_KEY", raising=False)
    cfg = tmp_path / "config.yml"
    cfg.write_text(CONFIG_TEMPLATE)
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(kw))
    args = SimpleNamespace(config=str(cfg), host="127.0.0.1", port=None)
    assert cli._cmd_serve(args) == 0
    assert started["host"] == "127.0.0.1"


def test_cmd_encrypt_with_env_salt(monkeypatch, capsys):
    monkeypatch.setenv("BLUELLM_SALT_KEY", SALT_KEY)
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a, **k: "secret-value")
    args = SimpleNamespace(config=None)
    assert cli._cmd_encrypt(args) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("encrypted:")
    token = out[len("encrypted:") :]
    assert crypto.decrypt_value(token, SALT_KEY) == "secret-value"


def test_cmd_encrypt_rejects_short_salt(monkeypatch):
    # H9: a too-short salt key must be rejected before producing a token.
    monkeypatch.setenv("BLUELLM_SALT_KEY", "short")
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a, **k: "v")
    with pytest.raises(ValueError):
        cli._cmd_encrypt(SimpleNamespace(config=None))


def test_cmd_encrypt_salt_from_config(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BLUELLM_SALT_KEY", raising=False)
    cfg = tmp_path / "config.yml"
    cfg.write_text(
        'models:\n  - name: "*"\n    params:\n      model: azure/x\n'
        "      key: plain\n"
        f"generals:\n  salt: {SALT_KEY}\n"
    )
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a, **k: "v")
    assert cli._cmd_encrypt(SimpleNamespace(config=str(cfg))) == 0
    out = capsys.readouterr().out.strip()
    assert crypto.decrypt_value(out[len("encrypted:") :], SALT_KEY) == "v"


def test_cmd_encrypt_no_salt_returns_1(monkeypatch, capsys):
    monkeypatch.delenv("BLUELLM_SALT_KEY", raising=False)
    assert cli._cmd_encrypt(SimpleNamespace(config=None)) == 1
    assert "No salt key" in capsys.readouterr().err


def test_cmd_encrypt_empty_value_returns_1(monkeypatch, capsys):
    monkeypatch.setenv("BLUELLM_SALT_KEY", SALT_KEY)
    monkeypatch.setattr(cli.getpass, "getpass", lambda *a, **k: "")
    assert cli._cmd_encrypt(SimpleNamespace(config=None)) == 1
    assert "Empty value" in capsys.readouterr().err


def test_main_dispatches_serve_with_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_cmd_serve", lambda args: seen.update(port=args.port) or 0)
    assert cli.main(["serve", "--port", "1234"]) == 0
    assert seen["port"] == 1234


def test_main_defaults_to_serve(monkeypatch):
    monkeypatch.setattr(cli, "_cmd_serve", lambda args: 0)
    assert cli.main([]) == 0


def test_main_prepends_serve_for_bare_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "_cmd_serve", lambda args: captured.update(host=args.host) or 0)
    assert cli.main(["--host", "0.0.0.0"]) == 0
    assert captured["host"] == "0.0.0.0"


def test_main_dispatches_encrypt(monkeypatch):
    monkeypatch.setattr(cli, "_cmd_encrypt", lambda args: 7)
    assert cli.main(["encrypt"]) == 7
