"""CLI エントリポイント。

    python -m bluellm [serve] [--config config.yml] [--host H] [--port P]
    python -m bluellm encrypt [--config config.yml]   # config に記載するシークレットを暗号化する

--config は省略可能。デフォルトでは、現在の作業ディレクトリに関わらず、
プログラムディレクトリ（bluellm パッケージの隣にあるプロジェクトルート）の
config.yml が使用される。--config または BLUELLM_CONFIG 環境変数で上書き可能。
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv


class _HttpxFormatter(logging.Formatter):
    """httpx ログの見た目を整えるフォーマッタ。

    - ``INFO:httpx`` のような詰まった接頭辞を ``INFO: httpx`` 形式に統一する。
    - ``"HTTP/1.1 200 OK"`` の引用符を外してからステータス句を色付けする。
    """

    _RESET = "\x1b[0m"
    _LEVEL_COLORS = {
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[31m",
    }
    _STATUS_COLORS = {
        1: "\x1b[37m",
        2: "\x1b[32m",
        3: "\x1b[33m",
        4: "\x1b[31m",
        5: "\x1b[31m",
    }

    def __init__(self, *, use_colors: bool = True) -> None:
        """フォーマッタを初期化する。"""
        super().__init__("%(levelname)s: %(name)s: %(message)s")
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """ログレベル接頭辞とステータス句を統一形式で整形する。"""
        message = super().format(record)
        message = re.sub(r"\bINFO:\s+", "INFO: ", message)
        message = re.sub(
            r'"(HTTP/\d\.\d \d{3} [A-Z][A-Z ]*)"', r"\1", message
        )
        if not self._use_colors:
            return message

        level_color = self._LEVEL_COLORS.get(record.levelno)
        if level_color:
            prefix = f"{record.levelname}:"
            message = message.replace(prefix, f"{level_color}{prefix}{self._RESET}", 1)

        def _paint_status(match: re.Match[str]) -> str:
            code = int(match.group(1))
            status = match.group(0)
            color = self._STATUS_COLORS.get(code // 100)
            if color is None:
                return status
            return f"{color}{status}{self._RESET}"

        return re.sub(r"\b([1-5]\d\d) [A-Z][A-Z ]*\b", _paint_status, message)



def _default_config_path() -> str:
    """プログラムディレクトリ（プロジェクトルート）の config.yml を返す。

    プロジェクトルートは pyproject.toml を含む最も近い祖先ディレクトリとして決定するため、
    現在の作業ディレクトリに関わらず src レイアウト（src/bluellm/）でも正しく解決される。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return str(parent / "config.yml")
    return str(here.parent / "config.yml")


_DEFAULT_CONFIG = _default_config_path()


def _is_loopback_host(host: str) -> bool:
    """``host`` がローカルループバックインターフェースのみにバインドする場合 True を返す。

    一般的な表記（``localhost``、``127.0.0.0/8``、``::1``）を網羅する。
    それ以外の値（``0.0.0.0`` を含む）は外部からアクセス可能とみなす。
    """
    normalized = host.lower().rstrip(".")
    if normalized in ("localhost", "::1"):
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _cmd_serve(args: argparse.Namespace) -> int:
    """``serve`` サブコマンドを実行する: config を読み込み uvicorn サーバーを起動する。"""
    import uvicorn
    import uvicorn.config

    from bluellm import redaction
    from bluellm.config import load_config
    from bluellm.server import create_app

    # ルートロガーは httpx ログ書式を統一し、uvicorn は独自 log_config で統一する。
    logging.basicConfig(level=logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(_HttpxFormatter())

    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"]["use_colors"] = True
    log_config["formatters"]["access"]["use_colors"] = True
    log_config["formatters"]["access"]["fmt"] = (
        "%(levelprefix)s %(client_addr)s - %(request_line)s %(status_code)s"
    )

    config = load_config(args.config)
    redaction.install()  # load_config でシークレットが登録された後に呼び出す

    host = args.host or config.general_settings.host
    port = args.port or config.general_settings.port

    if not config.general_settings.master_key:
        if not _is_loopback_host(host):
            print(
                f"起動を拒否しました: master_key が設定されておらず（プロキシが未認証状態になります）、"
                f"host {host!r} はループバックインターフェースではありません。"
                f"generals.key / BLUELLM_MASTER_KEY を設定するか、127.0.0.1 にバインドしてください。",
                file=sys.stderr,
            )
            return 1
        logging.getLogger("bluellm").warning(
            "master_key が未設定です: プロキシは未認証状態で動作します。"
            "信頼されたループバックインターフェース上でのみ使用してください。"
        )

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=log_config)
    return 0


def _cmd_encrypt(args: argparse.Namespace) -> int:
    """``encrypt`` サブコマンドを実行する: 値に対して ``encrypted:`` token を出力する。

    salt キーは ``generals.salt``（--config 経由）または ``BLUELLM_SALT_KEY`` から取得する。
    master key からは取得しない（H8）。
    """
    from bluellm import crypto

    from bluellm.config import validate_salt_key

    salt_key = None
    if args.config and os.path.exists(args.config):
        from bluellm.config import load_config

        cfg = load_config(args.config)
        salt_key = cfg.general_settings.salt_key
    # salt キーは専用のものであり、master key からは導出しない（H8）:
    # generals.salt または BLUELLM_SALT_KEY からのみ取得する。
    salt_key = salt_key or crypto.get_salt_key()
    if not salt_key:
        print(
            "salt キーが利用できません。BLUELLM_SALT_KEY を設定するか、"
            "generals.salt を含む config を --config で指定してください。",
            file=sys.stderr,
        )
        return 1
    # 復号パス（H9）と同じ最低強度要件を適用し、弱い salt キーで
    # ブルートフォース可能な encrypted: token が生成されないようにする。
    validate_salt_key(salt_key)

    value = getpass.getpass("Value to encrypt: ")
    if not value:
        print("Empty value; nothing to do.", file=sys.stderr)
        return 1
    token = crypto.encrypt_value(value, salt_key)
    print(f"encrypted:{token}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント: 引数をパースして serve/encrypt サブコマンドにディスパッチする。"""
    load_dotenv()
    parser = argparse.ArgumentParser(prog="bluellm")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="run the proxy server")
    p_serve.add_argument("--config", default=os.getenv("BLUELLM_CONFIG") or _DEFAULT_CONFIG)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=_cmd_serve)

    p_enc = sub.add_parser("encrypt", help="encrypt a secret value for the config")
    p_enc.add_argument("--config", default=os.getenv("BLUELLM_CONFIG") or _DEFAULT_CONFIG)
    p_enc.set_defaults(func=_cmd_encrypt)

    # サブコマンドが指定されない場合は 'serve' をデフォルトとする
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["serve"]
    elif raw[0] not in ("serve", "encrypt", "-h", "--help"):
        raw = ["serve", *raw]

    args = parser.parse_args(raw)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
