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
import sys
from pathlib import Path

from dotenv import load_dotenv


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

    from bluellm import redaction
    from bluellm.config import load_config
    from bluellm.server import create_app

    logging.basicConfig(level=logging.INFO)
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
    uvicorn.run(app, host=host, port=port, log_level="info")
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
