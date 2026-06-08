"""salt keyを使った値の暗号化（PyNaCl SecretBox）。

salt keyはscryptで32バイトの鍵にストレッチされてXSalsa20-Poly1305 SecretBoxに使用され、
暗号文はURL-safe base64でエンコードされます。
scrypt（単純なSHA-256と比べて）は、弱い/人間が選んだsalt keyに対するブルートフォースを
高コスト化します（H9）。salt keyは専用（``BLUELLM_SALT_KEY``）でありマスターキーへの
フォールバックは行わないため、auth keyが漏洩しても暗号化されたconfig値は露出しません。

破壊的変更: 旧SHA-256スキームで生成されたtokenはscryptでは復号できません。
``bluellm encrypt`` を再実行して ``encrypted:`` config値を再生成してください。
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

import nacl.secret

# scryptパラメータ。saltは固定のドメイン区切り文字（秘密ではない）:
# encrypt/decryptのラウンドトリップを保証するため導出は決定論的でなければならず、
# SecretBoxのnonceがメッセージごとのランダム性を提供します。n=2**15は導出を
# 1秒以内に収めつつ約32 MBのコストをかけ、ブルートフォースを支配します。
_SCRYPT_SALT = b"bluellm-secretbox-v1"
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 96 * 1024 * 1024


def get_salt_key() -> Optional[str]:
    """``BLUELLM_SALT_KEY`` から専用のsalt keyを返します。設定されていない場合はNoneを返します。

    salt keyはマスターキーとは意図的に独立しています: 明示的に設定する必要があるため、
    auth（マスター）keyが漏洩しても暗号化されたconfig値は露出しません。
    """
    return os.getenv("BLUELLM_SALT_KEY")


def _box(salt_key: str) -> "nacl.secret.SecretBox":
    """``salt_key`` をscryptで32バイト鍵にストレッチしてSecretBoxを構築します。"""
    key = hashlib.scrypt(
        salt_key.encode(),
        salt=_SCRYPT_SALT,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
        maxmem=_SCRYPT_MAXMEM,
    )
    return nacl.secret.SecretBox(key)


def encrypt_value(value: str, salt_key: str) -> str:
    """``value`` を ``salt_key`` で暗号化し、URL-safe base64 tokenを返します。"""
    encrypted = _box(salt_key).encrypt(value.encode())
    return base64.urlsafe_b64encode(encrypted).decode("utf-8")


def decrypt_value(value: str, salt_key: str) -> str:
    """:func:`encrypt_value` で生成されたURL-safe base64 tokenを復号します。"""
    decoded = base64.urlsafe_b64decode(value.encode("utf-8"))
    return _box(salt_key).decrypt(decoded).decode("utf-8")
