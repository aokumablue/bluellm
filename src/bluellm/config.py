"""設定ローダー（YAML）。

スキーマ:

    models:
      - name: "*"
        params:
          model: azure/<deployment>
          endpoint: https://...
          key: os.environ/AZURE_API_KEY           # または encrypted:<b64> もしくは平文
          version: "2025-01-01-preview"
          extra_params: {reasoning_effort: medium} # 省略可、リクエストにマージされる

    generals:
      key: os.environ/BLUELLM_MASTER_KEY          # クライアントが提示する master key
      salt: os.environ/BLUELLM_SALT_KEY           # encrypted: 値の復号に使う key

上流が非対応として拒否した params は自動的に除去されてリトライされるため、
モデルごとの除外リストは不要。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bluellm import crypto
from bluellm.redaction import register_secret
from bluellm.reliability import DEFAULT_RETRY_POLICY, RetryPolicy


class _RawModelEntry(BaseModel):
    """``models`` エントリ 1 件の生 YAML スキーマ（params は自由形式のまま保持）。"""

    model_config = ConfigDict(extra="forbid")
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class _RawConfig(BaseModel):
    """前処理で一括検証する生 YAML スキーマ（不正な設定を早期にエラーとする）（M10）。"""

    model_config = ConfigDict(extra="forbid")
    models: List[_RawModelEntry] = Field(default_factory=list)
    generals: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ModelConfig:
    """1 つのモデルエントリの解決済み設定（provider、deployment、認証情報、信頼性）。"""

    model_name: str
    provider: str
    deployment: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry: RetryPolicy = DEFAULT_RETRY_POLICY
    fallback_to: Optional[str] = None


@dataclass
class GeneralSettings:
    """サーバー全体の設定: 認証・暗号鍵、host/port、ボディ上限、HTTP 境界防御。

    runaway ガードは単一グローバルのトークンバケットで暴走のみ抑止する（寛容な
    既定値で常時 ON・単一ユーザーを妨げない）、allowlist は既定空＝全許可
    （既存挙動と等価）。
    """

    master_key: Optional[str] = None
    salt_key: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 4000
    max_request_body_mb: int = 10
    runaway_guard_rps: float = 200.0
    allowlist_cidrs: List[str] = field(default_factory=list)
    otel_disabled: bool = False
    otel_endpoint: str = "http://127.0.0.1:4318/v1/traces"
    otel_service_name: str = "bluellm"


@dataclass
class Config:
    """完全解決済みのプロキシ設定（モデルリスト＋全体設定）。"""

    model_list: List[ModelConfig]
    general_settings: GeneralSettings


def _resolve_secret(value: Any, salt_key: Optional[str]) -> Any:
    """os.environ/ および encrypted: の間接参照を解決し、結果を登録する。"""
    if not isinstance(value, str):
        return value
    if value.startswith("os.environ/"):
        resolved = os.environ.get(value[len("os.environ/") :])
    elif value.startswith("encrypted:"):
        if not salt_key:
            raise ValueError(
                "Encountered an 'encrypted:' value but no salt_key/master_key is set"
            )
        resolved = crypto.decrypt_value(value[len("encrypted:") :], salt_key)
    else:
        resolved = value
    register_secret(resolved)
    return resolved


def _validate_api_base(api_base: Any, *, allow_local: bool = False) -> Any:
    """SSRF 対策として、設定された上流 base URL を検証する。

    2 つのチェック（M10）:
    1. ``http``/``https`` スキームのみ許可。その他のスキーム（``file:``、
       ``gopher:`` など）は拒否し、改ざんされた設定がプロキシを任意の
       ローカルリソースに向けられないようにする。
    2. ループバックまたはリンクローカル範囲の IP リテラルホスト（後者は
       クラウドメタデータ endpoint ``169.254.169.254`` を含む）は拒否し、
       改ざんされた設定がプロキシを SSRF ガジェットとしてホストローカル
       サービスに悪用できないようにする。ホスト名とオンプレ/プライベート IP は
       通過させる（オペレーター信頼の正当な endpoint）。

    ``allow_local=True`` の場合のみループバック/リンクローカルを許容する。
    ローカル Ollama（既定 ``http://localhost:11434/v1``）を上流として
    使う正当なケースのための明示的な carve-out であり、SSRF ガード本体は
    維持する（``allow_local`` でないエントリは従来どおり拒否）。

    ``None``/空値は通過させる（OpenAI SDK がデフォルト base URL を使用）。
    """
    if not api_base:
        return api_base
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(str(api_base))
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"endpoint must use an http/https URL, got {scheme or 'no'} scheme"
        )
    host = parsed.hostname or ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None  # ホスト名であり IP リテラルではない
    if not allow_local and ip is not None and (ip.is_loopback or ip.is_link_local):
        raise ValueError(
            f"endpoint host {host!r} is a loopback/link-local address; refusing "
            "to use it as an upstream (SSRF / metadata-endpoint defense)"
        )
    return api_base


def _validate_allowlist_cidrs(value: Any) -> List[str]:
    """``generals.allowlist_cidrs`` を検証して CIDR 文字列リストに正規化する。

    各要素を ``ipaddress.ip_network`` でパースして不正な CIDR を起動時に弾く
    （誤設定による全拒否や無効ルールの黙殺を防ぐ）。``None``/未指定は空リスト。
    """
    if not value:
        return []
    if not isinstance(value, list):
        raise ValueError("generals.allowlist_cidrs must be a list of CIDR strings")
    import ipaddress

    cidrs: List[str] = []
    for entry in value:
        try:
            ipaddress.ip_network(str(entry), strict=False)
        except ValueError as e:
            raise ValueError(f"invalid CIDR in allowlist_cidrs: {entry!r}") from e
        cidrs.append(str(entry))
    return cidrs


_MIN_MASTER_KEY_LENGTH = 16


def _validate_master_key(master_key: Any) -> Any:
    """推測攻撃に耐えられないほど短い master key を拒否する（H9）。

    弱い/人が選んだ master key は定数時間比較に関わらずブルートフォースされるため、
    少なくとも ``_MIN_MASTER_KEY_LENGTH`` 文字を要求する。
    ``None``（認証なしのローカルモード）は通過させる。
    """
    if master_key and len(str(master_key)) < _MIN_MASTER_KEY_LENGTH:
        raise ValueError(
            "generals.key (master key) is too short; use at least "
            f"{_MIN_MASTER_KEY_LENGTH} characters (e.g. `openssl rand -base64 24`)"
        )
    return master_key


def validate_salt_key(salt_key: Any) -> Any:
    """短すぎる暗号化 salt key を拒否する（H9）。

    scrypt は固定（非秘密）のドメインセパレーターを使用するため、弱い salt key が
    攻撃者と暗号化済み設定値の間の唯一の壁となる。短いと scrypt のコストにも
    関わらずブルートフォース可能になる。``None`` は通過させる（暗号化未設定）。
    """
    if salt_key and len(str(salt_key)) < _MIN_MASTER_KEY_LENGTH:
        raise ValueError(
            "salt key (generals.salt / BLUELLM_SALT_KEY) is too short; use at "
            f"least {_MIN_MASTER_KEY_LENGTH} characters"
        )
    return salt_key


def _resolve_retry_policy(params: Dict[str, Any]) -> RetryPolicy:
    """``params.retry`` から :class:`RetryPolicy` を構築する（未指定は SDK 既定相当）。

    キー未指定の項目は :data:`DEFAULT_RETRY_POLICY` の値を引き継ぐため、
    retry を一切書かないモデルは既存挙動（SDK の自動リトライ相当）を維持する。
    """
    retry_raw = params.get("retry") or {}
    return RetryPolicy(
        max_attempts=int(
            retry_raw.get("max_attempts", DEFAULT_RETRY_POLICY.max_attempts)
        ),
        initial_backoff_ms=int(
            retry_raw.get("initial_backoff_ms", DEFAULT_RETRY_POLICY.initial_backoff_ms)
        ),
        max_backoff_ms=int(
            retry_raw.get("max_backoff_ms", DEFAULT_RETRY_POLICY.max_backoff_ms)
        ),
        jitter_ratio=float(
            retry_raw.get("jitter_ratio", DEFAULT_RETRY_POLICY.jitter_ratio)
        ),
    )


def _split_provider(model: str) -> tuple[str, str]:
    """``provider/deployment`` をペアに分割する。provider が省略された場合は openai をデフォルトとする。"""
    if "/" in model:
        provider, deployment = model.split("/", 1)
        return provider, deployment
    return "openai", model


def load_config(path: str) -> Config:
    """``path`` の YAML 設定を読み込み、:class:`Config` に解決する。

    ``os.environ/`` および ``encrypted:`` の間接参照を解決し、
    endpoint スキームを検証する。モデルリストが空の場合は ``ValueError`` を送出する。
    path は解決済みとし（M10）、生の YAML 形式はスキーマ検証を事前に行うことで
    不正な設定が不可解な KeyError ではなく明確なエラーで失敗するようにする。
    """
    resolved_path = str(Path(path).resolve())
    with open(resolved_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        _RawConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"invalid config schema: {e}") from e

    gs_raw = raw.get("generals") or {}
    # master key / salt key は os.environ/ と平文をサポートする（encrypted: は不可）。
    master_key = _validate_master_key(_resolve_secret(gs_raw.get("key"), salt_key=None))
    salt_key = validate_salt_key(_resolve_secret(gs_raw.get("salt"), salt_key=None))
    # salt key は専用であり、master key へのフォールバックは行わない。
    effective_salt = validate_salt_key(salt_key or crypto.get_salt_key())

    general_settings = GeneralSettings(
        master_key=master_key,
        salt_key=salt_key,
        host=gs_raw.get("host", "127.0.0.1"),
        port=int(gs_raw.get("port", 4000)),
        max_request_body_mb=int(gs_raw.get("max_request_body_mb", 10)),
        runaway_guard_rps=float(gs_raw.get("runaway_guard_rps", 200.0)),
        allowlist_cidrs=_validate_allowlist_cidrs(gs_raw.get("allowlist_cidrs")),
        otel_disabled=bool(gs_raw.get("otel_disabled", False)),
        otel_endpoint=gs_raw.get(
            "otel_endpoint", "http://127.0.0.1:4318/v1/traces"
        ),
        otel_service_name=gs_raw.get("otel_service_name", "bluellm"),
    )

    model_list: List[ModelConfig] = []
    for entry in raw.get("models") or []:
        params = entry.get("params") or {}
        model = params.get("model", entry.get("name", ""))
        provider, deployment = _split_provider(model)
        timeout_raw = params.get("request_timeout_seconds")
        model_list.append(
            ModelConfig(
                model_name=entry["name"],
                provider=provider,
                deployment=deployment,
                api_base=_validate_api_base(
                    _resolve_secret(params.get("endpoint"), effective_salt),
                    allow_local=(provider == "ollama"),
                ),
                api_key=_resolve_secret(params.get("key"), effective_salt),
                api_version=params.get("version"),
                extra_params=dict(params.get("extra_params") or {}),
                timeout=float(timeout_raw) if timeout_raw is not None else None,
                retry=_resolve_retry_policy(params),
                fallback_to=params.get("fallback_to"),
            )
        )

    if not model_list:
        raise ValueError("config has an empty 'models' list")

    return Config(model_list=model_list, general_settings=general_settings)
