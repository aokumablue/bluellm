"""openai SDK を使用した OpenAI / Azure (Microsoft Foundry) provider。

どちらも Chat Completions を使用して同一のレスポンス形式を返すため、
単一の実装でカバーできる。設定済みモデルの ``provider/`` プレフィックスで
クライアントの種類を選択する。
"""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Optional, Set

from openai import AsyncAzureOpenAI, AsyncOpenAI, BadRequestError

from bluellm.config import ModelConfig
from bluellm.providers.base import BaseProvider

logger = logging.getLogger("bluellm")

# 未サポート param の自動削除時に絶対に削除しないパラメータ。
_PROTECTED_PARAMS = {"model", "messages", "max_completion_tokens"}

# OpenAI のワイヤースキーマに存在しないメッセージキー。送信前に除去する。
_NON_OPENAI_MESSAGE_KEYS = (
    "thinking_blocks",
    "reasoning_content",
    "reasoning_items",
    "cache_control",
    "provider_specific_fields",
)


def _strip_cache_control(obj: Any) -> Any:
    """dict/list ツリーから Anthropic の ``cache_control`` キーを再帰的に削除する。

    translator はメッセージコンテンツパーツやツールパラメータ等に、受信モデル名が
    Claude に見える場合に ``cache_control`` を付加する。bluellm の全上流は
    Chat Completions を使用しており、そのフィールドを拒否するため、
    リクエスト送信前にここで除去する。
    """
    if isinstance(obj, dict):
        return {
            k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"
        }
    if isinstance(obj, list):
        return [_strip_cache_control(v) for v in obj]
    return obj


class OpenAILikeProvider(BaseProvider):
    """openai SDK を使用した provider（Azure OpenAI または通常の OpenAI）。"""

    def __init__(self, model_config: ModelConfig) -> None:
        """``model_config`` に対応する非同期クライアントを構築する（Azure または OpenAI）。"""
        self.mc = model_config
        self._client = self._build_client()

    def _build_client(self):
        """設定から AsyncAzureOpenAI / AsyncOpenAI クライアントを構築する。"""
        mc = self.mc
        if mc.provider == "azure":
            # Azure OpenAI の ``.../openai/v1`` は正式な OpenAI-compatible endpoint なので、
            # OpenAI client をそのまま使う。
            if mc.api_base and "/openai/v1" in mc.api_base:
                return AsyncOpenAI(api_key=mc.api_key, base_url=mc.api_base)
            params: Dict[str, Any] = {
                "api_key": mc.api_key,
                "api_version": mc.api_version,
            }
            # /openai/deployments を含む api_base は完全な base_url であり、
            # azure_endpoint ではない。
            if mc.api_base and "/openai/deployments" in mc.api_base:
                params["base_url"] = mc.api_base
            else:
                params["azure_endpoint"] = mc.api_base
            return AsyncAzureOpenAI(**params)
        return AsyncOpenAI(api_key=mc.api_key, base_url=mc.api_base or None)

    def _prepare(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """翻訳済みリクエストをこの deployment の Chat Completions 呼び出しに適合させる。

        deployment 名を設定し、Anthropic 専用パラメータを OpenAI 相当のものに
        書き換え（max_tokens、stop_sequences、thinking）、Chat Completions に
        対応するものがないパラメータを削除し（top_k、stream 制御）、
        設定済みの extra_params をマージする。
        """
        req = dict(request)
        req["model"] = self.mc.deployment

        # Anthropic は常に max_tokens を送信するが、現行の OpenAI/Azure chat
        # completions は max_completion_tokens を期待する。
        if "max_tokens" in req and "max_completion_tokens" not in req:
            req["max_completion_tokens"] = req.pop("max_tokens")

        # Anthropic の `stop_sequences` は Chat Completions では `stop` という名前。
        # Azure/OpenAI は `stop` に list[str] を期待するため、上流の型ゆらぎ
        # （string / list[non-str] / dict など）を正規化する。
        stop_sequences = req.pop("stop_sequences", None)
        if stop_sequences is not None:
            if isinstance(stop_sequences, list) and all(isinstance(x, str) for x in stop_sequences):
                # 上流（Claude互換クライアント）が `stop` トップレベルを既に指定している場合
                # は上書きしない（＝既存値を優先）
                req.setdefault("stop", stop_sequences)
            else:
                # `stop_sequences` が不正（list[str] ではない）場合は `stop` も落とす
                req.pop("stop", None)
        # stop_sequences が無い場合は `stop` を維持する（上流が正しい `stop` を指定している
        # 可能性を残す）。

        stop = req.get("stop")
        if stop is not None:
            # OpenAI/Azure は `stop` に list[str] を期待
            if isinstance(stop, list) and all(isinstance(x, str) for x in stop):
                pass
            else:
                req.pop("stop", None)

        # Anthropic の `top_k` は Chat Completions に相当するものがないため事前に削除する。
        # こうしないと、acreate の自動削除リトライが1往復かけて復旧することになる。
        req.pop("top_k", None)

        # Anthropic の `context_management` は現状の bluellm では入力ポリフィルされず、
        # OpenAI/Azure Chat Completions にそのまま渡すと SDK が TypeError で失敗するため削除する。
        req.pop("context_management", None)

        # Anthropic の `thinking` は Chat Completions に相当するものがない。translator は
        # Claude 以外のモデル名に対してのみ `reasoning_effort` に変換するが、
        # 受信する claude-* リクエスト名だとそのパスをスキップして生の `thinking` ブロックを
        # 出力してしまう。実際のワイヤーフォーマット（Chat Completions）が判明している
        # ここで書き換える。
        if "thinking" in req:
            thinking = req.pop("thinking")
            if "reasoning_effort" not in req and isinstance(thinking, dict):
                from bluellm.translation import BlueLLMMessagesAdapter

                # M1: build_reasoning_effort_param は thinking の summary を保持する
                # （detailed/auto）。bare effort 変換だと claude-* リクエストパスで
                # それが失われてしまう。
                effort = BlueLLMMessagesAdapter.build_reasoning_effort_param(
                    thinking, req.get("output_config")
                )
                if effort:
                    req["reasoning_effort"] = effort

        req.pop("stream", None)
        req.pop("stream_options", None)

        messages = req.get("messages")
        if isinstance(messages, list):
            req["messages"] = [self._sanitize_message(m) for m in messages]

        # tool のトップレベルにも Anthropic の cache_control が付いている場合がある。
        tools = req.get("tools")
        if isinstance(tools, list):
            req["tools"] = [_strip_cache_control(t) for t in tools]

        for key, value in self.mc.extra_params.items():
            req.setdefault(key, value)

        return req

    @staticmethod
    def _sanitize_message(message: Any) -> Any:
        """メッセージからトップレベルおよびネストされた cache_control の Anthropic 専用キーを除去する。"""
        if not isinstance(message, dict):
            return message
        # _NON_OPENAI_MESSAGE_KEYS でメッセージトップレベルの Anthropic 専用キーを削除し、
        # その後 _strip_cache_control でコンテンツパーツ dict（text / image / tool_result ブロック）
        # の内部にネストされた cache_control を除去する。
        cleaned = {k: v for k, v in message.items() if k not in _NON_OPENAI_MESSAGE_KEYS}
        return _strip_cache_control(cleaned)

    async def acreate(self, request: Dict[str, Any], *, stream: bool):
        """上流を呼び出し、未サポートパラメータを自動削除してリトライする。

        非ストリーム時は ChatCompletion を、ストリーム時は整形済み非同期チャンク
        イテレータを返す。BadRequestError で未サポートパラメータが指摘された場合、
        そのパラメータを削除し（保護対象でなければ）呼び出しをリトライする。
        """
        # モデルが拒否した未サポートパラメータを自動削除してリトライする。
        # モデルごとの対応表は不要。
        prepared = self._prepare(request)
        # debug 用: upsteam が渡した候補パラメータの存在/型記録
        logger.debug(
            "Prepared request keys/types: stop=%r (%s) stop_sequences=%r (%s) thinking=%r (%s) reasoning_effort=%r (%s) stream=%r stream_options=%r",
            type(prepared.get("stop")).__name__,
            "present" if "stop" in prepared else None,
            type(prepared.get("stop_sequences")).__name__ if "stop_sequences" in prepared else None,
            "present" if "stop_sequences" in prepared else None,
            type(prepared.get("thinking")).__name__ if "thinking" in prepared else None,
            "present" if "thinking" in prepared else None,
            type(prepared.get("reasoning_effort")).__name__ if "reasoning_effort" in prepared else None,
            "present" if "reasoning_effort" in prepared else None,
            "present" if "stream" in prepared else None,
            "present" if "stream_options" in prepared else None,
        )
        dropped: Set[str] = set()
        while True:
            try:
                if stream:
                    openai_stream = await self._client.chat.completions.create(
                        **prepared, stream=True, stream_options={"include_usage": True}
                    )
                    return self._shape_stream(openai_stream)
                return await self._client.chat.completions.create(**prepared)
            except BadRequestError as e:
                # unsupported パラメータ特定のため、例外本文も記録する
                # 正常系: 未対応 param を落としてリトライするため、本文ログは通常出力に出さない
                logger.debug(
                    "BadRequestError from provider: %s (param=%r code=%r)",
                    str(e),
                    getattr(e, "param", None),
                    getattr(e, "code", None),
                )
                param = self._droppable_param(e)
                if (
                    param
                    and param in prepared
                    and param not in dropped
                    and param not in _PROTECTED_PARAMS
                ):
                    prepared.pop(param, None)
                    dropped.add(param)
                    logger.info("Dropping unsupported param '%s' and retrying", param)
                    continue
                raise

    @staticmethod
    def _droppable_param(e: BadRequestError) -> Optional[str]:
        """BadRequestError が未サポートと示しているリクエストパラメータを返す。該当なしは None。"""
        param = getattr(e, "param", None)
        if not param:
            return None
        code = getattr(e, "code", None)
        if code in ("unsupported_parameter", "unsupported_value"):
            return param
        # メッセージへのフォールバックは明示的な未サポートパラメータの表現に限定する。
        # 単なる "support" 部分一致（例："contact support"）は広すぎて
        # 正当なパラメータを削除してしまう可能性がある。
        text = str(e).lower()
        if "unsupported" in text or "not supported" in text:
            return param
        return None

    @staticmethod
    async def _shape_stream(openai_stream: Any) -> AsyncIterator[Any]:
        """SSE ラッパー向けにチャンクを yield する。

        ``include_usage`` を指定すると最終チャンクの ``choices`` リストが空になり
        ``usage`` が設定される。ラッパーは ``choices[0]`` にアクセスするため、
        この usage のみのチャンクを不活性な choice を1つ持つ形に整形する。
        """
        async for chunk in openai_stream:
            if not getattr(chunk, "choices", None) and getattr(chunk, "usage", None):
                empty_delta = SimpleNamespace(
                    content=None, tool_calls=None, role=None, function_call=None
                )
                empty_choice = SimpleNamespace(
                    index=0, delta=empty_delta, finish_reason=None
                )
                yield SimpleNamespace(
                    id=getattr(chunk, "id", ""),
                    model=getattr(chunk, "model", ""),
                    choices=[empty_choice],
                    usage=chunk.usage,
                )
            else:
                yield chunk


# id(model_config) ではなく接続関連フィールドをキーとして使用する。
# id(model_config) はモデルが GC された後に再利用され、別の設定と衝突して
# 誤ったクライアントを返す可能性があるため。同一内容の設定は同じ provider を共有する（意図した動作）。
_PROVIDER_CACHE: Dict[tuple, BaseProvider] = {}


def _cache_key(mc: ModelConfig) -> tuple:
    """設定の接続フィールドから provider キャッシュキーを構築する。

    api_key の SHA-256 フィンガープリントを使用する（平文ではなく）ため、
    長期間保持されるモジュールグローバルにシークレットが残らず、
    かつローテーションされたキーを識別できる。
    """
    # api_key はハッシュ化して保持する。長期間有効なモジュールグローバルな
    # キャッシュキーに平文で保持しないため（メモリ / repr にシークレットが残るのを防ぐ）。
    # ハッシュにより新旧のキーを区別することも可能。
    key_fingerprint = (
        hashlib.sha256(mc.api_key.encode()).hexdigest() if mc.api_key else None
    )
    return (
        mc.model_name,
        mc.provider,
        mc.deployment,
        mc.api_base,
        key_fingerprint,
        mc.api_version,
    )


def get_provider(model_config: ModelConfig) -> BaseProvider:
    """``model_config`` に対するキャッシュ済み（または新規構築）の provider を返す。

    未サポートの provider の場合は ValueError を送出する。同一の設定で1つの
    クライアントを共有できるよう、接続フィンガープリントで provider をキャッシュする。
    """
    if model_config.provider not in ("openai", "azure"):
        raise ValueError(f"Unsupported provider: {model_config.provider!r}")
    key = _cache_key(model_config)
    provider = _PROVIDER_CACHE.get(key)
    if provider is None:
        provider = OpenAILikeProvider(model_config)
        _PROVIDER_CACHE[key] = provider
    return provider
