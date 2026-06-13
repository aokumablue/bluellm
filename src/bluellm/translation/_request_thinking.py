from typing import (
    Any,
    Dict,
    Optional,
    cast,
)

from bluellm._compat import is_reasoning_auto_summary_enabled
from bluellm.types.anthropic_request import (
    AnthropicMessagesRequest,
)
from bluellm.types.openai import (
    ChatCompletionRequest,
)


class _RequestThinkingMixin:
    """Anthropic thinking パラメーターを OpenAI reasoning へ変換する変換群。"""

    @staticmethod
    def translate_anthropic_thinking_to_reasoning_effort(
        thinking: Dict[str, Any],
    ) -> Optional[str]:
        """
        Anthropic の thinking パラメーターを OpenAI の reasoning_effort に変換する。

        Anthropic thinking 形式: {'type': 'enabled'|'disabled', 'budget_tokens': int}
        OpenAI reasoning_effort: 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'default'

        マッピング:
        - budget_tokens >= 10000 -> 'high'
        - budget_tokens >= 5000  -> 'medium'
        - budget_tokens >= 2000  -> 'low'
        - budget_tokens < 2000   -> 'minimal'
        """
        if not isinstance(thinking, dict):
            return None

        thinking_type = thinking.get("type", "disabled")

        if thinking_type == "disabled":
            return None
        elif thinking_type == "enabled":
            budget_tokens = thinking.get("budget_tokens", 0)
            if budget_tokens >= 10000:
                return "high"
            elif budget_tokens >= 5000:
                return "medium"
            elif budget_tokens >= 2000:
                return "low"
            else:
                return "minimal"
        elif thinking_type == "adaptive":
            # Adaptive thinking: effort は budget_tokens ではなく output_config.effort で制御される。
            # デフォルト値を返す。利用可能な場合、呼び出し元は output_config.effort で上書きすること。
            return "medium"

        return None

    @staticmethod
    def build_reasoning_effort_param(
        thinking: Dict[str, Any], output_config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Anthropic thinking から OpenAI の ``reasoning_effort`` 値を構築する。

        summary が要求されている場合（``thinking.summary`` で明示的に指定、または
        ``BLUELLM_REASONING_AUTO_SUMMARY`` オプトインによる）は ``{"effort", "summary"}``
        の dict を返す（M1: 古い effort のみの変換で失われていた summary を保持する）。
        Adaptive thinking は ``output_config.effort`` から effort を取得する。
        thinking が effort にマッピングされない場合は ``None`` を返す。
        """
        reasoning_effort = (
            _RequestThinkingMixin.translate_anthropic_thinking_to_reasoning_effort(
                thinking
            )
        )
        if not reasoning_effort:
            return None
        if (
            isinstance(thinking, dict)
            and thinking.get("type") == "adaptive"
            and isinstance(output_config, dict)
            and output_config.get("effort")
        ):
            reasoning_effort = output_config["effort"]
        summary = thinking.get("summary") if isinstance(thinking, dict) else None
        if summary:
            return {"effort": reasoning_effort, "summary": summary}
        if is_reasoning_auto_summary_enabled():
            return {"effort": reasoning_effort, "summary": "detailed"}
        return reasoning_effort

    def _translate_thinking_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest,
        new_kwargs: ChatCompletionRequest,
    ) -> None:
        """Anthropic の thinking を thinking または reasoning_effort に変換する。"""
        if "thinking" not in anthropic_message_request:
            return

        thinking = anthropic_message_request["thinking"]
        if not thinking:
            return

        model = new_kwargs.get("model", "")
        if self.is_anthropic_claude_model(model):
            new_kwargs["thinking"] = thinking  # type: ignore
            return

        reasoning_effort = self.build_reasoning_effort_param(
            cast(Dict[str, Any], thinking),
            anthropic_message_request.get("output_config"),
        )
        if reasoning_effort is not None:
            new_kwargs["reasoning_effort"] = cast(Any, reasoning_effort)
