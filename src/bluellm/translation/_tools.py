import hashlib

# OpenAI 形式に変換が必要な Anthropic リクエストパラメーターのセット。
# _copy_untranslated_anthropic_params でこれ以外のキーをそのまま転送するために使用する。
_TRANSLATABLE_ANTHROPIC_PARAMS: frozenset = frozenset(
    [
        "messages",
        "metadata",
        "system",
        "tool_choice",
        "tools",
        "thinking",
        "output_format",
        "output_config",
    ]
)

# OpenAI は function/tool 名を64文字に制限している
# Anthropic にはこの制限がないため、長い名前を切り詰める必要がある
OPENAI_MAX_TOOL_NAME_LENGTH = 64
TOOL_NAME_HASH_LENGTH = 8
TOOL_NAME_PREFIX_LENGTH = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1  # 55


def truncate_tool_name(name: str) -> str:
    """
    OpenAI の64文字制限を超える tool 名を切り詰める。

    複数の tool が類似した長い名前を持つ場合の衝突を避けるため、
    {55文字のプレフィックス}_{8文字のハッシュ} 形式を使用する。

    Args:
        name: 元の tool 名

    Returns:
        64文字以下の場合は元の名前、それ以外はハッシュ付きの切り詰め済み名前
    """
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name

    # 衝突を避けるため、完全な名前から決定論的ハッシュを生成する
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{name_hash}"
