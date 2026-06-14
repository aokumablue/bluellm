"""usage トークン記録（UsageLogger）のテスト。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from bluellm.cost import UsageLogger


def _read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_records_tokens_to_dated_file(tmp_path):
    logger = UsageLogger(base_dir=tmp_path)
    when = datetime(2026, 6, 14, 1, 2, 3, tzinfo=timezone.utc)
    logger.record(
        "gpt-5.4",
        "azure",
        {"input_tokens": 12, "output_tokens": 7},
        now=lambda: when,
    )
    f = tmp_path / "2026-06-14.jsonl"
    rows = _read_lines(f)
    assert rows == [
        {
            "ts": "2026-06-14T01:02:03+00:00",
            "model": "gpt-5.4",
            "provider": "azure",
            "input_tokens": 12,
            "output_tokens": 7,
        }
    ]


def test_records_cache_tokens_when_present(tmp_path):
    logger = UsageLogger(base_dir=tmp_path)
    when = datetime(2026, 6, 14, tzinfo=timezone.utc)
    logger.record(
        "llama3.3",
        "ollama",
        {
            "input_tokens": 5,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2,
        },
        now=lambda: when,
    )
    row = _read_lines(tmp_path / "2026-06-14.jsonl")[0]
    assert row["cache_read_input_tokens"] == 4
    assert row["cache_creation_input_tokens"] == 2


def test_cache_tokens_omitted_when_zero_or_absent(tmp_path):
    logger = UsageLogger(base_dir=tmp_path)
    when = datetime(2026, 6, 14, tzinfo=timezone.utc)
    logger.record(
        "gpt-5.4",
        "openai",
        {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0},
        now=lambda: when,
    )
    row = _read_lines(tmp_path / "2026-06-14.jsonl")[0]
    assert "cache_read_input_tokens" not in row
    assert "cache_creation_input_tokens" not in row


def test_missing_token_fields_default_to_zero(tmp_path):
    logger = UsageLogger(base_dir=tmp_path)
    when = datetime(2026, 6, 14, tzinfo=timezone.utc)
    logger.record("m", "openai", {}, now=lambda: when)
    row = _read_lines(tmp_path / "2026-06-14.jsonl")[0]
    assert row["input_tokens"] == 0 and row["output_tokens"] == 0


def test_splits_files_by_date(tmp_path):
    logger = UsageLogger(base_dir=tmp_path)
    d1 = datetime(2026, 6, 14, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 15, tzinfo=timezone.utc)
    logger.record("m", "openai", {"input_tokens": 1, "output_tokens": 1}, now=lambda: d1)
    logger.record("m", "openai", {"input_tokens": 2, "output_tokens": 2}, now=lambda: d2)
    logger.record("m", "openai", {"input_tokens": 3, "output_tokens": 3}, now=lambda: d1)
    assert len(_read_lines(tmp_path / "2026-06-14.jsonl")) == 2
    assert len(_read_lines(tmp_path / "2026-06-15.jsonl")) == 1


def test_record_swallows_errors(tmp_path):
    # 記録失敗（now が例外）でも例外を伝播させない。
    logger = UsageLogger(base_dir=tmp_path)

    def boom():
        raise RuntimeError("clock failure")

    # 例外が出ないこと（握り潰し）。ファイルも作られない。
    logger.record("m", "openai", {"input_tokens": 1, "output_tokens": 1}, now=boom)
    assert list(tmp_path.iterdir()) == []


def test_default_base_dir_used_when_unset(monkeypatch, tmp_path):
    # base_dir 未指定なら _DEFAULT_BASE_DIR を使う（conftest が tmp に差し替え済み）。
    monkeypatch.setattr("bluellm.cost._DEFAULT_BASE_DIR", tmp_path / "costs")
    logger = UsageLogger()
    when = datetime(2026, 6, 14, tzinfo=timezone.utc)
    logger.record("m", "openai", {"input_tokens": 1, "output_tokens": 1}, now=lambda: when)
    assert (tmp_path / "costs" / "2026-06-14.jsonl").exists()
