"""Secret-redaction tests.

The logging filter must scrub registered secrets from BOTH the rendered message
and any exception traceback (which the formatter renders from exc_info, bypassing
the message path). Short secrets must also be registered.
"""

from __future__ import annotations

import logging
import sys

import pytest

from bluellm import redaction
from bluellm.redaction import RedactionFilter, redact, register_secret


@pytest.fixture(autouse=True)
def clear_secrets():
    redaction._SECRETS.clear()
    yield
    redaction._SECRETS.clear()


def test_short_secret_is_redacted():
    # L1: the old len>=4 floor let short keys (e.g. a 2-char master key) leak.
    register_secret("mk")
    assert redact("x-api-key: mk done") == "x-api-key: ***REDACTED*** done"


def test_empty_secret_not_registered():
    register_secret("")
    assert redact("nothing here") == "nothing here"


def test_message_is_redacted():
    register_secret("azkey-SECRET")
    rec = logging.LogRecord(
        "bluellm", logging.INFO, __file__, 1, "calling with azkey-SECRET", (), None
    )
    RedactionFilter().filter(rec)
    assert "azkey-SECRET" not in rec.getMessage()


def test_traceback_is_redacted():
    # M7: secrets in exception tracebacks must be masked, not just in the message.
    register_secret("supersecretvalue")
    try:
        raise ValueError("boom supersecretvalue boom")
    except ValueError:
        exc = sys.exc_info()
    rec = logging.LogRecord(
        "bluellm", logging.ERROR, __file__, 1, "request failed", (), exc
    )
    RedactionFilter().filter(rec)
    assert rec.exc_text is not None
    assert "supersecretvalue" not in rec.exc_text
    assert rec.exc_info is None


def test_filter_tolerates_unrenderable_record():
    # A record whose getMessage() raises (format args mismatch) must not crash
    # the filter; it returns True and leaves the record untouched.
    rec = logging.LogRecord("bluellm", logging.INFO, __file__, 1, "%s %s", ("one",), None)
    assert RedactionFilter().filter(rec) is True


def test_install_attaches_filter_to_loggers():
    redaction.install()
    root_filters = logging.getLogger("").filters
    bluellm_filters = logging.getLogger("bluellm").filters
    assert any(isinstance(f, RedactionFilter) for f in root_filters)
    assert any(isinstance(f, RedactionFilter) for f in bluellm_filters)
    # Clean up the filters this test added so other tests aren't affected.
    for name in ("", "bluellm", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.filters = [f for f in lg.filters if not isinstance(f, RedactionFilter)]


def test_pre_rendered_exc_text_is_redacted():
    # H-1: a traceback already rendered into record.exc_text (by an upstream
    # handler) must also be scrubbed, not just freshly-rendered exc_info.
    register_secret("supersecretvalue")
    rec = logging.LogRecord("bluellm", logging.ERROR, __file__, 1, "boom", (), None)
    rec.exc_text = "Traceback (most recent call last):\nValueError: supersecretvalue\n"
    RedactionFilter().filter(rec)
    assert "supersecretvalue" not in rec.exc_text
