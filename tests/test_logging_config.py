"""
Tests for `config/logging_config.py`.

One behaviour, and it is a security one: the HTTP client libraries must not log
request URLs at INFO. `httpx` emits one "HTTP Request: POST https://…" line per
call, and the Telegram Bot API carries the bot token *in the path*
(api.telegram.org/bot<token>/getUpdates). With polling that is several lines a
second, so every page of a long-running worker log leaked the credential — a
246 MB file of it on the phone, readable by anything with filesystem access and
by anyone the log was ever sent to for diagnosis.
"""

import logging

import pytest

from config.logging_config import _NOISY_THIRD_PARTY_LOGGERS, setup_logging


@pytest.fixture(autouse=True)
def _reconfigure():
    """Levels are applied by setup_logging, which other tests may have reset."""
    setup_logging()


@pytest.mark.parametrize("name", _NOISY_THIRD_PARTY_LOGGERS)
def test_request_loggers_are_raised_to_warning(name):
    assert logging.getLogger(name).level == logging.WARNING


def test_an_httpx_info_record_is_filtered_out():
    """The exact record shape that leaked the token."""
    logger = logging.getLogger("httpx")
    assert not logger.isEnabledFor(logging.INFO)

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger.addHandler(handler)
    try:
        logger.info(
            'HTTP Request: POST https://api.telegram.org/bot123456:FAKE-TOKEN/getUpdates "HTTP/1.1 200 OK"'
        )
        assert records == []
    finally:
        logger.removeHandler(handler)


def test_failures_from_those_loggers_still_get_through():
    """WARNING and above is what makes this a level change, not a gag."""
    logger = logging.getLogger("httpx")
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger.addHandler(handler)
    try:
        logger.warning("connection timed out")
        assert [r.getMessage() for r in records] == ["connection timed out"]
    finally:
        logger.removeHandler(handler)
