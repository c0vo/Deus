"""
Shared date normalization utilities for news events, IPOs, and other
date strings extracted from unstructured sources.

Provides a single normalize_date() function used by event_tracker,
ipo_detector, and any other component that needs to parse LLM-generated
or API-returned date strings into YYYY-MM-DD format.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from config.logging_config import get_logger

log = get_logger(__name__)

# Cache the dateparser import (it's somewhat heavy to load)
_dateparser = None


def _get_dateparser():
    """Lazy-import dateparser to avoid overhead when not needed."""
    global _dateparser
    if _dateparser is None:
        try:
            import dateparser as _dateparser_mod
            _dateparser = _dateparser_mod
        except ImportError:
            log.warning("date_utils.dateparser_not_available")
            _dateparser = False  # sentinel
    return _dateparser if _dateparser is not False else None


# Regex patterns tried before giving up
_QUARTER_RE = re.compile(r"Q([1-4])\s*(\d{4})", re.IGNORECASE)
_MONTH_DAY_YEAR_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})",
    re.IGNORECASE,
)
_MONTH_DAY_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?",
    re.IGNORECASE,
)

_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_QUARTER_START = {"1": "01-01", "2": "04-01", "3": "07-01", "4": "10-01"}
_ABBR_MONTH_RE = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})",
    re.IGNORECASE,
)
_ABBR_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_date(raw_date: str) -> str:
    """Parse a raw date string into ISO 8601 ``YYYY-MM-DD``.

    Tries, in order:
    1. Python ``datetime.fromisoformat()`` (handles ISO-8601 dates)
    2. ``dateparser.parse()`` (handles natural-language dates)
    3. Regex patterns: Q1-Q4, "Month DD, YYYY", "Mon DD, YYYY"
    4. Returns empty string and logs a warning if nothing worked.

    Parameters
    ----------
    raw_date : str
        A date string from Finnhub, DeepSeek LLM output, or user input.

    Returns
    -------
    str
        ``YYYY-MM-DD`` if parseable, otherwise ``""``.
    """
    if not raw_date or raw_date.strip() in ("", "null", "None", "TBA", "tbd"):
        return ""

    raw = raw_date.strip()

    # --- 1. Try ISO-8601 ---
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass

    # --- 2. Try dateparser (natural language) ---
    dp = _get_dateparser()
    if dp is not None:
        try:
            dt = dp.parse(
                raw,
                settings={
                    "TIMEZONE": "UTC",
                    "RETURN_AS_TIMEZONE_AWARE": False,
                    "PREFER_DATES_FROM": "future",
                },
            )
            if dt is not None:
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # --- 3. Regex fallbacks ---

    # 3a. "Q3 2026" or "Q4-2026"
    m = _QUARTER_RE.match(raw)
    if m:
        return f"{m.group(2)}-{_QUARTER_START[m.group(1)]}"

    # 3b. "January 15, 2026" or "January 15th, 2026"
    m = _MONTH_DAY_YEAR_RE.match(raw)
    if m:
        month = _MONTH_NUM[m.group(1).lower()]
        return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"

    # 3c. "Jan 15, 2026"
    m = _ABBR_MONTH_RE.match(raw)
    if m:
        month = _ABBR_MONTH_NUM[m.group(1).lower()]
        return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"

    # 3d. "January 15" — assume current year if it's in the future, else next year
    m = _MONTH_DAY_RE.match(raw)
    if m:
        month = _MONTH_NUM[m.group(1).lower()]
        day = int(m.group(2))
        year = datetime.now(timezone.utc).year
        candidate = f"{year}-{month:02d}-{day:02d}"
        # If the inferred date is already past, bump to next year
        if candidate < datetime.now(timezone.utc).strftime("%Y-%m-%d"):
            candidate = f"{year + 1}-{month:02d}-{day:02d}"
        return candidate

    # --- 4. Give up ---
    log.warning("date_utils.unparseable_date", raw=raw)
    return ""


def normalize_company_name(name: str) -> str:
    """Normalize a company name for fuzzy deduplication matching.

    Strips legal suffixes, punctuation, and extra whitespace so that
    "Reddit Inc." and "Reddit, Inc." and "Reddit" all match the same entity.

    Parameters
    ----------
    name : str
        Raw company name from LLM output.

    Returns
    -------
    str
        Normalised lower-case name with suffixes removed.
    """
    if not name:
        return ""

    normalized = name.lower().strip()
    # Order matters — longer suffixes first to avoid partial stripping
    suffixes = [
        r"\s+incorporated",
        r"\s+corporation",
        r"\s+holdings",
        r"\s+inc\.?",
        r"\s+corp\.?",
        r"\s+ltd\.?",
        r"\s+llc\.?",
        r"\s+limited",
        r"\s+group",
        r"\s+co\.?",
        r"\s+plc",
        r"\s+lp\.?",
        r"\s+llp\.?",
        r"\s+l\.?p\.?",
    ]
    for suffix in suffixes:
        normalized = re.sub(suffix + r"$", "", normalized, flags=re.IGNORECASE)

    # Strip punctuation (keep spaces and alphanumeric)
    normalized = re.sub(r"[^\w\s]", "", normalized).strip()
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized
