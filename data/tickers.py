"""
Ticker market classification.

Tracked tickers are stored as a flat list of symbols with no exchange metadata
(see Database.get_tracked_tickers), but the data sources hanging off them are
market-specific: SEC EDGAR only knows US registrants, KRX only knows Korean
listings, and neither knows what to do with a crypto pair. Every such fetcher
routes through classify_market() so the split lives in exactly one place.
"""

from __future__ import annotations

import re

US = "US"
KR = "KR"
OTHER = "OTHER"

# yfinance suffixes for the two Korean boards.
_KR_SUFFIXES = (".KS", ".KQ")

# Bare KRX codes are six digits. yfinance needs the suffix; pykrx needs it gone.
_KRX_CODE = re.compile(r"^\d{6}$")

# Crypto pairs, FX and index symbols reach the watchlist via the scanner's
# default list and the market-regime tickers. None of them have a CIK.
_NON_EQUITY_PREFIXES = ("^",)
_NON_EQUITY_SUFFIXES = ("-USD", "=X", "=F")


def classify_market(ticker: str) -> str:
    """Return US, KR or OTHER for a ticker symbol.

    OTHER covers indices (^VIX), crypto (BTC-USD), FX and futures — anything
    that has no company registration behind it.
    """
    if not ticker:
        return OTHER
    t = str(ticker).strip().upper()

    if t.startswith(_NON_EQUITY_PREFIXES) or t.endswith(_NON_EQUITY_SUFFIXES):
        return OTHER
    if t.endswith(_KR_SUFFIXES) or _KRX_CODE.match(t):
        return KR
    return US


def to_krx_code(ticker: str) -> str | None:
    """'005930.KS' or '005930' -> '005930'. None if not a Korean listing.

    pykrx addresses stocks by bare six-digit code; yfinance requires the board
    suffix. Anything crossing between them goes through here.
    """
    if classify_market(ticker) != KR:
        return None
    t = str(ticker).strip().upper()
    for suffix in _KR_SUFFIXES:
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return t if _KRX_CODE.match(t) else None


def to_yfinance_symbol(krx_code: str, board: str = "KS") -> str:
    """'005930' -> '005930.KS'. Passes through anything already suffixed."""
    t = str(krx_code).strip().upper()
    if t.endswith(_KR_SUFFIXES):
        return t
    return f"{t}.{board.upper().lstrip('.')}"
