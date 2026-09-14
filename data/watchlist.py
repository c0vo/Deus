"""
The default symbol universe, the two index symbols that give it context, and
the symbol lists the direction model trains and reads against.

Lives in `data/` rather than in `pipeline/market_scanner.py`, where
`DEFAULT_WATCHLIST` used to sit, because the price feed now has to warm the same
universe the scanner reads — and `price_feed -> market_scanner -> bot.alerts ->
bot.formatters` is an import chain a data constant has no business dragging in.
`market_scanner` re-exports the name so existing importers keep working.

SPY and QQQ are in the list for a second reason beyond being worth watching: a
price alert cannot say whether a 3% drop is the company's or the market's
without the same session's index move, and that reading comes from
`latest_prices` rows these two symbols put there.

The sector map moved here from `pipeline/predictor.py` for the same reason the
watchlist did: the price feed and the feature builder both need it, and neither
should import the predictor (and its LLM client) to read a dictionary.
`pipeline.predictor` re-imports the three names so existing importers keep
working.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_WATCHLIST: list[str] = [
    "SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "BTC-USD",
]

# The session context every price alert is read against. Order matters only in
# that it is the order they are rendered in.
INDEX_TICKERS: tuple[str, ...] = ("SPY", "QQQ")

# The eleven Select Sector SPDRs, one per GICS sector. They are feature inputs
# (each ticker's return is read against its sector's) and, in the `core`
# universe, training rows in their own right.
SECTOR_ETFS: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLC", "XLI", "XLU", "XLRE", "XLB",
]

# Extra training rows for the pooled direction model, never shown anywhere.
#
# The tracked watchlist is a handful of correlated growth names, which is too
# few independent rows to learn a pooled model from and teaches it one factor
# (semis) rather than the market. These are liquid, long-listed US large caps
# chosen for sector spread — every GICS sector, two or three names where the
# sector is large — and none of them is in DEFAULT_WATCHLIST. Listing age
# matters more than name recognition: a stock that IPO'd last year adds a year
# of rows, each of these adds decades.
TRAINING_BACKBONE: list[str] = [
    # Technology / semiconductors, beyond the watchlist's NVDA/AAPL/MSFT
    "AVGO", "AMD", "ORCL",
    # Communication services
    "GOOGL", "META",
    # Consumer cyclical
    "AMZN", "HD", "MCD",
    # Financials
    "JPM", "BAC", "GS",
    # Energy
    "XOM", "CVX",
    # Health care
    "UNH", "JNJ", "LLY",
    # Consumer defensive
    "PG", "KO", "COST",
    # Industrials
    "CAT", "UNP",
    # Basic materials, utilities, real estate
    "LIN", "NEE", "PLD",
]

# Market-wide series the features read but the model never trains on as rows.
# ^GSPC sits alongside SPY because the features take whichever of the two has
# the longer clean daily history (see pipeline.features), so both have to stay
# current — a fallback that stopped updating would silently blank every market
# feature on the live row.
MARKET_INPUT_TICKERS: list[str] = ["SPY", "^GSPC", "^VIX", "^TNX"]

# Symbols that are funds rather than companies, for the `is_etf` feature. Static
# on purpose: a quoteType lookup is a network call per symbol, and a feature must
# not depend on whether Yahoo answered that morning. Index symbols (a leading
# '^') are treated as funds by the feature code without being listed here.
ETF_TICKERS: frozenset[str] = frozenset(
    {"SPY", "QQQ", "QQQM", "VOO", "IWM", "DIA", *SECTOR_ETFS}
)

# Keys are normalized via _normalize_sector() so that both yfinance's own strings
# ("Financial Services", "Consumer Cyclical") and the older underscored spellings
# resolve to the same ETF. Before normalization only "Technology" matched anything
# actually tracked, so sector_etf_return_1d was silently 0.0 for most tickers.
SECTOR_ETF_MAP = {
    "technology": "XLK",
    "financial services": "XLF",
    "finance": "XLF",
    "financial": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "health care": "XLV",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "communication services": "XLC",
    "industrials": "XLI",
    "defense": "XLI",
    "utilities": "XLU",
    "real estate": "XLRE",
    "basic materials": "XLB",
    "index": "SPY",
}


def _normalize_sector(sector: Optional[str]) -> str:
    """Fold sector spellings to a single lookup key ('Consumer_Cyclical' -> 'consumer cyclical')."""
    if not sector:
        return ""
    return str(sector).replace("_", " ").replace("-", " ").strip().lower()


def _sector_etf(sector: Optional[str]) -> Optional[str]:
    """Resolve a sector name to its tracking ETF, or None if unmapped."""
    return SECTOR_ETF_MAP.get(_normalize_sector(sector))


# The public name. The underscored original stays because pipeline.predictor
# imported it under that name.
sector_etf_for = _sector_etf
