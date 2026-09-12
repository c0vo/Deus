"""
The default symbol universe, and the two index symbols that give it context.

Lives in `data/` rather than in `pipeline/market_scanner.py`, where
`DEFAULT_WATCHLIST` used to sit, because the price feed now has to warm the same
universe the scanner reads — and `price_feed -> market_scanner -> bot.alerts ->
bot.formatters` is an import chain a data constant has no business dragging in.
`market_scanner` re-exports the name so existing importers keep working.

SPY and QQQ are in the list for a second reason beyond being worth watching: a
price alert cannot say whether a 3% drop is the company's or the market's
without the same session's index move, and that reading comes from
`latest_prices` rows these two symbols put there.
"""

from __future__ import annotations

DEFAULT_WATCHLIST: list[str] = [
    "SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "BTC-USD",
]

# The session context every price alert is read against. Order matters only in
# that it is the order they are rendered in.
INDEX_TICKERS: tuple[str, ...] = ("SPY", "QQQ")
