"""
Shared filtering constants for the pipeline and data sources.
"""
import re

FINANCIAL_KEYWORDS = {
    # Core financial terms
    "earnings", "dividend", "acquisition", "merger", "revenue", "profit", "loss",
    "shares", "ipo", "sec", "inflation", "fed", "interest rate", "recession",
    "bullish", "bearish", "ticker", "stock", "etf", "nasdaq", "nyse", "sp500",
    "treasury", "yield", "commodity", "crude", "crypto", "bitcoin", "short squeeze",
    # Geopolitical & macro — war, conflict, trade
    "war", "conflict", "sanctions", "tariff", "trade war", "military", "invasion",
    "geopolitical", "central bank", "gdp", "employment", "unemployment",
    "jobs report", "cpi", "ppi", "consumer price", "producer price",
    "manufacturing", "pmi", "industrial production", "retail sales",
    "monetary policy", "fiscal policy", "stimulus",
    "government shutdown", "debt ceiling", "budget deficit",
    "market crash", "volatility", "vix",
    "supply chain", "regulation", "regulatory", "antitrust",
    "pandemic", "lockdown", "rate hike", "rate cut",
    "economic growth", "consumer spending", "labor market", "wage",
    "housing market",
    "corporate bond", "credit market",
    "bankruptcy", "default", "downgrade", "credit rating",
    "currency", "forex", "dollar", "euro", "yen", "yuan",
    "energy", "oil", "gas", "natural gas", "gold", "silver",
    "semiconductor", "chip",
    "presidential", "legislation",
    "fed chair", "federal reserve", "rate decision",
    # Armed-conflict vocabulary. "war"/"military"/"invasion" alone missed the
    # highest-scoring geopolitical stories in the archive — a 9.2 on strikes
    # against Iranian forces and an 8.2 on drone strikes in Russia both had
    # zero keyword hits and were being dropped before classification.
    "troops", "airstrike", "missile", "drone", "warfare", "retaliation",
    "ceasefire", "armed forces", "nuclear",
    # Retail investment products, likewise absent and likewise load-bearing.
    "401(k)", "401k", "index fund", "mutual fund", "pension", "hedge fund",
    "buyback", "shareholder", "insider", "etf",
}

REDDIT_KEYWORDS = {
    "calls", "puts", "yolo", "bagholder", "shorts", "hedgies", "options", 
    "strike", "expiration", "liquidation", "margin call", "apes", "moon", "drill",
    "diamond hands", "paper hands", "tendies", "rug pull", "stonks", "loss porn",
    "gain porn", "pump and dump", "fomo", "btfd", "wendys", "wife's boyfriend"
}

# A bare run of 2-5 capitals. Only meaningful against ORIGINAL-case text: run it
# over an uppercased string and every short word matches, which is exactly the
# bug that let non-financial articles through the aggregator's noise gate.
TICKER_PATTERN = re.compile(r'\b[A-Z]{2,5}\b')

# An explicit cashtag ($AAPL). Unambiguous enough to stand alone as a financial
# signal, unlike TICKER_PATTERN. A bare "$" is not — it matches "$100".
CASHTAG_PATTERN = re.compile(r'\$[A-Z]{1,5}\b')

EXCLUDED_WORDS = {
    "THE", "AND", "FOR", "OUT", "NEW", "NOW", "ALL", "BUT", "HAS", "ITS", 
    "ARE", "NOT", "WHO", "HOW", "WHY", "YOU", "OUR", "GET", "CAN", "ONE",
    "LOL", "OMG", "WTF", "DIY", "FAQ", "OS", "PC", "TV", "CPU", "GPU", "RAM", "SSD"
}
