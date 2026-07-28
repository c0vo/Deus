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
}

REDDIT_KEYWORDS = {
    "calls", "puts", "yolo", "bagholder", "shorts", "hedgies", "options", 
    "strike", "expiration", "liquidation", "margin call", "apes", "moon", "drill",
    "diamond hands", "paper hands", "tendies", "rug pull", "stonks", "loss porn",
    "gain porn", "pump and dump", "fomo", "btfd", "wendys", "wife's boyfriend"
}

TICKER_PATTERN = re.compile(r'\b[A-Z]{2,5}\b')

EXCLUDED_WORDS = {
    "THE", "AND", "FOR", "OUT", "NEW", "NOW", "ALL", "BUT", "HAS", "ITS", 
    "ARE", "NOT", "WHO", "HOW", "WHY", "YOU", "OUR", "GET", "CAN", "ONE",
    "LOL", "OMG", "WTF", "DIY", "FAQ", "OS", "PC", "TV", "CPU", "GPU", "RAM", "SSD"
}
