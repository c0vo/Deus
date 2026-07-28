import asyncio
from dotenv import load_dotenv

# Load env vars before anything else
load_dotenv()

from data.sources.finnhub_source import FinnhubSource
from data.sources.alpha_vantage_source import AlphaVantageSource
from config.settings import settings

async def test_api_sources():
    print("🚀 Testing Finnhub and Alpha Vantage Sources...\n")
    
    print(f"Finnhub Key Set: {settings.has_key('finnhub_api_key')}")
    print(f"Alpha Vantage Key Set: {settings.has_key('alpha_vantage_api_key')}\n")
    
    # 1. Test Finnhub
    print("--- 🔵 FINNHUB ---")
    if settings.has_key("finnhub_api_key"):
        finnhub = FinnhubSource()
        finnhub_articles = await finnhub.fetch()
        print(f"Fetched {len(finnhub_articles)} articles.")
        if finnhub_articles:
            sample = finnhub_articles[0]
            print(f"Sample Headline: {sample.headline}")
            print(f"Sample Summary: {sample.summary[:150]}...")
            print(f"Sample Raw Data: {sample.raw_data}")
            print(f"Extracted Tickers (if any): {sample.affected_tickers}")
    else:
        print("Skipping Finnhub (no key)")

    print("\n--- 🟠 ALPHA VANTAGE ---")
    if settings.has_key("alpha_vantage_api_key"):
        av = AlphaVantageSource()
        av_articles = await av.fetch()
        print(f"Fetched {len(av_articles)} articles.")
        if av_articles:
            sample = av_articles[0]
            print(f"Sample Headline: {sample.headline}")
            print(f"Sample Summary: {sample.summary[:150]}...")
            print(f"Sample Sentiment Score (Pre-computed): {sample.sentiment_score}")
            print(f"Sample Raw Data Keys: {list(sample.raw_data.keys())}")
            print(f"Extracted Tickers: {sample.affected_tickers}")
    else:
        print("Skipping Alpha Vantage (no key)")

if __name__ == "__main__":
    asyncio.run(test_api_sources())
