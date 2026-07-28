import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from data.sources.reddit_source import RedditSource
from pipeline.classifier import ArticleClassifier

async def test_pipeline():
    print("🚀 Starting Reddit Pipeline Integration Test...")
    
    # Initialize components
    source = RedditSource(subreddits=["wallstreetbets"], max_posts_per_sub=3)
    classifier = ArticleClassifier()
    
    # 1. Fetch
    print("\n📡 Fetching from Reddit...")
    articles = await source.fetch()
    
    if not articles:
        print("❌ No articles fetched.")
        return
        
    print(f"✅ Fetched {len(articles)} articles.")
    
    # 1.5 Enrich
    print("\n📡 Enriching articles (fetching comments)...")
    await source.enrich(articles)
    print("✅ Enrichment complete.")
    
    # 2. Inspect first article
    article = articles[0]
    print("\n📌 FIRST ARTICLE:")
    print(f"Title: {article.headline}")
    print(f"URL: {article.url}")
    
    comments = article.raw_data.get("comments", [])
    print(f"Comments extracted: {len(comments)}")
    
    if comments:
        print(f"Sample Comment: {comments[0]['author']} - {comments[0]['body'][:100]}...")
        
    # 3. Classify
    print("\n🧠 Sending to LLM Classifier...")
    classified_article = await classifier.classify(article)
    
    print("\n📊 CLASSIFICATION RESULTS:")
    print(f"Event Type: {classified_article.event_type}")
    print(f"Sentiment: {classified_article.sentiment_score} ({classified_article.suggested_direction})")
    print(f"Urgency: {classified_article.urgency}")
    print(f"Affected Sectors: {classified_article.affected_sectors}")
    print(f"Affected Tickers: {classified_article.affected_tickers}")
    print(f"Summary: {classified_article.classification_summary}")
    
    print("\n✅ Test Complete.")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
