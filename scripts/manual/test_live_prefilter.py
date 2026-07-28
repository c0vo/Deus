import sys
import asyncio
from pathlib import Path

# Add root directory to sys.path to allow imports when run directly
sys.path.append(str(Path(__file__).parent.parent))

from data.sources.reddit_source import RedditSource
from pipeline.classifier import ArticleClassifier

async def run_live_test():
    print("Fetching live posts from Reddit...")
    source = RedditSource(subreddits=["wallstreetbets", "stocks", "investing"], max_posts_per_sub=10)
    articles = await source.fetch()
    
    if not articles:
        print("No articles fetched from Reddit. Check internet connection or Reddit rate limits.")
        return
        
    classifier = ArticleClassifier()
    
    passed_count = 0
    filtered_count = 0
    
    print("\n--- Live Reddit Pre-Filter Feed Analysis ---")
    for article in articles:
        should_keep = classifier.should_classify(article)
        status = "KEEP" if should_keep else "FILTERED (NOISE)"
        
        # Check if SPCE is mentioned in the text
        comments_data = article.raw_data.get("comments", [])
        comments_text = " ".join([c['body'] for c in comments_data]) if comments_data else ""
        full_text = f"{article.headline} {article.summary} {comments_text}"
        spce_mention = "[SPCE Mentioned!]" if "SPCE" in full_text.upper() else ""
        
        print(f"[{status}] {article.headline[:60]}... {spce_mention}")
        
        if should_keep:
            passed_count += 1
        else:
            filtered_count += 1
            
    print(f"\nResults: Keep={passed_count}, Filtered={filtered_count}, Total={len(articles)}")
    
    # Try classifying at least one that passed and one that got filtered (if any)
    passed_articles = [a for a in articles if classifier.should_classify(a)]
    filtered_articles = [a for a in articles if not classifier.should_classify(a)]
    
    if passed_articles:
        print("\n--- Simulating classification on a live PASSED article ---")
        target = passed_articles[0]
        print(f"Classifying: {target.headline}")
        res = await classifier.classify(target)
        print(f"Result -> Event Type: {res.event_type}, Sentiment: {res.sentiment_score}, Tickers: {res.affected_tickers}")
        
    if filtered_articles:
        print("\n--- Simulating classification on a live FILTERED article ---")
        target = filtered_articles[0]
        print(f"Classifying: {target.headline}")
        res = await classifier.classify(target)
        print(f"Result -> Event Type: {res.event_type} (LLM Bypassed)")

if __name__ == "__main__":
    asyncio.run(run_live_test())

