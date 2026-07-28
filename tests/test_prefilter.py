import asyncio
from datetime import datetime
from data.models import NewsArticle
from pipeline.classifier import ArticleClassifier

async def test_filter():
    classifier = ArticleClassifier()
    
    # 1. High-signal SPCE post
    spce_article = NewsArticle(
        id="test_spce_1",
        headline="[r/wallstreetbets] What are your thoughts on SPCE?",
        summary="Is Virgin Galactic still a buy or is it drilling?",
        source_name="reddit_wallstreetbets",
        source_type="social",
        url="https://reddit.com/r/wallstreetbets/comments/test_spce_1",
        published_at=datetime.now(),
        raw_data={"comments": [{"author": "user1", "body": "I have calls at 10!"}]}
    )
    
    # 2. Noisy off-topic Reddit post
    noise_article = NewsArticle(
        id="test_noise_1",
        headline="[r/investing] What is the best mechanical keyboard?",
        summary="Looking for suggestions under $100.",
        source_name="reddit_investing",
        source_type="social",
        url="https://reddit.com/r/investing/comments/test_noise_1",
        published_at=datetime.now(),
        raw_data={"comments": [{"author": "user2", "body": "I like MX Browns."}]}
    )
    
    # 3. SPCE post with comments holding the context
    spce_comments_article = NewsArticle(
        id="test_spce_2",
        headline="[r/stocks] Look at this",
        summary="Is it time to load up?",
        source_name="reddit_stocks",
        source_type="social",
        url="https://reddit.com/r/stocks/comments/test_spce_2",
        published_at=datetime.now(),
        raw_data={"comments": [{"author": "user3", "body": "Definitely calls on SPCE here."}]}
    )
    
    # Test should_classify
    print("--- Testing pre-filter heuristics ---")
    
    keep_spce1 = classifier.should_classify(spce_article)
    keep_noise = classifier.should_classify(noise_article)
    keep_spce2 = classifier.should_classify(spce_comments_article)
    
    print(f"SPCE Post 1 (Direct Mention) - Keep? {keep_spce1} (Expected: True)")
    print(f"Noise Post (Keyboard)       - Keep? {keep_noise} (Expected: False)")
    print(f"SPCE Post 2 (Comment Ment.) - Keep? {keep_spce2} (Expected: True)")
    
    assert keep_spce1 is True, "Failed to keep SPCE post"
    assert keep_noise is False, "Failed to filter out keyboard post"
    assert keep_spce2 is True, "Failed to keep comment-mentioned SPCE post"
    print("\n[OK] Heuristic check passed successfully!")
    
    # Run classification simulation
    print("\n--- Running classification test ---")
    classified_spce1 = await classifier.classify(spce_article)
    print(f"SPCE 1 Event Type: {classified_spce1.event_type} (Expected: Not 'noise')")
    
    classified_noise = await classifier.classify(noise_article)
    print(f"Noise Event Type: {classified_noise.event_type} (Expected: 'noise')")
    
    assert classified_noise.event_type == "noise", "Noise article was not marked as noise"
    print("\n[OK] Classification bypass worked perfectly!")

if __name__ == "__main__":
    asyncio.run(test_filter())
