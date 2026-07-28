import sys
import asyncio
from pathlib import Path

# Add root directory to sys.path to allow imports when run directly
sys.path.append(str(Path(__file__).parent.parent))

from data.sources.reddit_source import RedditSource
import json

async def main():
    source = RedditSource(subreddits=["wallstreetbets"], max_posts_per_sub=3)
    articles = await source.fetch()
    
    print(f"Fetched {len(articles)} articles.")
    for a in articles:
        print("-----")
        print(f"ID: {a.id}")
        print(f"Headline: {a.headline}")
        print(f"URL: {a.url}")
        print(f"Published: {a.published_at}")
        comments = a.raw_data.get("comments", [])
        print(f"Top 2 comments:")
        for c in comments[:2]:
            print(f"  - {c.get('author')}: {c.get('body')[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())

