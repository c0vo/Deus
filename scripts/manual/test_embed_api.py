import asyncio
import numpy as np
from config.llm import get_client

async def test_embedder():
    client = get_client()
    if not client:
        print("No client")
        return
        
    try:
        res = client.models.embed_content(
            model="text-embedding-004", 
            contents="This is a test article about TSLA."
        )
        vec = res.embeddings[0].values
        print(f"text-embedding-004 len: {len(vec)}")
    except Exception as e:
        print("004 error:", e)
        
    try:
        res = client.models.embed_content(
            model="gemini-embedding-001", 
            contents="This is a test article about TSLA."
        )
        vec = res.embeddings[0].values
        print(f"gemini-embedding-001 len: {len(vec)}")
    except Exception as e:
        print("001 error:", e)

if __name__ == "__main__":
    asyncio.run(test_embedder())
