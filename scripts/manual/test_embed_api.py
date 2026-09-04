"""
Manual check: is MODEL_EMBEDDING still producing vectors the stored corpus
can be compared against?

Run this after changing MODEL_EMBEDDING, and before letting a pipeline cycle
write anything. Two things have to hold, and neither fails loudly on its own:

  1. Width is EMBEDDING_DIM. A different width is rejected by the embedder, so
     the visible symptom is articles that never get embedded.
  2. The vectors occupy the SAME space as the ones already stored. A model that
     returns the right width but a different space breaks nothing visibly — it
     silently miscalibrates dedup (0.70 on raw vectors), theme clustering (0.30
     on centered ones) and thesis grounding (0.65), all at once.

Requires OPENROUTER_API_KEY. Not a pytest test — it makes real API calls.
"""

import asyncio

import numpy as np

from config.llm import embed
from config.settings import settings
from data.database import Database
from pipeline.embedder import EMBEDDING_DIM, _article_text


async def main() -> None:
    model = settings.model_embedding
    print(f"MODEL_EMBEDDING = {model!r}\n")

    result = await embed(["This is a test article about TSLA."], model=model)
    vector = result.vectors[0]
    if not vector:
        print("FAIL: no vector returned")
        return

    print(f"dimension : {len(vector)} (expected {EMBEDDING_DIM})")
    print(f"tokens    : {getattr(result.usage, 'prompt_tokens', None)}")
    print(f"cost      : ${result.cost}")
    if len(vector) != EMBEDDING_DIM:
        print("\nFAIL: wrong width — this model cannot be used without re-embedding.")
        return

    # The part that matters: same width is not the same space.
    db = Database()
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id, headline, summary, embedding FROM articles "
            "WHERE embedding IS NOT NULL LIMIT 5"
        ).fetchall()

    if not rows:
        print("\nNo stored embeddings to compare against — nothing to corrupt yet.")
        return

    texts = [
        _article_text(type("A", (), {"headline": r["headline"], "summary": r["summary"]})())
        for r in rows
    ]
    fresh = await embed(texts, model=model)

    print("\ncosine vs stored:")
    worst = 1.0
    for row, values in zip(rows, fresh.vectors):
        stored = np.frombuffer(row["embedding"], dtype=np.float32)
        new = np.array(values, dtype=np.float32)
        cos = float(np.dot(stored, new) / (np.linalg.norm(stored) * np.linalg.norm(new)))
        worst = min(worst, cos)
        print(f"  {cos:.6f}  {row['headline'][:60]}")

    print()
    if worst > 0.99:
        print("PASS: same vector space — the stored corpus stays valid.")
    else:
        print(f"FAIL: worst cosine {worst:.4f}. This model occupies a different "
              "space; the whole corpus needs re-embedding before it is used.")


if __name__ == "__main__":
    asyncio.run(main())
