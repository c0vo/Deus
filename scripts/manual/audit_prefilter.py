"""
Read-only dry run for the aggregator's noise pre-filter.

Replays `_has_financial_content` over every stored article and reports what the
current code would newly reject. The filter decides which articles are worth an
LLM classification call, so tightening it saves real money — and over-tightening
silently drops real financial news at ingestion, where nothing downstream will
ever notice.

The go/no-go signal is the "newly noise" list sorted by importance. A tightened
filter should reject lifestyle and off-topic articles; if it rejects anything
the ranker scored highly, or wipes out a whole source, it has gone too far.

Never writes. Run before shipping a change to data/filters.py or
pipeline/aggregator.py:

    python scripts/manual/audit_prefilter.py
"""

from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.models import NewsArticle  # noqa: E402
from pipeline.aggregator import _has_financial_content  # noqa: E402

DB_PATH = os.environ.get("DB_PATH", "storage/scrooge.db")

# event_types that mean the classifier found a real story. Anything here being
# newly rejected is a false negative, not a saving.
REAL_EVENT_TYPES = {
    "earnings", "macro", "geopolitical", "merger", "product_launch",
    "regulatory", "ipo", "personnel", "meme_stock",
}


def main() -> int:
    if not os.path.exists(DB_PATH):
        print(f"No database at {DB_PATH}. Set DB_PATH to override.")
        return 1

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT headline, summary, source_name, source_type, url, published_at,
               event_type, importance_score, raw_data
        FROM articles
        """
    ).fetchall()

    total = len(rows)
    was_noise = 0
    now_noise = 0
    newly_noise: list[dict] = []
    newly_by_source: collections.Counter = collections.Counter()
    newly_real_events: collections.Counter = collections.Counter()

    for row in rows:
        try:
            raw = json.loads(row["raw_data"]) if row["raw_data"] else {}
        except (json.JSONDecodeError, TypeError):
            raw = {}

        article = NewsArticle(
            id=row["url"],
            headline=row["headline"] or "",
            summary=row["summary"] or "",
            source_name=row["source_name"] or "unknown",
            source_type=row["source_type"] or "rss",
            url=row["url"],
            published_at=row["published_at"],
            raw_data=raw if isinstance(raw, dict) else {},
        )

        old_noise = row["event_type"] == "noise"
        new_noise = not _has_financial_content(article)

        was_noise += old_noise
        now_noise += new_noise

        if new_noise and not old_noise:
            newly_noise.append(dict(row))
            newly_by_source[row["source_name"]] += 1
            if row["event_type"] in REAL_EVENT_TYPES:
                newly_real_events[row["event_type"]] += 1

    print(f"Articles scanned: {total}")
    print(f"  currently noise: {was_noise:>5} ({100.0 * was_noise / max(total, 1):.1f}%)")
    print(f"  would be noise:  {now_noise:>5} ({100.0 * now_noise / max(total, 1):.1f}%)")
    print(f"  newly rejected:  {len(newly_noise):>5}")
    print()

    print("── Newly rejected, highest ranked first ─────────────────────────")
    print("   Anything scoring >= 5.0 here means the filter is over-correcting.")
    ranked = sorted(
        newly_noise, key=lambda r: r["importance_score"] or -1.0, reverse=True
    )
    for r in ranked[:25]:
        score = r["importance_score"]
        score_s = f"{score:.1f}" if score is not None else "  -"
        print(f"   {score_s}  [{r['event_type'] or 'unclassified'}] {(r['headline'] or '')[:70]}")
    if not ranked:
        print("   (none)")
    print()

    over = [r for r in newly_noise if (r["importance_score"] or 0) >= 5.0]
    print(f"── Rejected with importance >= 5.0: {len(over)} (want 0)")
    print(f"── Rejected despite a real event_type: {sum(newly_real_events.values())} (want ~0)")
    for et, n in newly_real_events.most_common():
        print(f"     {et:<16} {n}")
    print()

    print("── Newly rejected by source ─────────────────────────────────────")
    print("   A source losing most of its volume is a red flag.")
    source_totals = collections.Counter(r["source_name"] for r in rows)
    for source, n in newly_by_source.most_common():
        pct = 100.0 * n / max(source_totals[source], 1)
        print(f"   {source:<26} {n:>4} of {source_totals[source]:>4} ({pct:.0f}%)")
    if not newly_by_source:
        print("   (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
