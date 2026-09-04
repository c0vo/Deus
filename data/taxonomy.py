"""
Briefing lane taxonomy.

The daily brief sorts articles into three fixed lanes — Global & Macro, Tech,
and Markets — rather than grouping by ``affected_sectors``. That column is
unconstrained LLM free text: the database holds 177 distinct sector strings
("Technology", "Artificial Intelligence", "Cloud Computing", "Semiconductors"
are four separate groups), so grouping by it produces a scatter of one-item
sections instead of a readable brief.

The lane predicates deliberately read several weak signals rather than one
strong one, because no strong one exists:

- There is no "tech" event_type. The classifier's taxonomy has nine categories
  and none of them is technology, so tech is recognised from sector stems,
  source provenance, and headline keywords.
- ``event_type`` is an unvalidated ``str`` on ClassifierResult, unlike urgency
  and suggested_direction which carry regex patterns. The LLM invents
  categories ("analyst_rating_change", "price_hike"), so ~40 distinct values
  exist against the 9 defined. Falling back to source and keyword signals is
  what makes the lanes robust to that drift.
- The ``countries`` column is populated in 0 of 2445 rows despite a backfill
  job and a gazetteer existing, so "global" is defined on event type and
  subject matter, not geography.

This module imports nothing from the project so data/, pipeline/, bot/, and
api/ can all use it without cycles.
"""

from __future__ import annotations

import json
import re

# ── Selection bounds ─────────────────────────────────────────────────────
# Roughly 85% of ranked articles score below 7.0, so the floor is what makes
# the brief a brief. On a quiet day it is correct for this to return nothing —
# the caller sends a "no high-impact stories" message rather than padding.
BRIEFING_MAX_ITEMS = 6
BRIEFING_MIN_IMPORTANCE = 7.0

LANE_ORDER = ("global", "tech", "markets")
LANE_LABELS = {
    "global": "🌍 GLOBAL & MACRO",
    "tech": "💻 TECH",
    "markets": "📊 MARKETS",
}
LANE_MIN = {"global": 2, "tech": 2}
LANE_MAX = {"global": 3, "tech": 3}

# ── Global & macro signals ───────────────────────────────────────────────
MACRO_EVENT_TYPES = {"macro", "geopolitical"}

# Feeds whose whole editorial remit is macro or non-US coverage.
MACRO_SOURCES = {"fed_press", "nyt_business"}
MACRO_SOURCE_PREFIXES = ("korea_times",)

# Note the absence of "antitrust": that is regulatory, not macro, and including
# it would pull single-company enforcement stories out of their real lane.
MACRO_TERMS = {
    "tariff", "tariffs", "sanctions", "central bank", "federal reserve",
    "rate decision", "rate hike", "rate cut", "interest rate", "cpi",
    "inflation", "gdp", "jobs report", "payrolls", "unemployment", "pmi",
    "trade war", "export controls", "geopolitical", "imf", "opec", "ecb",
    "boj", "treasury yield", "recession", "stimulus", "debt ceiling",
    "government shutdown", "ceasefire", "election", "sovereign",
}

# ── Tech signals ─────────────────────────────────────────────────────────
# Stems, matched as substrings against individual affected_sectors entries.
# "technolog" catches both Technology and Information Technology; "telecom"
# catches Telecommunications. Substring matching is safe here because these run
# against short controlled sector strings, not free prose.
#
# Cryptocurrency is deliberately absent: it trades as an asset class, so it
# belongs in Markets even though the underlying subject is technical.
TECH_SECTOR_TERMS = (
    "technolog", "semiconductor", "software", "artificial intelligence",
    "cloud", "internet", "cybersecurity", "hardware", "e-commerce",
    "ecommerce", "gaming", "telecom", "electronics", "chip", "data center",
    "fintech", "social media",
)

TECH_SOURCES = {"hackernews", "wsj_tech"}

TECH_TERMS = {
    "ai", "gpu", "llm", "chip", "chips", "semiconductor", "cloud", "software",
    "smartphone", "data center", "openai", "nvidia", "quantum", "cyberattack",
    "chatbot", "app store", "foundry", "wafer",
}


def _compile_terms(terms: set[str]) -> re.Pattern:
    """
    Build one word-boundary alternation from a term set.

    Word boundaries are not optional here. A naive substring test for the term
    "ai" matches "Retail", "Ukraine", and "chain"; "chip" matches "chipotle".
    Longest-first ordering keeps multi-word terms from being shadowed by a
    shorter prefix during alternation.
    """
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ordered) + r")\b")


MACRO_RE = _compile_terms(MACRO_TERMS)
TECH_RE = _compile_terms(TECH_TERMS)


def _sectors(row: dict) -> list[str]:
    """
    Read affected_sectors off a row whatever shape it arrived in.

    Rows come from sqlite (a JSON string), from code that already parsed them
    (a list), or from a test (either). Anything unparseable is treated as no
    sectors rather than raising — a malformed column should cost an article its
    tech classification, not break the brief.
    """
    raw = row.get("affected_sectors")
    if isinstance(raw, list):
        return [str(s) for s in raw]
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(s) for s in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _text(row: dict) -> str:
    """Headline plus whichever summary is present, lowercased once."""
    summary = row.get("classification_summary") or row.get("summary") or ""
    return f"{row.get('headline') or ''} {summary}".lower()


def _source(row: dict) -> str:
    return (row.get("source_name") or "").lower()


def is_global(row: dict) -> bool:
    """Macro or geopolitical news — the world-level story, not a single name."""
    if (row.get("event_type") or "").lower() in MACRO_EVENT_TYPES:
        return True

    source = _source(row)
    if source in MACRO_SOURCES or source.startswith(MACRO_SOURCE_PREFIXES):
        return True

    return bool(MACRO_RE.search(_text(row)))


def is_tech(row: dict) -> bool:
    """Technology news, recognised without a dedicated event_type."""
    for sector in _sectors(row):
        lowered = sector.lower()
        if any(term in lowered for term in TECH_SECTOR_TERMS):
            return True

    if _source(row) in TECH_SOURCES:
        return True

    return bool(TECH_RE.search(_text(row)))


def _dedupe_candidates(rows: list[dict]) -> list[dict]:
    """
    Drop articles whose classification summary repeats one already in the pool.

    The pipeline's semantic dedup runs on embeddings and misses articles that
    were never embedded: the live database holds pairs with byte-identical
    classification summaries and no duplicate_of link. At a six-item cap one
    such pair costs a third of the brief, so this is a cheap second net — not a
    replacement for the embedding path.

    Matching on the summary rather than the headline is deliberate. Syndicated
    coverage of one event is rewritten headline-first: the two Dominion/NextEra
    articles in the database share only 27% of their headline words, while
    "Fed raises rates by 25bps" and "Fed signals pause after 25bps hike" — two
    genuinely different stories — share 29%. There is no headline threshold
    that separates those cases, so this only claims what it can actually do.

    Assumes rows are sorted best-first, so the survivor of any pair is the
    higher-scoring one.
    """
    kept: list[dict] = []
    seen_summaries: set[str] = set()

    for row in rows:
        summary = " ".join((row.get("classification_summary") or "").lower().split())
        if summary and summary in seen_summaries:
            continue
        if summary:
            seen_summaries.add(summary)
        kept.append(row)

    return kept


def _sort_key(row: dict):
    """
    Importance descending, then oldest-first, then url.

    The tie-breaks exist so two runs over the same data produce the same brief.
    Importance is a coarse one-decimal score, so ties are common.
    """
    return (
        -(row.get("importance_score") or 0.0),
        row.get("published_at") or "",
        row.get("url") or "",
    )


def select_briefing_lanes(
    rows: list[dict], cap: int = BRIEFING_MAX_ITEMS
) -> list[tuple[str, list[dict]]]:
    """
    Pick at most ``cap`` articles and arrange them into lanes.

    Rows are expected to already clear the importance floor (the SQL applies
    it). Global and Tech each get a guaranteed minimum so the two lanes the
    brief exists to surface cannot be crowded out by a busy market day; unused
    slots flow onward rather than being wasted.

    Returns [(lane_label, rows)] in LANE_ORDER, omitting empty lanes. An empty
    return means nothing cleared the floor.
    """
    rows = _dedupe_candidates(sorted(rows, key=_sort_key))

    # Partition. Exactly one lane per article, and global is tested first —
    # that ordering IS the precedence rule for a story that is both (an EU
    # tariff on cloud services is a macro story that happens to touch tech).
    # Global gets precedence because it is the scarcer bucket: routing
    # ambiguous articles to tech would starve it while tech fills on its own.
    buckets: dict[str, list[dict]] = {"global": [], "tech": [], "markets": []}
    for row in rows:
        if is_global(row):
            buckets["global"].append(row)
        elif is_tech(row):
            buckets["tech"].append(row)
        else:
            buckets["markets"].append(row)

    chosen: dict[str, list[dict]] = {"global": [], "tech": [], "markets": []}
    taken = 0

    def take(lane: str, n: int) -> None:
        """Move up to n more rows into a lane, clamped by supply and by cap."""
        nonlocal taken
        start = len(chosen[lane])
        n = max(0, min(n, len(buckets[lane]) - start, cap - taken))
        chosen[lane].extend(buckets[lane][start : start + n])
        taken += n

    # Guaranteed minimums for the two lead lanes.
    take("global", LANE_MIN["global"])
    take("tech", LANE_MIN["tech"])

    # Markets claims its share BEFORE global and tech expand to their maximum.
    # Without this ordering a busy day (global 3 + tech 3) fills the cap and
    # markets is permanently empty, which is not what "the remainder" means.
    take("markets", cap - LANE_MIN["global"] - LANE_MIN["tech"])

    # A lane fell short of its minimum, or markets had nothing. Spare slots
    # flow back in lane order, up to each lane's maximum.
    take("global", LANE_MAX["global"] - len(chosen["global"]))
    take("tech", LANE_MAX["tech"] - len(chosen["tech"]))
    take("markets", cap - taken)

    # Still short because a lane ran dry. Relax the per-lane maximum: shipping
    # six articles beats shipping four to honour a quota that only exists to
    # stop one lane dominating.
    for lane in LANE_ORDER:
        if taken >= cap:
            break
        take(lane, cap - taken)

    return [(LANE_LABELS[lane], chosen[lane]) for lane in LANE_ORDER if chosen[lane]]
