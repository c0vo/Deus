"""
Geo Tagger

Assigns ISO 3166-1 alpha-2 country codes to articles.

New articles get their countries from the classifier, which has the full text
and can reason about where impact lands. This module is the cheap path: a
gazetteer of country names, major cities, demonyms and currencies used to
backfill the articles that were ingested before the classifier knew to ask.
It costs no API calls, so the whole archive can be tagged for free.

Matching is deliberately conservative — a missing tag is better than a wrong
one, because the globe reads as authoritative.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from config.logging_config import get_logger
from data.database import Database

log = get_logger(__name__)

# code -> (display name, [aliases]). Aliases cover demonyms, capitals, major
# financial centres, currencies and central banks — the words that actually
# appear in market copy.
GAZETTEER: dict[str, tuple[str, list[str]]] = {
    "US": ("United States", [
        "united states", "u.s.", "us economy", "america", "american",
        "washington", "new york", "wall street", "federal reserve", "the fed",
        "fomc", "nasdaq", "s&p 500", "dow jones", "treasury yields", "sec",
    ]),
    "CN": ("China", [
        "china", "chinese", "beijing", "shanghai", "shenzhen", "hong kong",
        "pboc", "yuan", "renminbi", "hang seng", "csi 300",
    ]),
    "JP": ("Japan", [
        "japan", "japanese", "tokyo", "bank of japan", "boj", "yen", "nikkei",
    ]),
    "KR": ("South Korea", [
        "south korea", "korean", "seoul", "kospi", "bank of korea", "won",
        "samsung electronics", "sk hynix",
    ]),
    "IN": ("India", [
        "india", "indian", "mumbai", "new delhi", "reserve bank of india",
        "rupee", "sensex", "nifty",
    ]),
    "GB": ("United Kingdom", [
        "united kingdom", "britain", "british", "london", "bank of england",
        "ftse", "sterling", "pound sterling", "uk economy",
    ]),
    "DE": ("Germany", [
        "germany", "german", "berlin", "frankfurt", "bundesbank", "dax",
    ]),
    "FR": ("France", ["france", "french", "paris", "cac 40"]),
    "IT": ("Italy", ["italy", "italian", "rome", "milan", "ftse mib"]),
    "ES": ("Spain", ["spain", "spanish", "madrid", "ibex"]),
    "NL": ("Netherlands", ["netherlands", "dutch", "amsterdam", "asml"]),
    "CH": ("Switzerland", [
        "switzerland", "swiss", "zurich", "swiss national bank", "franc",
    ]),
    "RU": ("Russia", ["russia", "russian", "moscow", "kremlin", "rouble", "ruble"]),
    "UA": ("Ukraine", ["ukraine", "ukrainian", "kyiv", "kiev"]),
    "CA": ("Canada", [
        "canada", "canadian", "toronto", "ottawa", "bank of canada", "tsx",
    ]),
    "MX": ("Mexico", ["mexico", "mexican", "mexico city", "peso"]),
    "BR": ("Brazil", ["brazil", "brazilian", "sao paulo", "bovespa", "real"]),
    "AR": ("Argentina", ["argentina", "argentine", "buenos aires"]),
    "AU": ("Australia", [
        "australia", "australian", "sydney", "reserve bank of australia", "asx",
    ]),
    "NZ": ("New Zealand", ["new zealand", "wellington", "auckland"]),
    "TW": ("Taiwan", ["taiwan", "taiwanese", "taipei", "tsmc"]),
    "SG": ("Singapore", ["singapore", "singaporean"]),
    "SA": ("Saudi Arabia", ["saudi arabia", "saudi", "riyadh", "aramco", "opec"]),
    "AE": ("United Arab Emirates", ["united arab emirates", "uae", "dubai", "abu dhabi"]),
    "IL": ("Israel", ["israel", "israeli", "tel aviv"]),
    "IR": ("Iran", ["iran", "iranian", "tehran"]),
    "TR": ("Turkey", ["turkey", "turkish", "ankara", "istanbul", "lira"]),
    "EG": ("Egypt", ["egypt", "egyptian", "cairo", "suez"]),
    "ZA": ("South Africa", ["south africa", "johannesburg", "rand"]),
    "NG": ("Nigeria", ["nigeria", "nigerian", "lagos"]),
    "ID": ("Indonesia", ["indonesia", "indonesian", "jakarta"]),
    "TH": ("Thailand", ["thailand", "thai", "bangkok"]),
    "VN": ("Vietnam", ["vietnam", "vietnamese", "hanoi"]),
    "MY": ("Malaysia", ["malaysia", "malaysian", "kuala lumpur"]),
    "PH": ("Philippines", ["philippines", "philippine", "manila"]),
    "PL": ("Poland", ["poland", "polish", "warsaw", "zloty"]),
    "SE": ("Sweden", ["sweden", "swedish", "stockholm", "riksbank", "krona"]),
    "NO": ("Norway", ["norway", "norwegian", "oslo"]),
    "DK": ("Denmark", ["denmark", "danish", "copenhagen"]),
    "FI": ("Finland", ["finland", "finnish", "helsinki"]),
    "IE": ("Ireland", ["ireland", "irish", "dublin"]),
    "PT": ("Portugal", ["portugal", "portuguese", "lisbon"]),
    "GR": ("Greece", ["greece", "greek", "athens"]),
    "QA": ("Qatar", ["qatar", "qatari", "doha"]),
    "CL": ("Chile", ["chile", "chilean", "santiago"]),
    "CO": ("Colombia", ["colombia", "colombian", "bogota"]),
    "PE": ("Peru", ["peru", "peruvian", "lima"]),
    "PK": ("Pakistan", ["pakistan", "pakistani", "islamabad", "karachi"]),
    "BD": ("Bangladesh", ["bangladesh", "dhaka"]),
    "VE": ("Venezuela", ["venezuela", "venezuelan", "caracas"]),
}

# Short forms that are only unambiguous when capitalised. Matched case
# sensitively so "US" the country is caught but "us" the pronoun is not, and
# "Fed" the central bank is caught but "fed" the verb is not.
STRICT_ALIASES: dict[str, list[str]] = {
    "US": ["US", "U.S.", "USA", "Fed", "Fed's", "Treasury"],
    "GB": ["UK", "U.K.", "BoE"],
    "CN": ["PBOC", "PBoC"],
    "JP": ["BOJ", "BoJ"],
    "KR": ["BOK"],
    "EU": ["EU", "ECB"],
    "IN": ["RBI"],
    "AU": ["RBA"],
}

# The EU is not a country and has no ISO-3166 alpha-2 entry on the map, so it
# is deliberately absent from GAZETTEER; give it a display name here.
_EXTRA_NAMES = {"EU": "European Union"}

# Longest alias first, so "south korea" wins over "korea"-style substrings and
# "new zealand" is not shadowed by a shorter neighbour.
_PATTERNS: list[tuple[str, re.Pattern]] = sorted(
    (
        (code, re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE))
        for code, (_, aliases) in GAZETTEER.items()
        for alias in aliases
    ),
    key=lambda pair: -len(pair[1].pattern),
)

_STRICT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (code, re.compile(r"\b" + re.escape(alias) + r"\b"))
    for code, aliases in STRICT_ALIASES.items()
    for alias in aliases
]

MAX_COUNTRIES = 3


def country_name(code: str) -> str:
    """Display name for a code, falling back to the code itself."""
    entry = GAZETTEER.get(code.upper())
    if entry:
        return entry[0]
    return _EXTRA_NAMES.get(code.upper(), code)


def extract_countries(*texts: Optional[str], limit: int = MAX_COUNTRIES) -> list[str]:
    """
    Return up to ``limit`` country codes mentioned in the given texts.

    Ranked by how often a country's aliases appear, so the subject of the story
    outranks a country named only in passing.
    """
    blob = " ".join(t for t in texts if t)
    if not blob:
        return []

    scores: dict[str, int] = {}
    for code, pattern in _PATTERNS:
        hits = len(pattern.findall(blob))
        if hits:
            scores[code] = scores.get(code, 0) + hits

    for code, pattern in _STRICT_PATTERNS:
        hits = len(pattern.findall(blob))
        if hits:
            scores[code] = scores.get(code, 0) + hits

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [code for code, _ in ranked[:limit]]


class GeoTagger:
    """Backfills country tags on articles that predate classifier geo output."""

    def __init__(self, db: Database):
        self.db = db

    def backfill(self, limit: int = 500) -> int:
        """
        Tag a bounded batch of untagged articles. Returns the number updated.

        Every article examined is written back — an empty list included — so
        the backlog converges rather than re-examining the same unmatchable
        rows forever.
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, headline, summary, classification_summary
                FROM articles
                WHERE countries IS NULL
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            if not rows:
                return 0

            updated = 0
            for row in rows:
                codes = extract_countries(
                    row["headline"], row["summary"], row["classification_summary"]
                )
                conn.execute(
                    "UPDATE articles SET countries = ? WHERE id = ?",
                    (json.dumps(codes), row["id"]),
                )
                if codes:
                    updated += 1

        log.info("geo_tagger.backfill", examined=len(rows), tagged=updated)
        return updated
