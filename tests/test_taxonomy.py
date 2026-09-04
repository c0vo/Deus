"""Tests for data.taxonomy — briefing lane predicates and quota selection."""

import pytest

from data.taxonomy import (
    BRIEFING_MAX_ITEMS,
    LANE_LABELS,
    _dedupe_candidates,
    is_global,
    is_tech,
    select_briefing_lanes,
)


def row(headline, *, score=8.0, sectors=None, source="cnbc", event_type="general",
        summary="", url=None, published_at="2026-08-09T00:00:00Z"):
    """Build a briefing candidate row shaped like get_briefing_candidates output."""
    import json
    return {
        "headline": headline,
        "summary": "",
        "classification_summary": summary,
        "importance_score": score,
        "url": url or f"https://example.com/{abs(hash(headline)) % 10**8}",
        "affected_sectors": json.dumps(sectors or []),
        "affected_tickers": "[]",
        "source_name": source,
        "published_at": published_at,
        "sentiment_score": 0.0,
        "suggested_direction": "neutral",
        "event_type": event_type,
    }


# ── Global predicate ────────────────────────────────────────────────────────

class TestIsGlobal:

    def test_macro_event_type(self):
        assert is_global(row("Fed holds rates", event_type="macro")) is True

    def test_geopolitical_event_type(self):
        assert is_global(row("Sanctions widen", event_type="geopolitical")) is True

    def test_macro_source(self):
        assert is_global(row("Statement released", source="fed_press")) is True

    def test_korea_times_prefix_matches(self):
        assert is_global(row("Won weakens", source="korea_times_economy")) is True

    def test_macro_keyword_in_headline(self):
        assert is_global(row("New tariffs on imports announced")) is True

    def test_macro_keyword_in_summary(self):
        assert is_global(row("Report published", summary="CPI came in hotter than expected")) is True

    def test_plain_company_news_is_not_global(self):
        assert is_global(row("Acme announces new CFO", event_type="personnel")) is False

    def test_antitrust_alone_is_not_global(self):
        """Antitrust is regulatory, not macro — it must not pull tech into Global."""
        assert is_global(row("EU opens antitrust probe into cloud", event_type="regulatory")) is False


# ── Tech predicate ──────────────────────────────────────────────────────────

class TestIsTech:

    @pytest.mark.parametrize("sector", [
        "Technology", "Information Technology", "Semiconductors", "Software",
        "Artificial Intelligence", "Cloud Computing", "Cybersecurity",
        "Telecommunications", "E-commerce", "Consumer Electronics",
    ])
    def test_real_sector_vocabulary_matches(self, sector):
        """Every one of these appears verbatim in the production database."""
        assert is_tech(row("Some headline", sectors=[sector])) is True

    def test_tech_source(self):
        assert is_tech(row("Show HN: a thing", source="hackernews")) is True

    def test_tech_keyword_in_headline(self):
        assert is_tech(row("Nvidia unveils new GPU")) is True

    def test_cryptocurrency_is_not_tech(self):
        """Crypto trades as an asset class, so it belongs in Markets."""
        assert is_tech(row("Bitcoin rallies", sectors=["Cryptocurrency"])) is False

    def test_energy_is_not_tech(self):
        assert is_tech(row("Oil spikes on supply cut", sectors=["Energy"])) is False

    def test_word_boundaries_prevent_substring_matches(self):
        """'ai' must not match inside Retail, Ukraine, or chain."""
        assert is_tech(row("Retail sales chain stores report", sectors=["Retail"])) is False

    def test_sectors_accepts_a_plain_list(self):
        r = row("Chip demand rises")
        r["affected_sectors"] = ["Semiconductors"]
        assert is_tech(r) is True

    def test_malformed_sectors_does_not_raise(self):
        r = row("Some headline")
        r["affected_sectors"] = "not valid json"
        assert is_tech(r) is False


# ── Overlap precedence ──────────────────────────────────────────────────────

class TestOverlapPrecedence:

    def test_global_wins_when_an_article_is_both(self):
        r = row("EU tariffs on cloud services announced", sectors=["Cloud Computing"])
        assert is_global(r) is True
        assert is_tech(r) is True

        lanes = dict(select_briefing_lanes([r]))
        assert LANE_LABELS["global"] in lanes
        assert LANE_LABELS["tech"] not in lanes


# ── Selection and quotas ────────────────────────────────────────────────────

def _pool(n_global=0, n_tech=0, n_markets=0):
    rows = []
    for i in range(n_global):
        rows.append(row(f"Central bank decision {i}", event_type="macro", score=9.0 - i * 0.1))
    for i in range(n_tech):
        rows.append(row(f"Chipmaker news {i}", sectors=["Semiconductors"], score=8.5 - i * 0.1))
    for i in range(n_markets):
        rows.append(row(f"Utility merger {i}", sectors=["Utilities"], event_type="merger",
                        score=8.0 - i * 0.1))
    return rows


class TestSelectBriefingLanes:

    def test_empty_pool_returns_empty(self):
        assert select_briefing_lanes([]) == []

    def test_never_exceeds_the_cap(self):
        lanes = select_briefing_lanes(_pool(10, 10, 10))
        assert sum(len(a) for _, a in lanes) == BRIEFING_MAX_ITEMS

    def test_balanced_pool_gives_two_each(self):
        lanes = dict(select_briefing_lanes(_pool(5, 5, 5)))
        assert len(lanes[LANE_LABELS["global"]]) == 2
        assert len(lanes[LANE_LABELS["tech"]]) == 2
        assert len(lanes[LANE_LABELS["markets"]]) == 2

    def test_short_global_lane_gives_its_slots_away(self):
        lanes = dict(select_briefing_lanes(_pool(1, 5, 5)))
        assert len(lanes[LANE_LABELS["global"]]) == 1
        assert sum(len(a) for a in lanes.values()) == BRIEFING_MAX_ITEMS

    def test_empty_lane_is_omitted_not_rendered_blank(self):
        labels = [label for label, _ in select_briefing_lanes(_pool(0, 5, 5))]
        assert LANE_LABELS["global"] not in labels

    def test_busy_market_day_still_leaves_room_for_markets(self):
        """Global and Tech must not consume the whole cap between them."""
        lanes = dict(select_briefing_lanes(_pool(10, 10, 3)))
        assert len(lanes[LANE_LABELS["markets"]]) >= 1

    def test_lanes_come_back_in_order(self):
        labels = [label for label, _ in select_briefing_lanes(_pool(3, 3, 3))]
        assert labels == [LANE_LABELS["global"], LANE_LABELS["tech"], LANE_LABELS["markets"]]

    def test_highest_scoring_article_is_selected_first(self):
        rows = _pool(0, 0, 4)
        rows[2]["importance_score"] = 9.9
        top = select_briefing_lanes(rows)[0][1][0]
        assert top["importance_score"] == 9.9

    def test_selection_is_deterministic_across_orderings(self):
        rows = _pool(3, 3, 3)
        a = select_briefing_lanes(list(rows))
        b = select_briefing_lanes(list(reversed(rows)))
        assert [r["url"] for _, arts in a for r in arts] == [r["url"] for _, arts in b for r in arts]

    def test_thin_pool_returns_what_exists(self):
        lanes = select_briefing_lanes(_pool(1, 1, 0))
        assert sum(len(a) for _, a in lanes) == 2


# ── Duplicate suppression ───────────────────────────────────────────────────

class TestDedupeCandidates:

    def test_identical_summaries_collapse_to_one(self):
        """The live database holds such pairs with no duplicate_of link."""
        shared = "Dominion's merger with NextEra lifts its EPS growth target."
        rows = [
            row("The $67B NextEra-Dominion merger triggered its clock", summary=shared, score=7.5),
            row("Is Dominion Energy Stock a Buy After the Merger?", summary=shared, score=7.5),
        ]
        assert len(_dedupe_candidates(rows)) == 1

    def test_the_higher_scoring_copy_survives(self):
        shared = "Same story, two outlets."
        rows = [
            row("Version A", summary=shared, score=9.0),
            row("Version B", summary=shared, score=7.0),
        ]
        assert _dedupe_candidates(rows)[0]["headline"] == "Version A"

    def test_whitespace_differences_still_count_as_duplicates(self):
        rows = [
            row("A", summary="Shared summary text."),
            row("B", summary="  Shared   summary\n text.  "),
        ]
        assert len(_dedupe_candidates(rows)) == 1

    def test_distinct_summaries_are_both_kept(self):
        rows = [
            row("Fed raises rates by 25bps", summary="The Fed hiked."),
            row("Fed signals pause after hike", summary="The Fed may stop."),
        ]
        assert len(_dedupe_candidates(rows)) == 2

    def test_missing_summaries_are_never_treated_as_duplicates(self):
        rows = [row("First story", summary=""), row("Second story", summary="")]
        assert len(_dedupe_candidates(rows)) == 2

    def test_duplicates_do_not_consume_briefing_slots(self):
        shared = "One story reported twice."
        rows = _pool(2, 2, 0) + [
            row("Dup one", summary=shared, sectors=["Utilities"], score=7.9),
            row("Dup two", summary=shared, sectors=["Utilities"], score=7.8),
        ]
        lanes = dict(select_briefing_lanes(rows))
        assert len(lanes[LANE_LABELS["markets"]]) == 1
