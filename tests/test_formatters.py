"""Tests for bot.formatters — HTML rendering and Telegram message chunking."""

import pytest

from bot.formatters import (
    BRIEFING_SUMMARY_MAX,
    EMPTY_BRIEFING_TEXT,
    TELEGRAM_CHUNK_LIMIT,
    chunk_html,
    escape_html,
    render_briefing,
)


def article(headline="A headline", score=8.0, summary="A summary.",
            url="https://example.com/a", classification_summary=None):
    return {
        "headline": headline,
        "importance_score": score,
        "summary": summary,
        "classification_summary": classification_summary,
        "url": url,
    }


# ── chunk_html ──────────────────────────────────────────────────────────────

class TestChunkHtml:

    def test_short_text_stays_one_chunk(self):
        assert chunk_html("short message") == ["short message"]

    def test_empty_text_yields_no_chunks(self):
        assert chunk_html("") == []

    def test_splits_on_paragraph_boundaries(self):
        text = "\n\n".join(["x" * 2000, "y" * 2000])
        chunks = chunk_html(text, limit=3000)
        assert len(chunks) == 2
        assert chunks[0].startswith("x")
        assert chunks[1].startswith("y")

    def test_every_chunk_respects_the_limit(self):
        text = "\n\n".join("paragraph %d %s" % (i, "z" * 500) for i in range(30))
        assert all(len(c) <= 1000 for c in chunk_html(text, limit=1000))

    def test_oversize_paragraph_is_hard_split(self):
        """A single paragraph over the limit must not be emitted whole."""
        chunks = chunk_html("q" * 5000, limit=1000)
        assert len(chunks) == 5
        assert all(len(c) <= 1000 for c in chunks)

    def test_oversize_paragraph_never_produces_an_empty_chunk(self):
        chunks = chunk_html("intro\n\n" + "q" * 5000, limit=1000)
        assert all(c.strip() for c in chunks)

    def test_content_is_preserved_across_chunks(self):
        chunks = chunk_html("m" * 4000, limit=1000)
        assert "".join(chunks) == "m" * 4000

    def test_default_limit_fits_telegram(self):
        text = "\n\n".join("line %d" % i + "w" * 400 for i in range(40))
        assert all(len(c) <= TELEGRAM_CHUNK_LIMIT for c in chunk_html(text))


# ── render_briefing ─────────────────────────────────────────────────────────

class TestRenderBriefing:

    def test_renders_lane_labels_and_headlines(self):
        text = render_briefing([("🌍 GLOBAL & MACRO", [article(headline="Fed holds")])])
        assert "GLOBAL &amp; MACRO" in text
        assert "Fed holds" in text

    def test_prefers_classification_summary(self):
        text = render_briefing([("L", [article(summary="raw", classification_summary="classified")])])
        assert "classified" in text
        assert "raw" not in text

    def test_falls_back_to_summary(self):
        text = render_briefing([("L", [article(summary="raw", classification_summary=None)])])
        assert "raw" in text

    def test_null_summary_never_renders_the_string_none(self):
        """The scheduler used to print a literal 'None' for unclassified rows."""
        text = render_briefing([("L", [article(summary=None, classification_summary=None)])])
        assert "None" not in text

    def test_long_summary_is_truncated(self):
        text = render_briefing([("L", [article(classification_summary="s" * 900)])])
        assert "s" * 900 not in text
        assert "…" in text

    def test_truncation_happens_before_escaping(self):
        """Escaping can sextuple a character, so trimming after would not bound it."""
        text = render_briefing([("L", [article(classification_summary="<&>" * 400)])])
        assert len(text) < BRIEFING_SUMMARY_MAX * 6 + 1000

    def test_html_special_characters_are_escaped(self):
        text = render_briefing([("L", [article(headline="AT&T beats <expectations>")])])
        assert "&amp;" in text
        assert "<expectations>" not in text

    def test_score_is_shown_to_one_decimal(self):
        assert "(8.5)" in render_briefing([("L", [article(score=8.5)])])

    def test_missing_score_does_not_raise(self):
        assert "(0.0)" in render_briefing([("L", [article(score=None)])])

    def test_multiple_lanes_appear_in_given_order(self):
        text = render_briefing([
            ("FIRST", [article(headline="one")]),
            ("SECOND", [article(headline="two")]),
        ])
        assert text.index("FIRST") < text.index("SECOND")

    def test_no_lanes_renders_title_only(self):
        assert "Daily Market Briefing" in render_briefing([])

    def test_typical_six_article_brief_is_one_message(self):
        """Representative of live data, which measures ~3.1k characters."""
        lane = [article(headline="H" * 90, classification_summary="S" * 300,
                        url="https://www.cnbc.com/2026/08/09/a-fairly-long-story-slug.html")] * 2
        lanes = [("🌍 GLOBAL & MACRO", lane), ("💻 TECH", lane), ("📊 MARKETS", lane)]
        assert len(chunk_html(render_briefing(lanes))) == 1

    def test_worst_case_brief_still_sends_within_the_limit(self):
        """
        Maximal headlines plus 290-character google_news URLs push a six-item
        brief past one message. Splitting is correct and expected there; what
        must hold is that every part is sendable and nothing is dropped.
        """
        lane = [article(headline="H" * 120, classification_summary="S" * 400,
                        url="https://news.google.com/rss/articles/" + "A" * 250)] * 2
        lanes = [("🌍 GLOBAL & MACRO", lane), ("💻 TECH", lane), ("📊 MARKETS", lane)]
        rendered = render_briefing(lanes)
        chunks = chunk_html(rendered)

        assert all(len(c) <= TELEGRAM_CHUNK_LIMIT for c in chunks)
        assert len(chunks) <= 2
        assert "Read more" in chunks[-1]


class TestEscapeHtml:

    def test_none_becomes_empty_string(self):
        assert escape_html(None) == ""

    def test_quotes_are_escaped_for_attribute_safety(self):
        assert "'" not in escape_html("it's")


def test_empty_briefing_text_is_user_facing():
    assert EMPTY_BRIEFING_TEXT.strip()
