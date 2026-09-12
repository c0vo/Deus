"""Tests for bot.formatters — HTML rendering and Telegram message chunking."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.formatters import (
    ALERT_FIELD_MAX,
    BRIEFING_SUMMARY_MAX,
    EMPTY_BRIEFING_TEXT,
    STANCE_FIELD_MAX,
    STANCE_THESIS_MAX,
    TELEGRAM_CHUNK_LIMIT,
    chunk_html,
    escape_html,
    render_briefing,
    render_price_alert,
    render_weekly_tip,
    render_stance_message,
)
from pipeline.grounded_answer import (
    GradeVerdict,
    GroundedAnswer,
    MoveExplanation,
    classify_session_context,
)
from pipeline.daily_stance import (
    ARROW_DOWNGRADED,
    ARROW_NEW,
    NO_CALL,
    UNAVAILABLE,
    Stance,
    StanceRow,
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


class TestRenderWeeklyTip:
    def test_sections_and_model_tip_are_rendered_and_escaped(self):
        tip = MagicMock(
            title="Rates <risk>",
            precedent="Median -1.5% & 34 years",
            evidence="FOMC 2026-09-16",
            action="Watch <the release>",
            severity="warning",
        )
        facts = {
            "period_start": "2026-09-13",
            "period_end": "2026-09-19",
            "seasonality": [{"name": "September", "stat_line": "SPY -1.5%"}],
            "events": [{"date": "2026-09-16", "name": "FOMC <decision>",
                        "importance": 3, "confirmed": True}],
            "news_warnings": {},
            "performance": [{"ticker": "SPY", "pct": -1.5, "vs_spy": None}],
        }

        text = render_weekly_tip(facts, [tip])

        for heading in ("Weekly Tip", "Seasonality", "Coming up", "Watch", "Your tickers"):
            assert heading in text
        assert "&lt;risk&gt;" in text
        assert "FOMC &lt;decision&gt;" in text
        assert "Rates <risk>" not in text

    def test_facts_only_status_is_visible(self):
        text = render_weekly_tip({"tip_status": "not_configured"}, [])
        assert "MODEL_WEEKLY_TIP not configured" in text


class TestEscapeHtml:

    def test_none_becomes_empty_string(self):
        assert escape_html(None) == ""

    def test_quotes_are_escaped_for_attribute_safety(self):
        assert "'" not in escape_html("it's")


def test_empty_briefing_text_is_user_facing():
    assert EMPTY_BRIEFING_TEXT.strip()


# ── AlertManager.send_html ──────────────────────────────────────────────────

class TestAlertManagerSendHtml:
    """
    The sender side of chunk_html.

    Telegram rejects a message over 4096 characters outright, so a long brief
    used to produce no message at all. send_html is the one place that loop
    lives now, and every scheduled push goes through it.
    """

    def _manager(self):
        from bot.alerts import AlertManager

        bot = MagicMock()
        bot.send_message = AsyncMock()
        manager = AlertManager(db=MagicMock(), bot=bot)
        manager.chat_id = "12345"
        return manager, bot

    def test_long_message_is_split_across_sends(self):
        manager, bot = self._manager()
        # Paragraphs, because that is where chunk_html is allowed to break.
        text = "\n\n".join(["x" * 500] * 10)
        assert len(text) > TELEGRAM_CHUNK_LIMIT

        asyncio.run(manager.send_html(text))

        assert bot.send_message.await_count > 1
        for call in bot.send_message.await_args_list:
            assert len(call.kwargs["text"]) <= TELEGRAM_CHUNK_LIMIT
            assert call.kwargs["parse_mode"] == "HTML"
            assert call.kwargs["chat_id"] == "12345"
        # Nothing may be dropped on the way out.
        sent = "".join(c.kwargs["text"] for c in bot.send_message.await_args_list)
        assert sent.count("x") == text.count("x")

    def test_short_message_is_one_send(self):
        manager, bot = self._manager()
        asyncio.run(manager.send_html("<b>short</b>"))
        assert bot.send_message.await_count == 1
        assert bot.send_message.await_args.kwargs["text"] == "<b>short</b>"

    def test_previews_are_disabled_by_default(self):
        manager, bot = self._manager()
        asyncio.run(manager.send_html("hello"))
        assert bot.send_message.await_args.kwargs["disable_web_page_preview"] is True

    def test_previews_can_be_kept(self):
        """Breaking-news alerts want the preview; nothing else does."""
        manager, bot = self._manager()
        asyncio.run(manager.send_html("hello", disable_preview=False))
        assert bot.send_message.await_args.kwargs["disable_web_page_preview"] is False


# ── render_price_alert ──────────────────────────────────────────────────────

def _source(n=1, title=None, url=None):
    return {
        "title": title or f"Headline {n}",
        "url": url or f"https://example.com/{n}",
        "source": "reuters",
        "published_at": "2026-09-11",
        "kind": "db",
    }


def _grounded(n_sources=3, **overrides):
    explanation = MoveExplanation(
        catalyst_found=True,
        catalyst_kind="company",
        cause="Guidance cut to $8.1B from $8.9B on 2026-09-11.",
        sustainability="Structural rather than one-off.",
        what_to_watch="The Q4 print on 2026-11-04.",
        source_indices=list(range(1, n_sources + 1)),
        **overrides,
    )
    return GroundedAnswer(
        text=explanation.cause,
        sources=[_source(i) for i in range(1, n_sources + 1)],
        grounded_by="db",
        grade=GradeVerdict(sufficient=True, specificity="specific", reason="ok"),
        explanation=explanation,
    )


def _unexplained():
    session = classify_session_context(-3.2, {"SPY": -2.1, "QQQ": -2.8})
    explanation = MoveExplanation(
        catalyst_found=False,
        catalyst_kind="macro",
        cause="No catalyst in the last 48h of news or on the web. "
              "Market-wide: SPY -2.10%, QQQ -2.80%.",
        sustainability="",
        what_to_watch="",
        source_indices=[],
    )
    return GroundedAnswer(
        text=explanation.cause, sources=[], grounded_by="none",
        grade=GradeVerdict(sufficient=False, specificity="generic", reason="x"),
        explanation=explanation, session=session,
    )


class TestRenderPriceAlert:

    def _render(self, **kwargs):
        defaults = dict(
            ticker="NVDA", pct=-3.24, price=1234.5, kind="price_drop",
            answer=_grounded(),
            index_context={"SPY": -2.1, "QQQ": -2.8},
            macro_lines=["CPI (September 2026) today 08:30 ET"],
            technical_line="Technical rating for NVDA (26 indicators): sell",
        )
        defaults.update(kwargs)
        return render_price_alert(**defaults)

    def test_fits_one_chunk_with_three_sources(self):
        """Sources on a second Telegram message are sources nobody reads."""
        text = self._render()
        assert len(chunk_html(text)) == 1

    def test_header_move_and_price(self):
        text = self._render()
        assert "PRICE DROP: NVDA" in text
        assert "3.24%" in text
        assert "$1,234.50" in text
        assert "is down" in text

    def test_session_context_line_is_present(self):
        text = self._render()
        assert "SPY -2.10%" in text
        assert "market-wide" in text

    def test_idiosyncratic_when_the_index_is_flat(self):
        text = self._render(index_context={"SPY": 0.1, "QQQ": 0.2})
        assert "idiosyncratic" in text

    def test_macro_line_is_rendered(self):
        assert "CPI (September 2026) today 08:30 ET" in self._render()

    def test_cause_and_what_to_watch_are_rendered(self):
        text = self._render()
        assert "Guidance cut to $8.1B" in text
        assert "What to watch" in text
        assert "2026-11-04" in text

    def test_at_most_three_sources_are_linked(self):
        text = self._render(answer=_grounded(n_sources=5))
        assert text.count("<a href=") == 3

    def test_sources_are_anchors_with_their_metadata(self):
        text = self._render()
        assert '<a href="https://example.com/1">Headline 1</a>' in text
        assert "reuters, 2026-09-11" in text

    def test_unexplained_move_says_so_and_cites_nothing(self):
        text = self._render(answer=_unexplained())
        assert "<a href=" not in text
        assert "No dated source supports a cause" in text
        assert "Market-wide" in text

    def test_escapes_html_in_a_headline(self):
        answer = _grounded(n_sources=1)
        answer.sources[0]["title"] = 'Fab <b>halted</b> & "paused"'
        text = render_price_alert(
            ticker="NVDA", pct=-3.2, price=100.0, kind="price_drop", answer=answer,
        )
        assert "Fab &lt;b&gt;halted&lt;/b&gt;" in text
        assert "<b>halted</b>" not in text

    def test_escapes_html_in_the_ticker(self):
        text = render_price_alert(
            ticker="<script>", pct=-3.2, price=1.0, answer=_unexplained(),
        )
        assert "<script>" not in text
        assert "&lt;script&gt;" in text

    def test_long_fields_are_truncated_before_escaping(self):
        answer = _grounded(n_sources=1)
        answer.explanation.cause = "x" * (ALERT_FIELD_MAX * 3)
        text = render_price_alert(
            ticker="NVDA", pct=-3.2, price=1.0, answer=answer,
        )
        assert "x" * (ALERT_FIELD_MAX + 1) not in text
        assert "x" * ALERT_FIELD_MAX in text
        assert "…" in text

    def test_volume_alert_states_the_multiple(self):
        text = self._render(kind="volume", vol_multiple=4.2)
        assert "VOLUME SPIKE: NVDA" in text
        assert "4.2x" in text

    def test_unknown_kind_falls_back_to_a_move_header(self):
        text = self._render(kind="something_new")
        assert "PRICE MOVE: NVDA" in text

    def test_absent_technical_rating_omits_the_section(self):
        assert "Technicals" not in self._render(technical_line="")


# ── render_stance_message ───────────────────────────────────────────────────

def stance_row(ticker="NVDA", action="TRIM", **over):
    defaults = {
        "conviction": "Medium",
        "thesis": "Daily rating flipped to Sell with RSI14 at 71.",
        "key_risk": "Analyst target still implies +14% upside.",
        "evidence_used": ["rsi", "technical_rating"],
        "what_would_change_my_mind": "A daily close back above $185.",
    }
    row_kwargs = {k: over.pop(k) for k in list(over)
                  if k in {"arrow", "prev_action", "cached_debate"}}
    defaults.update(over)
    return StanceRow.from_stance(
        Stance(ticker=ticker, action=action, **defaults), **row_kwargs
    )


class TestRenderStanceMessage:

    def test_renders_action_conviction_and_thesis(self):
        text = render_stance_message([stance_row()], model_slug="x/y",
                                     date="2026-09-12")
        assert "<b>NVDA</b> TRIM (Medium)" in text
        assert "RSI14 at 71" in text
        assert "<b>Risk:</b>" in text
        assert "<b>Changes mind:</b>" in text
        assert "rsi, technical_rating" in text

    def test_header_names_the_model(self):
        text = render_stance_message([stance_row()], model_slug="google/flash",
                                     date="2026-09-12")
        assert "model: google/flash" in text
        assert "2026-09-12" in text

    def test_header_warns_when_no_model_is_configured(self):
        """
        The note this replaced printed HOLD when MODEL_DAILY_ADVISOR was unset,
        so a misconfiguration was indistinguishable from a decision to hold.
        """
        rows = [StanceRow(ticker="QQQ", action=UNAVAILABLE,
                          thesis="MODEL_DAILY_ADVISOR is not configured.",
                          arrow="")]
        text = render_stance_message(rows, model_slug="", date="2026-09-12")
        assert "not configured" in text
        assert "HOLD" not in text
        assert UNAVAILABLE in text

    def test_failure_rows_get_a_warning_marker_not_an_arrow(self):
        rows = [StanceRow(ticker="QQQ", action=NO_CALL, arrow="")]
        text = render_stance_message(rows, model_slug="x/y")
        assert "⚠️ <b>QQQ</b>" in text

    def test_arrow_is_rendered(self):
        text = render_stance_message(
            [stance_row(arrow=ARROW_DOWNGRADED, prev_action="HOLD")],
            model_slug="x/y")
        assert f"{ARROW_DOWNGRADED} <b>NVDA</b>" in text
        assert "was HOLD" in text

    def test_unchanged_action_does_not_print_a_was_line(self):
        text = render_stance_message(
            [stance_row(action="HOLD", arrow=ARROW_NEW, prev_action="HOLD")],
            model_slug="x/y")
        assert "was HOLD" not in text

    def test_cached_debate_is_footnoted(self):
        row = stance_row(cached_debate={"direction": "SELL", "conviction": "High",
                                        "date": "2026-09-10"})
        text = render_stance_message([row], model_slug="x/y")
        assert "<b>5-day debate:</b> SELL (High, 2026-09-10)" in text

    def test_html_is_escaped(self):
        row = stance_row(thesis="Margins <compressed> & guidance cut",
                         key_risk='A "short squeeze"')
        text = render_stance_message([row], model_slug="a<b>c")
        assert "&lt;compressed&gt;" in text
        assert "&amp; guidance" in text
        assert "&quot;short squeeze&quot;" in text
        # Only the markup this renderer opened itself may survive.
        assert "<compressed>" not in text
        assert "a<b>c" not in text

    def test_long_fields_are_clipped_before_escaping(self):
        """
        Escaping can sextuple a character, so trimming afterwards would not
        bound the result — the budget has to be applied to the raw text.
        """
        row = stance_row(thesis="<" * 900, key_risk="&" * 900)
        text = render_stance_message([row], model_slug="x/y")
        assert text.count("&lt;") <= STANCE_THESIS_MAX
        assert text.count("&amp;") <= STANCE_FIELD_MAX + 1  # +1 for the ellipsis entity
        assert "…" in text

    def test_many_tickers_still_chunk_within_the_limit(self):
        rows = [stance_row(ticker=f"TK{i}") for i in range(25)]
        text = render_stance_message(rows, model_slug="x/y", date="2026-09-12")
        chunks = chunk_html(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= TELEGRAM_CHUNK_LIMIT
        # Every ticker survives the split.
        joined = "".join(chunks)
        for i in range(25):
            assert f"TK{i}" in joined

    def test_empty_batch_still_renders_a_header(self):
        text = render_stance_message([], model_slug="x/y", date="2026-09-12")
        assert "Daily Stance" in text
