"""
Focused tests for the weekly-tip composer and its fact-checking boundary.

The feature's whole claim is that every number in the Sunday message is
traceable to our own data, so the tests that matter are the ones guarding the
seam: exactly one LLM call, over a FACTS block that is also the corpus the
verifier checks against, and a message that still goes out — saying why — when
the model is unset, fails, or returns nothing that survives verification.

`complete` is patched as imported into `pipeline.weekly_tip`, never `config.llm`:
the module bound the name at import time, so patching the source has no effect.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from bot.formatters import chunk_html, render_weekly_tip
from config.settings import settings
from pipeline.weekly_tip import WEEKLY_TIP_PROMPT, Tip, WeeklyTipComposer


def facts() -> dict:
    """A minimal, self-contained FACTS block with real quotable numbers."""
    return {
        "period_start": "2026-09-13",
        "period_end": "2026-09-19",
        "model": "test/weekly",
        "tip_status": "pending",
        "seasonality": [{
            "name": "September effect",
            "window": "September",
            "stat_line": (
                "SPY September: median -1.5%, up in 44% of 34 years since 1993; "
                "worst 2004 -4.0%, best 1999 +1.1%"
            ),
            "numbers": {},
        }],
        "events": [{
            "date": "2026-09-16", "name": "FOMC rate decision",
            "kind": "fomc", "importance": 3, "confirmed": True,
        }],
        "earnings": [],
        "ipos": [],
        "news_warnings": {},
        "performance": [],
    }


def tip(*, precedent: str = "SPY's median was -1.5% across 34 years.") -> Tip:
    return Tip(
        title="September precedent",
        precedent=precedent,
        evidence="SPY September effect and the September 16 FOMC decision.",
        action="Watch the decision and keep the historical result in context.",
        severity="watch",
    )


class TestWeeklyTipCompose:
    def test_unconfigured_model_returns_facts_only_without_request(self):
        composer = WeeklyTipComposer(MagicMock())
        payload = facts()

        with patch.object(settings, "model_weekly_tip", ""), \
             patch("pipeline.weekly_tip.complete", new=AsyncMock()) as complete:
            result = asyncio.run(composer.compose(payload))

        assert result == []
        assert payload["tip_status"] == "not_configured"
        complete.assert_not_awaited()

    def test_supported_tip_is_returned_from_one_schema_call(self):
        composer = WeeklyTipComposer(MagicMock())
        payload = facts()
        response = MagicMock(parsed=[tip()], text="")
        tracked = MagicMock()

        with patch.object(settings, "model_weekly_tip", "test/weekly"), \
             patch("pipeline.weekly_tip.is_llm_configured", return_value=True), \
             patch("pipeline.weekly_tip.track_llm", return_value=tracked), \
             patch("pipeline.weekly_tip.complete", new=AsyncMock(return_value=response)) as complete:
            result = asyncio.run(composer.compose(payload))

        assert result == [tip()]
        assert payload["tip_status"] == "ok"
        assert payload["tips_returned"] == 1
        assert payload["tips_dropped"] == 0
        complete.assert_awaited_once()
        assert complete.await_args.kwargs["schema"] == list[Tip]
        assert complete.await_args.kwargs["reasoning"] == "low"
        # The prompt the model sees is the rules plus the same facts text the
        # verifier reads, and nothing else.
        prompt = complete.await_args.kwargs["prompt"]
        assert prompt == WEEKLY_TIP_PROMPT + composer.render_facts_text(payload)

    def test_prose_wrapped_json_falls_back_to_parse_structured(self):
        # `strict` is off on the json_schema, so a model can still wrap the array
        # in a code fence. Every schema call site keeps this fallback.
        composer = WeeklyTipComposer(MagicMock())
        payload = facts()
        body = json.dumps([tip().model_dump()])
        response = MagicMock(parsed=None, text=f"```json\n{body}\n```")

        with patch.object(settings, "model_weekly_tip", "test/weekly"), \
             patch("pipeline.weekly_tip.is_llm_configured", return_value=True), \
             patch("pipeline.weekly_tip.track_llm", return_value=MagicMock()), \
             patch("pipeline.weekly_tip.complete", new=AsyncMock(return_value=response)):
            result = asyncio.run(composer.compose(payload))

        assert result == [tip()]
        assert payload["tip_status"] == "ok"

    def test_a_failed_call_degrades_instead_of_raising(self):
        composer = WeeklyTipComposer(MagicMock())
        payload = facts()

        with patch.object(settings, "model_weekly_tip", "test/weekly"), \
             patch("pipeline.weekly_tip.is_llm_configured", return_value=True), \
             patch("pipeline.weekly_tip.track_llm", return_value=MagicMock()), \
             patch("pipeline.weekly_tip.complete",
                   new=AsyncMock(side_effect=RuntimeError("429"))):
            result = asyncio.run(composer.compose(payload))

        assert result == []
        assert payload["tip_status"] == "failed"


class TestFactsOnlyRender:
    """A message always goes out, and it always says why it is facts-only."""

    def test_unconfigured_model_still_sends_the_measured_facts(self):
        composer = WeeklyTipComposer(MagicMock())
        payload = facts()

        with patch.object(settings, "model_weekly_tip", ""):
            tips = asyncio.run(composer.compose(payload))

        text = render_weekly_tip(payload, tips)

        assert payload["tip_status"] == "not_configured"
        assert text.strip()
        assert "MODEL_WEEKLY_TIP not configured" in text
        # The precedent and the dated event survive the model being absent —
        # that is the half of the tip that never needed one.
        assert "SPY September: median -1.5%" in text
        assert "FOMC rate decision" in text
        assert len(chunk_html(text)) == 1

    def test_the_facts_text_carries_every_number_the_model_may_quote(self):
        # render_facts_text is both the prompt and the verifier's corpus, so a
        # statistic missing from it is a tip that cannot be written.
        text = WeeklyTipComposer.render_facts_text(facts())
        assert "2026-09-13 to 2026-09-19" in text
        assert "up in 44% of 34 years since 1993" in text
        assert "2026-09-16" in text

    def test_number_not_present_in_facts_is_dropped(self):
        composer = WeeklyTipComposer(MagicMock())
        payload = facts()
        response = MagicMock(parsed=[tip(precedent="SPY fell 8.8% in 50 years.")], text="")

        with patch.object(settings, "model_weekly_tip", "test/weekly"), \
             patch("pipeline.weekly_tip.is_llm_configured", return_value=True), \
             patch("pipeline.weekly_tip.track_llm", return_value=MagicMock()), \
             patch("pipeline.weekly_tip.complete", new=AsyncMock(return_value=response)):
            result = asyncio.run(composer.compose(payload))

        assert result == []
        assert payload["tip_status"] == "empty"
        assert payload["tips_returned"] == 1
        assert payload["tips_dropped"] == 1


class TestWeeklyTipPersistence:
    def test_persists_the_audit_trail_and_publishes_notification(self):
        db = MagicMock()
        db.insert_digest.return_value = 42
        composer = WeeklyTipComposer(db)
        publish = AsyncMock()
        payload = facts()

        with patch("pipeline.weekly_tip.event_bus.publish", publish):
            digest_id = asyncio.run(
                composer.persist_and_publish(payload, [tip()], "<b>Weekly Tip</b>")
            )

        assert digest_id == 42
        call = db.insert_digest.call_args
        assert call.args == ("weekly_tip", "<b>Weekly Tip</b>")
        assert call.kwargs["facts_json"]["tips"] == [tip().model_dump()]
        publish.assert_awaited_once()
        assert publish.await_args.args[0] == "weekly_tip"
        assert publish.await_args.args[1]["id"] == 42
