"""
Tests for the daily stance engine.

Every test here exists because the note this replaced printed HOLD for three
different reasons — no news, no model, no response — and a reader could not tell
any of them from a considered decision to hold. So the assertions are mostly
negative: given a failure, the output must NOT say HOLD.

Network-free and LLM-free: `pipeline.daily_stance.complete` is patched as
imported into the module (it binds the name at import time, so patching
`config.llm.complete` would not take), and everything else runs against a
temporary SQLite file.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from unittest.mock import AsyncMock, patch

import pytest

from bot.formatters import render_stance_message
from config.settings import settings
from data.database import Database
from pipeline.daily_stance import (
    ARROW_DOWNGRADED,
    ARROW_NEW,
    ARROW_UNCHANGED,
    ARROW_UPGRADED,
    DAILY_STANCE_PROMPT,
    NO_CALL,
    UNAVAILABLE,
    DailyStanceEngine,
    Stance,
    StanceRow,
    compute_rsi14,
)
from pipeline.macro_calendar import today_et
from tests.conftest import make_llm_response

TICKER = "TEST"
MODEL = "test/stance-model"


@pytest.fixture
def db(tmp_path):
    database = Database(db_path=str(tmp_path / "stance.db"))
    database.initialize()
    _seed_prices(database, TICKER)
    _seed_technical_rating(database, TICKER)
    return database


@pytest.fixture
def engine(db):
    return DailyStanceEngine(db)


def _seed_prices(database: Database, ticker: str, *, start: float = 300.0,
                 step: float = -5.0, sessions: int = 30) -> None:
    """A 30-session straight-line decline.

    The slope is chosen so the last bar's 1-day return clears
    `alert_drop_pct_tracked` (-3.125% at 155 from 160): `material_changes` has to
    see a real move coming out of the fact sheet, not one injected by the test.
    """
    today = dt.date.today()
    rows = []
    for i in range(sessions):
        close = start + step * i
        rows.append({
            "date": (today - dt.timedelta(days=sessions - 1 - i)).isoformat(),
            "open": close, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": 1_000_000,
        })
    database.upsert_price_history(ticker, rows)


def _seed_technical_rating(database: Database, ticker: str) -> None:
    """One sell-side daily rating, so "no news" still leaves evidence to cite."""
    today = dt.date.today().isoformat()
    database.upsert_technical_ratings([{
        "ticker": ticker, "session_date": today, "timeframe": "1D",
        "summary_score": -0.64, "summary_label": "Sell",
        "ma_score": -0.80, "ma_label": "Strong Sell",
        "osc_score": -0.27, "osc_label": "Sell",
        "buy_votes": 2, "neutral_votes": 3, "sell_votes": 12,
        "bars_available": 250, "published_at": f"{today}T21:00:00+00:00",
    }])


def _stance(ticker: str, action: str, **over) -> Stance:
    data = {
        "ticker": ticker, "action": action, "conviction": "Medium",
        "thesis": "Daily rating is Sell with RSI14 at 0.0 and no news in 48h.",
        "key_risk": "A reversal off the 52-week low would strand the call.",
        "evidence_used": ["rsi", "technical_rating"],
        "what_would_change_my_mind": "A daily close back above $170.",
    }
    data.update(over)
    return Stance(**data)


def _row(ticker: str, action: str = "HOLD", *, arrow: str = ARROW_UNCHANGED,
         prev: str | None = None) -> StanceRow:
    return StanceRow.from_stance(_stance(ticker, action), arrow=arrow,
                                 prev_action=prev)


def _patched(parsed=None, text: str = "") -> tuple:
    """`complete` patched with a canned response, and a configured LLM client."""
    mock = AsyncMock(return_value=make_llm_response(text=text, parsed=parsed))
    return mock, patch("pipeline.daily_stance.complete", mock), \
        patch("pipeline.daily_stance.is_llm_configured", return_value=True)


# ── Fact sheet ──────────────────────────────────────────────────────────────

class TestFactSheet:

    def test_reads_the_seeded_series(self, engine):
        facts = engine.build_fact_sheet(TICKER)
        price = facts["price_block"]
        assert price["price"] == pytest.approx(155.0)
        # -5 on a 160 prior close. The threshold this has to clear is 3.0%.
        assert price["ret_1d"] == pytest.approx(-3.125, abs=0.01)
        assert price["rsi14"] == pytest.approx(0.0)
        assert price["pct_from_high"] < -40
        assert facts["ret_1d"] == price["ret_1d"]

    def test_absences_are_named_not_omitted(self, engine):
        """A missing section has to read as evidence, not as a gap to fill in."""
        block = engine.render_fact_block(engine.build_fact_sheet(TICKER))
        assert "no news in 48h" in block
        assert "no analyst coverage stored" in block
        assert "no live model prediction" in block
        assert "none in the last 5 days" in block
        assert "None" not in block

    def test_ml_line_writes_a_prior_row_as_no_edge(self, engine):
        """A no-edge row carries the base rate, which must not read as a call."""
        block = engine.render_fact_block({"ticker": TICKER, "ml": [
            {"horizon_days": 5, "direction": "UP", "confidence": 0.57,
             "probability_up": 0.57, "model_type": "prior"},
            {"horizon_days": 21, "direction": "DOWN", "confidence": 0.58,
             "probability_up": 0.42, "model_type": "universal"},
            {"horizon_days": 63, "direction": "UP", "confidence": 0.61,
             "probability_up": None, "model_type": "llm_only"},
        ]})
        assert ("- ML probability: 5d no edge (base 57% up); 21d DOWN 58%; "
                "63d UP 61% (llm_only)") in block

    def test_ml_block_prefers_the_model_row_over_a_debate_row(self, engine, db):
        """A debate restates the baseline in a multi_agent row; the model's own row speaks."""
        for model_type, direction, confidence in (("prior", "UP", 0.57),
                                                  ("multi_agent", "UNKNOWN", 0.0)):
            db.insert_prediction({
                "ticker": TICKER, "predicted_direction": direction,
                "confidence": confidence, "probability_up": 0.57,
                "feature_asof": dt.date.today().isoformat(), "horizon_days": 5,
                "model_type": model_type, "feature_snapshot": "{}",
                "llm_narrative": "", "resolve_after": "2099-01-01",
            })
        ml = engine.build_fact_sheet(TICKER)["ml"]
        assert [(m["horizon_days"], m["model_type"], m["direction"]) for m in ml] == [
            (5, "prior", "UP")
        ]

    def test_technical_rating_reaches_the_block(self, engine):
        block = engine.render_fact_block(engine.build_fact_sheet(TICKER))
        assert "1D=Sell" in block
        assert "12S" in block

    def test_unknown_ticker_says_so(self, engine):
        block = engine.render_fact_block(engine.build_fact_sheet("NOSUCH"))
        assert "NO price history stored" in block

    def test_rsi_needs_a_warmup(self):
        assert compute_rsi14([1.0, 2.0, 3.0]) is None
        assert compute_rsi14([100.0 + i for i in range(20)]) == pytest.approx(100.0)


# ── Prompt ──────────────────────────────────────────────────────────────────

class TestPrompt:

    def test_rules_forbid_hold_as_a_default(self, engine):
        prompt = engine.build_prompt({TICKER: engine.build_fact_sheet(TICKER)})
        assert "HOLD is not a default" in prompt
        assert "BUY/ADD" in prompt and "TRIM" in prompt

    def test_prompt_carries_the_fact_block(self, engine):
        prompt = engine.build_prompt({TICKER: engine.build_fact_sheet(TICKER)})
        assert f"FACTS for {TICKER}" in prompt
        assert "RSI14" in prompt
        assert "1D=Sell" in prompt

    async def test_sent_prompt_is_the_built_prompt(self, engine, monkeypatch):
        """What reaches `complete` has to be the fact sheet, not a summary of it."""
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        mock, p_complete, p_cfg = _patched(parsed=[_stance(TICKER, "SELL")])
        with p_complete, p_cfg:
            await engine.compose([TICKER])

        sent = mock.await_args.kwargs["prompt"]
        assert "HOLD is not a default" in sent
        assert f"FACTS for {TICKER}" in sent
        assert "1D=Sell" in sent
        assert mock.await_count == 1, "one batched call, not one per ticker"


# ── Failure never renders as HOLD ───────────────────────────────────────────

class TestFailureIsNotHold:

    async def test_omitted_ticker_becomes_no_call(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        _, p_complete, p_cfg = _patched(parsed=[_stance(TICKER, "TRIM")])
        with p_complete, p_cfg:
            batch = await engine.compose([TICKER, "OTHER"])

        actions = {r.ticker: r.action for r in batch.rows}
        assert actions == {TICKER: "TRIM", "OTHER": NO_CALL}
        assert "HOLD" not in actions.values()

        other = next(r for r in batch.rows if r.ticker == "OTHER")
        assert "no stance" in other.thesis.lower()
        # No arrow: a failure is neither an upgrade, a downgrade, nor a held view.
        assert other.arrow == ""

    async def test_no_call_never_prints_hold(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        _, p_complete, p_cfg = _patched(parsed=[])
        with p_complete, p_cfg:
            batch = await engine.compose([TICKER, "OTHER"])

        text = render_stance_message(batch.rows, model_slug=batch.model,
                                     date=batch.date)
        assert "HOLD" not in text
        assert NO_CALL in text

    async def test_unset_model_says_not_configured(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", "")
        mock, p_complete, p_cfg = _patched(parsed=[_stance(TICKER, "HOLD")])
        with p_complete, p_cfg:
            batch = await engine.compose([TICKER])

        assert mock.await_count == 0, "no model means no request"
        assert [r.action for r in batch.rows] == [UNAVAILABLE]

        text = render_stance_message(batch.rows, model_slug=batch.model,
                                     date=batch.date)
        assert "not configured" in text
        assert "HOLD" not in text
        # The fact sheets are still built, so the note has something to show and
        # a later re-run has the inputs it would have used.
        assert batch.facts[TICKER]["price_block"]["price"] == pytest.approx(155.0)

    async def test_llm_exception_degrades_to_no_call(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        mock = AsyncMock(side_effect=RuntimeError("upstream 502"))
        with patch("pipeline.daily_stance.complete", mock), \
             patch("pipeline.daily_stance.is_llm_configured", return_value=True):
            batch = await engine.compose([TICKER])

        assert [r.action for r in batch.rows] == [NO_CALL]

    async def test_unparsed_text_falls_back_to_parse_structured(self, engine,
                                                               monkeypatch):
        """`strict` is off on every schema call, so the text fallback is load-bearing.

        A bare array, which is what `parse_structured(text, list[Stance])`
        validates against — the `{"items": ...}` envelope is the facade's own
        wire detail and is already unwrapped before `.parsed` is set.
        """
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        raw = (
            '[{"ticker": "TEST", "action": "BUY/ADD", '
            '"conviction": "High", "thesis": "Oversold with RSI14 0.0.", '
            '"key_risk": "Falling knife.", "evidence_used": ["rsi"], '
            '"what_would_change_my_mind": "A close below $140."}]'
        )
        _, p_complete, p_cfg = _patched(parsed=None, text=raw)
        with p_complete, p_cfg:
            batch = await engine.compose([TICKER])

        assert [r.action for r in batch.rows] == ["BUY/ADD"]


# ── Arrows come from the record, not the model ──────────────────────────────

class TestArrows:

    async def _compose(self, engine, monkeypatch, action: str):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        _, p_complete, p_cfg = _patched(parsed=[_stance(TICKER, action)])
        with p_complete, p_cfg:
            return await engine.compose([TICKER])

    async def test_first_ever_stance_is_new(self, engine, monkeypatch):
        batch = await self._compose(engine, monkeypatch, "HOLD")
        assert batch.rows[0].arrow == ARROW_NEW
        assert batch.rows[0].prev_action is None

    async def test_upgrade_and_downgrade(self, db, engine, monkeypatch):
        yesterday = (today_et() - dt.timedelta(days=1)).isoformat()
        db.upsert_stance(TICKER, yesterday, "HOLD", conviction="Low")

        batch = await self._compose(engine, monkeypatch, "BUY/ADD")
        assert batch.rows[0].arrow == ARROW_UPGRADED
        assert batch.rows[0].prev_action == "HOLD"

        batch = await self._compose(DailyStanceEngine(db), monkeypatch, "SELL")
        assert batch.rows[0].arrow == ARROW_DOWNGRADED

    async def test_same_action_is_unchanged(self, db, engine, monkeypatch):
        yesterday = (today_et() - dt.timedelta(days=1)).isoformat()
        db.upsert_stance(TICKER, yesterday, "TRIM")
        batch = await self._compose(engine, monkeypatch, "TRIM")
        assert batch.rows[0].arrow == ARROW_UNCHANGED

    async def test_rerun_compares_against_yesterday_not_itself(self, db, engine,
                                                              monkeypatch):
        """A second run the same morning must not report every stance unchanged."""
        yesterday = (today_et() - dt.timedelta(days=1)).isoformat()
        db.upsert_stance(TICKER, yesterday, "HOLD")

        first = await self._compose(engine, monkeypatch, "SELL")
        engine.persist(first)

        second = await self._compose(DailyStanceEngine(db), monkeypatch, "SELL")
        assert second.rows[0].prev_action == "HOLD"
        assert second.rows[0].arrow == ARROW_DOWNGRADED


# ── Persistence ─────────────────────────────────────────────────────────────

class TestPersist:

    async def test_round_trip(self, db, engine, monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        _, p_complete, p_cfg = _patched(parsed=[_stance(TICKER, "SELL")])
        with p_complete, p_cfg:
            batch = await engine.compose([TICKER])

        assert engine.persist(batch) == 1
        stored = db.get_latest_stance(TICKER)
        assert stored["action"] == "SELL"
        assert stored["date"] == today_et().isoformat()
        assert stored["model"] == MODEL
        assert "rsi" in stored["evidence_json"]
        # The fact sheet travels with the call: a stance whose numbers cannot be
        # traced back to its inputs is indistinguishable from an invented one.
        assert "price_block" in stored["facts_json"]

    async def test_rerun_replaces_rather_than_duplicates(self, db, engine,
                                                         monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        for action in ("HOLD", "SELL"):
            _, p_complete, p_cfg = _patched(parsed=[_stance(TICKER, action)])
            with p_complete, p_cfg:
                engine.persist(await DailyStanceEngine(db).compose([TICKER]))

        with db.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) c FROM stances WHERE ticker = ?", (TICKER,)
            ).fetchone()["c"]
        assert count == 1
        assert db.get_latest_stance(TICKER)["action"] == "SELL"

    def test_latest_stances_keyed_by_ticker(self, db, engine):
        today = today_et().isoformat()
        yesterday = (today_et() - dt.timedelta(days=1)).isoformat()
        db.upsert_stance("AAA", yesterday, "HOLD")
        db.upsert_stance("AAA", today, "SELL", evidence_json=["rsi", "news"])
        db.upsert_stance("BBB", yesterday, "BUY/ADD")

        latest = db.get_latest_stances()
        assert set(latest) == {"AAA", "BBB"}
        assert latest["AAA"]["action"] == "SELL"
        # Decoded in the DB layer so the API hands the browser a list, not a
        # JSON string needing a second parse.
        assert latest["AAA"]["evidence_used"] == ["rsi", "news"]
        assert "evidence_json" not in latest["AAA"]


# ── Selective debate re-runs ────────────────────────────────────────────────

class TestMaterialChanges:

    def test_price_move_flags_a_rerun(self, engine):
        facts = {TICKER: engine.build_fact_sheet(TICKER)}
        assert engine.material_changes([_row(TICKER)], facts) == [TICKER]

    def test_quiet_ticker_is_not_flagged(self, engine):
        facts = {"QUIET": {"ticker": "QUIET", "ret_1d": -0.4,
                           "top_news_importance": 2.0}}
        assert engine.material_changes([_row("QUIET")], facts) == []

    def test_flip_flags_a_rerun_without_a_move(self, engine):
        facts = {"FLIP": {"ticker": "FLIP", "ret_1d": 0.2}}
        row = _row("FLIP", "SELL", arrow=ARROW_DOWNGRADED, prev="HOLD")
        assert engine.material_changes([row], facts) == ["FLIP"]

    def test_important_news_flags_a_rerun(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "advisor_rerun_min_importance", 8.0)
        facts = {"NEWS": {"ticker": "NEWS", "ret_1d": 0.1,
                          "top_news_importance": 8.5}}
        assert engine.material_changes([_row("NEWS")], facts) == ["NEWS"]

    def test_debate_already_run_today_is_excluded(self, engine):
        """Otherwise this job and the 08:30 prediction job both pay for it."""
        facts = {"DONE": {"ticker": "DONE", "ret_1d": -9.0,
                          "debate": {"direction": "SELL",
                                     "date": today_et().isoformat()}}}
        assert engine.material_changes([_row("DONE")], facts) == []

    def test_stale_debate_does_not_exclude(self, engine):
        old = (today_et() - dt.timedelta(days=4)).isoformat()
        facts = {"OLD": {"ticker": "OLD", "ret_1d": -9.0,
                         "debate": {"direction": "SELL", "date": old}}}
        assert engine.material_changes([_row("OLD")], facts) == ["OLD"]

    def test_cap_is_respected_and_ordered_flip_first(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "advisor_rerun_max_per_day", 2)
        facts = {
            "MOVE1": {"ret_1d": -4.0},
            "MOVE2": {"ret_1d": -8.0},
            "FLIP": {"ret_1d": -3.2},
            "NEWSY": {"ret_1d": -3.1, "top_news_importance": 9.5},
        }
        rows = [
            _row("MOVE1"), _row("MOVE2"),
            _row("FLIP", "SELL", arrow=ARROW_DOWNGRADED, prev="HOLD"),
            _row("NEWSY"),
        ]
        picked = engine.material_changes(rows, facts)
        assert picked == ["FLIP", "NEWSY"]

    def test_cap_of_zero_disables_reruns(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "advisor_rerun_max_per_day", 0)
        facts = {TICKER: engine.build_fact_sheet(TICKER)}
        assert engine.material_changes([_row(TICKER)], facts) == []


# ── Whole-batch behaviour ───────────────────────────────────────────────────

class TestCompose:

    async def test_empty_watchlist_makes_no_call(self, engine, monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        mock, p_complete, p_cfg = _patched(parsed=[])
        with p_complete, p_cfg:
            batch = await engine.compose([])
        assert batch.rows == [] and mock.await_count == 0

    async def test_ticker_spelling_is_normalised(self, engine, monkeypatch):
        """The grid and the watchlist key on the upper-cased symbol."""
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        _, p_complete, p_cfg = _patched(parsed=[_stance("test", "HOLD")])
        with p_complete, p_cfg:
            batch = await engine.compose(["test"])
        assert batch.rows[0].ticker == TICKER

    async def test_usage_is_logged_under_its_own_operation(self, db, engine,
                                                           monkeypatch):
        monkeypatch.setattr(settings, "model_daily_advisor", MODEL)
        _, p_complete, p_cfg = _patched(parsed=[_stance(TICKER, "SELL")])
        with p_complete, p_cfg:
            await engine.compose([TICKER])

        with db.connection() as conn:
            ops = [r["operation"] for r in
                   conn.execute("SELECT operation FROM llm_usage_log").fetchall()]
        assert "daily_stance_batch" in ops


# ── Schema ──────────────────────────────────────────────────────────────────

class TestStanceSchema:

    def test_prompt_constant_is_shared(self):
        """The rule text lives on the module so tests/prompt audits can see it."""
        assert "HOLD is not a default" in DAILY_STANCE_PROMPT

    def test_rejects_an_action_outside_the_literal(self):
        with pytest.raises(Exception):
            Stance(ticker=TICKER, action="MOON", conviction="High", thesis="t",
                   key_risk="r", what_would_change_my_mind="c")
