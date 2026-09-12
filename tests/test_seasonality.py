"""
Tests for pipeline.seasonality — statistics measured on synthetic price history.

Every series here is constructed so the expected statistic is known exactly
rather than eyeballed: the month series steps its level only at month
boundaries, so each monthly return is the number that was asked for, and the
event series is flat except on the event days themselves.

The negative cases matter as much as the positive ones. The module's contract is
that an effect with no computable statistic is omitted, never asserted
qualitatively, so "returns None on a thin sample" and "upcoming_effects drops the
effect" are the tests that protect the feature from turning back into folklore.
"""

import asyncio
import datetime as dt
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.seasonality import (
    Effect,
    MonthStats,
    build_ticker_seasonality,
    ensure_deep_history,
    event_drift_stats,
    month_of_year_stats,
    monthly_returns,
    period_end_stats,
    santa_rally_stats,
    short_week_stats,
    upcoming_effects,
    window_stats,
)

# Septembers with an exactly known distribution: five observations, odd count so
# the median is a member of the sample rather than an average of two.
SEPTEMBER_RETURNS = {2010: -9.0, 2011: -3.0, 2012: -1.0, 2013: 2.0, 2014: 8.0}


def month_end_series(
    first: tuple[int, int] = (2010, 1),
    last: tuple[int, int] = (2015, 1),
    september=None,
) -> list[dict]:
    """
    One bar per month, so each month-end-to-month-end return is exact.

    Every month returns 0% except September, which takes its value from
    `september`. `monthly_returns` only ever looks at the last bar of a month, so
    a single bar per month is a complete series for its purposes.
    """
    september = SEPTEMBER_RETURNS if september is None else september
    rows: list[dict] = []
    level = 100.0
    year, month = first
    while (year, month) <= last:
        if month == 9:
            level *= 1.0 + september.get(year, 0.0) / 100.0
        rows.append({"date": f"{year}-{month:02d}-28", "close": level})
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return rows


def daily_series(
    start: dt.date, end: dt.date, close_for, *, skip=()
) -> list[dict]:
    """Weekday bars between two dates, with the close decided by `close_for`."""
    skip = {dt.date.fromisoformat(d) if isinstance(d, str) else d for d in skip}
    rows: list[dict] = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in skip:
            rows.append({"date": day.isoformat(), "close": close_for(day)})
        day += dt.timedelta(days=1)
    return rows


class TestMonthlyReturns:
    """Reshaping daily closes into calendar-month returns."""

    def test_returns_the_requested_monthly_moves(self):
        returns = monthly_returns(month_end_series())
        assert returns[(2010, 9)] == pytest.approx(-9.0)
        assert returns[(2014, 9)] == pytest.approx(8.0)
        assert returns[(2011, 3)] == pytest.approx(0.0)

    def test_first_month_has_no_predecessor_and_is_excluded(self):
        assert (2010, 1) not in monthly_returns(month_end_series())

    def test_trailing_month_is_never_counted(self):
        # Nothing in the bars proves the last month finished, so it cannot enter
        # a distribution — a September still running is not a September.
        assert (2015, 1) not in monthly_returns(month_end_series())

    def test_gap_in_the_history_is_skipped_not_compounded(self):
        rows = [
            {"date": "2020-01-31", "close": 100.0},
            {"date": "2020-02-28", "close": 110.0},
            # May, not March: the Feb->May move must not be reported as a month.
            {"date": "2020-05-29", "close": 200.0},
            {"date": "2020-06-30", "close": 210.0},
            {"date": "2020-07-31", "close": 220.0},
        ]
        returns = monthly_returns(rows)
        assert (2020, 5) not in returns
        assert returns[(2020, 2)] == pytest.approx(10.0)
        assert returns[(2020, 6)] == pytest.approx(5.0)

    def test_empty_and_single_bar_inputs(self):
        assert monthly_returns([]) == {}
        assert monthly_returns([{"date": "2020-01-31", "close": 100.0}]) == {}

    def test_unparseable_rows_are_dropped(self):
        rows = month_end_series() + [
            {"date": None, "close": 100.0},
            {"date": "not-a-date", "close": 100.0},
            {"date": "2011-09-28", "close": None},
            {"date": "2011-09-28", "close": -5.0},
        ]
        returns = monthly_returns(rows)
        assert returns[(2011, 9)] == pytest.approx(-3.0)


class TestMonthOfYearStats:
    """The September-effect statistic, end to end."""

    def test_exact_distribution(self):
        stats = month_of_year_stats(month_end_series(), 9, min_observations=5)
        assert isinstance(stats, MonthStats)
        assert stats.years == 5
        assert stats.median == pytest.approx(-1.0)
        assert stats.mean == pytest.approx(-0.6)
        assert stats.hit_rate == pytest.approx(0.4)
        assert stats.worst[0] == "2010"
        assert stats.worst[1] == pytest.approx(-9.0)
        assert stats.best[0] == "2014"
        assert stats.best[1] == pytest.approx(8.0)
        assert stats.since == 2010

    def test_thin_sample_returns_none(self):
        stats = month_of_year_stats(month_end_series(), 9, min_observations=6)
        assert stats is None

    def test_two_observations_is_the_hard_floor(self):
        # min_observations=1 cannot get below two: one observation makes "worst"
        # and "best" the same number wearing two hats.
        one = month_end_series(last=(2010, 10))          # September 2010 only
        assert month_of_year_stats(one, 9, min_observations=1) is None

        two = month_end_series(last=(2011, 10))          # September 2010 and 2011
        stats = month_of_year_stats(two, 9, min_observations=1)
        assert stats is not None
        assert stats.years == 2

    def test_month_with_no_data_returns_none(self):
        rows = month_end_series(first=(2010, 1), last=(2010, 6))
        assert month_of_year_stats(rows, 9, min_observations=2) is None


class TestWindowStats:
    """The Sell-in-May halves, including the one that wraps the year."""

    def test_summer_half_compounds_to_the_september_move(self):
        # Every month but September is flat, so May-October is September.
        stats = window_stats(month_end_series(), (5, 6, 7, 8, 9, 10),
                             min_observations=5)
        assert stats.years == 5
        assert stats.median == pytest.approx(-1.0)
        assert stats.worst[1] == pytest.approx(-9.0)

    def test_winter_half_spans_the_year_boundary(self):
        stats = window_stats(month_end_series(), (11, 12, 1, 2, 3, 4),
                             min_observations=4)
        # Nov 2010-Apr 2011 through Nov 2013-Apr 2014; the 2014 season needs an
        # April 2015 the series does not reach.
        assert stats.years == 4
        assert stats.worst[0] == "2010/11"
        assert stats.median == pytest.approx(0.0)
        assert stats.hit_rate == pytest.approx(0.0)

    def test_incomplete_season_is_excluded(self):
        # The data reaches March 2014, so the 2013/14 season is missing its April
        # and must be dropped whole rather than compounded over five months:
        # 2010/11 through 2012/13 are the only complete runs.
        rows = month_end_series(last=(2014, 3))
        stats = window_stats(rows, (11, 12, 1, 2, 3, 4), min_observations=2)
        assert stats.years == 3

    def test_no_months_returns_none(self):
        assert window_stats(month_end_series(), ()) is None


class TestSantaRallyStats:
    """The five-plus-two session window, which needs daily bars."""

    SANTA = {2018: -2.0, 2019: -1.0, 2020: 0.5, 2021: 3.0, 2022: 7.0}

    def series(self) -> list[dict]:
        """
        Flat at 100 except the second January session, which carries the prior
        year's Santa return. The window's base close is the session before it
        opens — still 100 — so the measured return is exactly that number.
        """
        january_sessions: dict[int, list[dt.date]] = {}
        day = dt.date(2018, 1, 1)
        while day <= dt.date(2023, 12, 31):
            if day.weekday() < 5 and day.month == 1:
                january_sessions.setdefault(day.year, []).append(day)
            day += dt.timedelta(days=1)
        bumped = {
            sessions[1]: 100.0 * (1.0 + self.SANTA.get(year - 1, 0.0) / 100.0)
            for year, sessions in january_sessions.items()
            if len(sessions) > 1
        }
        return daily_series(
            dt.date(2018, 1, 1), dt.date(2023, 12, 31),
            lambda d: bumped.get(d, 100.0),
        )

    def test_exact_distribution(self):
        stats = santa_rally_stats(self.series(), min_observations=5)
        assert stats.years == 5
        assert stats.median == pytest.approx(0.5)
        assert stats.hit_rate == pytest.approx(0.6)
        assert stats.worst == ("2018/19", pytest.approx(-2.0))
        assert stats.best == ("2022/23", pytest.approx(7.0))
        assert stats.since == 2018

    def test_final_year_has_no_following_january(self):
        # 2023's window would end in January 2024, which the series never reaches.
        stats = santa_rally_stats(self.series(), min_observations=5)
        assert "2023/24" not in (stats.worst[0], stats.best[0])

    def test_truncated_december_is_not_measured(self):
        # Data ending in June means 2022's "last five sessions of the year" are
        # just the last five sessions we happen to hold — a different statistic,
        # so that season is dropped while 2018/19 through 2021/22 survive.
        rows = daily_series(dt.date(2018, 1, 1), dt.date(2022, 6, 30), lambda d: 100.0)
        stats = santa_rally_stats(rows, min_observations=2)
        assert stats is not None
        assert stats.years == 4

    def test_too_short_series_returns_none(self):
        rows = daily_series(dt.date(2020, 12, 1), dt.date(2020, 12, 4), lambda d: 100.0)
        assert santa_rally_stats(rows) is None


class TestEventDriftStats:
    """Drift into a recurring scheduled event."""

    @staticmethod
    def third_fridays(year: int) -> list[dt.date]:
        out = []
        for month in range(1, 13):
            first = dt.date(year, month, 1)
            offset = (4 - first.weekday()) % 7
            out.append(first + dt.timedelta(days=offset + 14))
        return out

    def series(self, year: int = 2020) -> list[dict]:
        """Flat at 100, except every third Friday prints 110."""
        spikes = set(self.third_fridays(year))
        return daily_series(
            dt.date(year, 1, 1), dt.date(year, 12, 31),
            lambda d: 110.0 if d in spikes else 100.0,
        )

    def test_window_ending_before_the_event_sees_nothing(self):
        stats = event_drift_stats(
            self.series(), self.third_fridays(2020), 1, min_observations=8
        )
        assert stats.years == 12
        assert stats.median == pytest.approx(0.0)
        assert stats.hit_rate == pytest.approx(0.0)

    def test_include_event_day_measures_into_the_print(self):
        stats = event_drift_stats(
            self.series(), self.third_fridays(2020), 1,
            include_event_day=True, min_observations=8,
        )
        assert stats.years == 12
        assert stats.median == pytest.approx(10.0)
        assert stats.hit_rate == pytest.approx(1.0)

    def test_too_few_events_inside_the_data_returns_none(self):
        stats = event_drift_stats(
            self.series(), self.third_fridays(2020)[:7], 1, min_observations=2
        )
        assert stats is None

    def test_dates_outside_the_data_do_not_count_toward_the_floor(self):
        dates = self.third_fridays(2020)[:7] + self.third_fridays(2030)
        assert event_drift_stats(self.series(), dates, 1, min_observations=2) is None

    def test_accepts_date_objects_and_iso_strings(self):
        as_strings = [d.isoformat() for d in self.third_fridays(2020)]
        from_strings = event_drift_stats(self.series(), as_strings, 1,
                                         include_event_day=True, min_observations=8)
        from_dates = event_drift_stats(self.series(), self.third_fridays(2020), 1,
                                       include_event_day=True, min_observations=8)
        assert from_strings == from_dates

    def test_unparseable_event_dates_are_ignored(self):
        dates = [d.isoformat() for d in self.third_fridays(2020)] + ["TBD", None, ""]
        stats = event_drift_stats(
            self.series(), dates, 1, include_event_day=True, min_observations=8
        )
        assert stats.years == 12


class TestShortWeekStats:
    """Holiday-shortened weeks, detected from the bars rather than a calendar."""

    def test_only_short_weeks_are_measured(self):
        holidays = [
            f"{year}-{md}"
            for year in range(2015, 2021)
            for md in ("01-01", "07-03", "11-26", "12-25")
        ]
        rows = daily_series(
            dt.date(2015, 1, 1), dt.date(2020, 12, 31),
            # A step up on every Friday that follows a holiday-shortened week is
            # not needed: a flat series proves the selection, which is the part
            # a calendar-free detector can get wrong.
            lambda d: 100.0,
            skip=holidays,
        )
        stats = short_week_stats(rows, min_observations=5)
        assert stats is not None
        assert stats.years >= 10
        assert stats.median == pytest.approx(0.0)

    def test_series_with_no_holidays_has_no_short_weeks(self):
        rows = daily_series(dt.date(2015, 1, 1), dt.date(2020, 12, 31), lambda d: 100.0)
        assert short_week_stats(rows, min_observations=2) is None

    def test_short_input_returns_none(self):
        rows = daily_series(dt.date(2020, 1, 1), dt.date(2020, 1, 8), lambda d: 100.0)
        assert short_week_stats(rows) is None


class TestPeriodEndStats:
    """The last sessions of a month, and of a quarter."""

    def series(self) -> list[dict]:
        # Level steps at each month boundary, so the last three sessions of a
        # month are flat and the statistic is exactly zero. What is being tested
        # is which months are selected.
        rows = []
        level = 100.0
        day = dt.date(2018, 1, 1)
        month = day.month
        while day <= dt.date(2020, 12, 31):
            if day.month != month:
                month = day.month
                level *= 1.01
            if day.weekday() < 5:
                rows.append({"date": day.isoformat(), "close": level})
            day += dt.timedelta(days=1)
        return rows

    def test_every_complete_month_is_measured(self):
        stats = period_end_stats(self.series(), min_observations=5)
        # 36 months in the series, minus the trailing one that cannot be proven
        # complete.
        assert stats.years == 35

    def test_quarter_months_only(self):
        stats = period_end_stats(self.series(), months=(3, 6, 9, 12),
                                 min_observations=5)
        # Four quarter ends a year for three years, minus December 2020, which is
        # the trailing month.
        assert stats.years == 11

    def test_short_input_returns_none(self):
        rows = [{"date": "2020-01-02", "close": 100.0}]
        assert period_end_stats(rows) is None


class TestBuildTickerSeasonality:
    """The per-ticker assembly used by the composer."""

    def test_deep_series_fills_every_statistic_it_can(self):
        rows = month_end_series(first=(1995, 1), last=(2026, 1))
        stats = build_ticker_seasonality("SPY", rows)
        assert stats.ticker == "SPY"
        assert stats.first_date == "1995-01-28"
        assert stats.months[9] is not None
        assert stats.summer_half is not None
        # A one-bar-per-month series has no sessions, so every session-level
        # statistic has to come back as None rather than as a made-up number:
        # without the contiguity guard "the last three sessions of the month"
        # would silently be measured over three months.
        assert stats.santa is None
        assert stats.short_week is None
        assert stats.month_end is None
        assert stats.quarter_end is None
        assert stats.fomc_run_up is None
        assert stats.opex_week is None
        assert stats.quad_week is None

    def test_thin_series_yields_no_month_statistics(self):
        rows = month_end_series(first=(2024, 1), last=(2025, 6))
        stats = build_ticker_seasonality("NEW", rows)
        assert stats is not None
        assert stats.months == {}
        assert stats.summer_half is None

    def test_empty_input_returns_none(self):
        assert build_ticker_seasonality("X", []) is None
        assert build_ticker_seasonality("X", None) is None

    def test_ticker_is_upper_cased(self):
        stats = build_ticker_seasonality("spy", month_end_series())
        assert stats.ticker == "SPY"


class TestUpcomingEffects:
    """Which effects apply to the coming week, and which are left out."""

    FOMC_ROW = {"date": "2026-09-16", "kind": "fomc", "name": "FOMC rate decision",
                "time_et": "14:00", "importance": 3, "source": "seed"}

    def deep(self):
        return build_ticker_seasonality(
            "SPY", month_end_series(first=(1995, 1), last=(2026, 1))
        )

    def thin(self):
        return build_ticker_seasonality(
            "NEW", month_end_series(first=(2025, 1), last=(2026, 1))
        )

    def test_month_of_year_effect_is_always_offered(self):
        effects = upcoming_effects(dt.date(2026, 9, 13), {"SPY": self.deep()}, [])
        assert any(e.name == "September effect" for e in effects)

    def test_september_line_is_a_finished_sentence_with_its_sample_size(self):
        effects = upcoming_effects(dt.date(2026, 9, 13), {"SPY": self.deep()}, [])
        line = next(e.stat_line for e in effects if e.name == "September effect")
        assert line.startswith("SPY September:")
        assert "median" in line and "up in" in line and "years since" in line
        assert "worst" in line and "best" in line

    def test_effect_without_a_statistic_is_omitted(self):
        effects = upcoming_effects(dt.date(2026, 9, 13), {"NEW": self.thin()}, [])
        assert effects == []

    def test_a_ticker_with_no_statistic_does_not_block_one_that_has_it(self):
        effects = upcoming_effects(
            dt.date(2026, 9, 13), {"SPY": self.deep(), "NEW": self.thin()}, []
        )
        tickers = {e.stat_line.split()[0] for e in effects}
        assert tickers == {"SPY"}

    def test_fomc_effect_needs_both_the_calendar_row_and_the_statistic(self):
        # The month-end series has no sessions, so fomc_run_up is None and the
        # effect must not appear even though the calendar row is there.
        effects = upcoming_effects(
            dt.date(2026, 9, 13), {"SPY": self.deep()}, [self.FOMC_ROW]
        )
        assert not any(e.name == "FOMC decision week" for e in effects)

    def test_fomc_effect_appears_when_the_statistic_exists(self):
        from data.macro_calendar import HISTORICAL_FOMC_DECISIONS

        rows = daily_series(dt.date(2015, 1, 1), dt.date(2026, 1, 31),
                            lambda d: 100.0 + (d - dt.date(2015, 1, 1)).days * 0.01)
        stats = build_ticker_seasonality("SPY", rows)
        assert stats.fomc_run_up is not None
        assert stats.fomc_run_up.years == len(
            [d for d in HISTORICAL_FOMC_DECISIONS if "2015-01-01" <= d <= "2026-01-31"]
        )
        effects = upcoming_effects(dt.date(2026, 9, 13), {"SPY": stats}, [self.FOMC_ROW])
        assert any(e.name == "FOMC decision week" for e in effects)

    def test_fomc_row_outside_the_window_does_not_fire(self):
        row = dict(self.FOMC_ROW, date="2026-12-09")
        effects = upcoming_effects(dt.date(2026, 9, 13), {"SPY": self.deep()}, [row])
        assert not any(e.name == "FOMC decision week" for e in effects)

    def test_quarter_end_window_prefers_quarter_over_month(self):
        rows = daily_series(dt.date(2005, 1, 1), dt.date(2026, 1, 31), lambda d: 100.0)
        stats = build_ticker_seasonality("SPY", rows)
        effects = upcoming_effects(dt.date(2026, 9, 28), {"SPY": stats}, [])
        names = {e.name for e in effects}
        assert "Quarter-end rebalancing" in names
        assert "Month-end window" not in names

    def test_non_quarter_month_end_uses_the_month_window(self):
        rows = daily_series(dt.date(2005, 1, 1), dt.date(2026, 1, 31), lambda d: 100.0)
        stats = build_ticker_seasonality("SPY", rows)
        effects = upcoming_effects(dt.date(2026, 8, 28), {"SPY": stats}, [])
        names = {e.name for e in effects}
        assert "Month-end window" in names
        assert "Quarter-end rebalancing" not in names

    def test_holiday_week_needs_a_calendar_row(self):
        holidays = [f"{y}-{md}" for y in range(2005, 2027)
                    for md in ("01-01", "07-03", "11-26", "12-25")]
        rows = daily_series(dt.date(2005, 1, 1), dt.date(2026, 1, 31),
                            lambda d: 100.0, skip=holidays)
        stats = build_ticker_seasonality("SPY", rows)
        assert stats.short_week is not None

        without = upcoming_effects(dt.date(2026, 9, 13), {"SPY": stats}, [])
        assert not any(e.name == "Holiday-shortened week" for e in without)

        row = {"date": "2026-09-17", "kind": "holiday", "name": "Test closure",
               "importance": 1, "source": "seed"}
        with_row = upcoming_effects(dt.date(2026, 9, 13), {"SPY": stats}, [row])
        assert any(e.name == "Holiday-shortened week" for e in with_row)

    def test_effects_carry_machine_readable_numbers(self):
        effects = upcoming_effects(dt.date(2026, 9, 13), {"SPY": self.deep()}, [])
        effect = next(e for e in effects if e.name == "September effect")
        assert isinstance(effect, Effect)
        assert set(effect.numbers) >= {
            "median_pct", "mean_pct", "hit_rate_pct", "observations", "since",
        }
        assert effect.numbers["observations"] > 5

    def test_empty_stats_map_yields_nothing(self):
        assert upcoming_effects(dt.date(2026, 9, 13), {}, []) == []

    def test_zero_day_window_yields_nothing(self):
        assert upcoming_effects(dt.date(2026, 9, 13), {"SPY": self.deep()}, [],
                                days=0) == []


class TestEnsureDeepHistory:
    """The plumbing half: who gets a full-history pull and who is skipped."""

    @staticmethod
    def run(db, feed, tickers, min_years=10):
        return asyncio.run(ensure_deep_history(db, feed, tickers, min_years))

    def test_only_shallow_tickers_are_refreshed(self):
        db = MagicMock()
        db.get_price_history_starts.return_value = {
            "SPY": "1993-01-29",          # deep enough
            "QQQ": (dt.date.today() - dt.timedelta(days=300)).isoformat(),
        }
        feed = MagicMock()
        feed.refresh_history_for = AsyncMock(return_value=2500)

        result = self.run(db, feed, ["SPY", "QQQ", "NVDA"], min_years=10)

        assert result["checked"] == 3
        # QQQ is too young and NVDA has no bars at all.
        assert result["needed"] == 2
        assert result["refreshed"] == 2
        assert result["rows"] == 5000
        pulled = [c.args[0][0] for c in feed.refresh_history_for.call_args_list]
        assert pulled == ["QQQ", "NVDA"]
        assert all(
            c.kwargs["history_range"] == "max"
            for c in feed.refresh_history_for.call_args_list
        )

    def test_nothing_to_do_makes_no_network_call(self):
        db = MagicMock()
        db.get_price_history_starts.return_value = {"SPY": "1993-01-29"}
        feed = MagicMock()
        feed.refresh_history_for = AsyncMock(return_value=0)

        result = self.run(db, feed, ["SPY"])

        assert result == {"checked": 1, "needed": 0, "refreshed": 0,
                          "rows": 0, "failed": 0}
        feed.refresh_history_for.assert_not_called()

    def test_a_failed_pull_is_counted_not_raised(self):
        db = MagicMock()
        db.get_price_history_starts.return_value = {}
        feed = MagicMock()
        feed.refresh_history_for = AsyncMock(side_effect=RuntimeError("429"))

        result = self.run(db, feed, ["AAA"])

        assert result["failed"] == 1
        assert result["refreshed"] == 0

    def test_zero_rows_stored_counts_as_a_failure(self):
        db = MagicMock()
        db.get_price_history_starts.return_value = {}
        feed = MagicMock()
        feed.refresh_history_for = AsyncMock(return_value=0)

        result = self.run(db, feed, ["AAA"])

        assert result["failed"] == 1

    def test_tickers_are_normalised_and_deduplicated(self):
        db = MagicMock()
        db.get_price_history_starts.return_value = {"SPY": "1993-01-29"}
        feed = MagicMock()
        feed.refresh_history_for = AsyncMock(return_value=1)

        result = self.run(db, feed, [" spy ", "SPY", "", None])

        assert result["checked"] == 1
        db.get_price_history_starts.assert_called_once_with(["SPY"])

    def test_a_failing_depth_query_degrades_quietly(self):
        db = MagicMock()
        db.get_price_history_starts.side_effect = RuntimeError("locked")
        feed = MagicMock()
        feed.refresh_history_for = AsyncMock(return_value=1)

        result = self.run(db, feed, ["SPY"])

        assert result["refreshed"] == 0
        feed.refresh_history_for.assert_not_called()

    def test_empty_ticker_list_is_a_no_op(self):
        db = MagicMock()
        feed = MagicMock()
        result = self.run(db, feed, [])
        assert result["checked"] == 0
        db.get_price_history_starts.assert_not_called()
