"""Tests for the free flow-data collectors: FINRA off-exchange, DIX/GEX, OCC, options.

Every test here runs against a saved fixture or a synthetic frame — no network,
no API keys. The fixtures are verbatim slices of real responses, because every
assertion below encodes a failure mode that only shows up against real data:
the trailing record count in FINRA's file, fractional share volumes, OCC's
space-padded exchange names, and Yahoo's unusable deep-ITM implied vols.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from pipeline.darkpool import DarkPoolTracker, FinraShortVolumeClient
from pipeline.market_regime import (
    METRIC_DIX,
    METRIC_GEX,
    METRIC_PUT_CALL,
    OccVolumeProvider,
    SqueezeMetricsProvider,
)
from pipeline.options_flow import OptionsSnapshotTracker, _dte, _num

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def finra_text():
    return (FIXTURES / "finra_cnms_sample.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def finra_rows(finra_text):
    return FinraShortVolumeClient.parse(finra_text, expected_date=date(2026, 8, 7))


# ── FINRA off-exchange volume ───────────────────────────────────────────────

def test_parse_drops_header_and_trailing_count(finra_rows, finra_text):
    """The file is header + data + a bare record count on the last line.

    The trailer splits into a single field, so any positional read of parts[1]
    raises IndexError on every file ever published. The header survives the
    field-count check (it has all six) and has to be dropped on its own.
    """
    assert len(finra_text.strip().splitlines()) == 14  # header + 12 rows + count
    assert len(finra_rows) == 12

    symbols = {r["ticker"] for r in finra_rows}
    assert "SYMBOL" not in symbols and "DATE" not in symbols
    assert "12" not in symbols


def test_volumes_parse_as_floats(finra_rows):
    """FINRA reports fractional share-adjusted volumes; int() would throw."""
    aapl = next(r for r in finra_rows if r["ticker"] == "AAPL")
    assert aapl["short_volume"] == pytest.approx(5540409.463985)
    assert aapl["total_volume"] == pytest.approx(13330297.818991)
    assert all(isinstance(r["total_volume"], float) for r in finra_rows)


def test_short_exempt_mixes_ints_and_floats(finra_rows):
    """Within one file this column is '31490.250000' on one row and '21' on the next."""
    aapl = next(r for r in finra_rows if r["ticker"] == "AAPL")
    a = next(r for r in finra_rows if r["ticker"] == "A")
    assert aapl["short_exempt_volume"] == pytest.approx(31490.25)
    assert a["short_exempt_volume"] == pytest.approx(21.0)
    assert all(isinstance(r["short_exempt_volume"], float) for r in finra_rows)


def test_offexchange_share_is_plausible(finra_rows):
    """Sanity anchor: this column is off-exchange volume, not consolidated volume.

    If a future change ever points these fields at the wrong column, the share
    of consolidated volume is what visibly breaks — measured off-exchange share
    runs 33-47% for large caps, and AAPL printed 13.3M against a 34.4M tape.
    """
    aapl = next(r for r in finra_rows if r["ticker"] == "AAPL")
    assert aapl["total_volume"] / 34_407_100 == pytest.approx(0.387, abs=0.01)
    for row in finra_rows:
        assert 0 < row["short_volume"] <= row["total_volume"]


@pytest.mark.parametrize("session,expected_utc_hour,label", [
    (date(2026, 8, 7), 22, "EDT — 18:00 ET is 22:00 UTC"),
    (date(2026, 1, 15), 23, "EST — 18:00 ET is 23:00 UTC"),
])
def test_published_at_follows_dst(session, expected_utc_hour, label):
    """The as-of key must track DST, not a fixed offset.

    The predictor cuts visibility at 23:59:59 UTC on the as-of date. A hardcoded
    -4h offset would place winter sessions at 22:00 UTC — still inside the same
    day, so nothing would fail loudly, it would just show the model data one day
    before it existed on roughly half the calendar.
    """
    stamp = FinraShortVolumeClient.published_at(session)
    parsed = datetime.fromisoformat(stamp)
    assert parsed.astimezone(timezone.utc).hour == expected_utc_hour, label
    assert parsed.astimezone(timezone.utc).date() == session


def test_parse_tolerates_garbage_lines():
    """Blank lines, short lines and unparseable numerics are skipped, not raised."""
    text = "\n".join([
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
        "20260807|AAPL|5540409.463985|31490.250000|13330297.818991|B,Q,N",
        "",
        "20260807|BAD|not_a_number|0|123.0|B",
        "20260807||1.0|0|2.0|B",       # missing symbol
        "short|line",
        "12180",
    ])
    rows = FinraShortVolumeClient.parse(text)
    assert [r["ticker"] for r in rows] == ["AAPL"]


@pytest.mark.asyncio
async def test_403_returns_none_and_is_not_retried():
    """403 is how this CDN spells 'no file' — weekend, holiday, or too old.

    Retrying it would mean three requests for every weekend of a multi-year
    backfill, and would read a routine absence as a transient failure.
    """
    class _Response:
        status_code = 403
        text = ""

    class _Client:
        def __init__(self):
            self.calls = 0

        async def get(self, url, **kwargs):
            self.calls += 1
            return _Response()

    client = _Client()
    result = await FinraShortVolumeClient().fetch_day(date(2026, 8, 8), client)
    assert result is None
    assert client.calls == 1


@pytest.mark.asyncio
async def test_sync_range_filters_to_tracked_tickers(monkeypatch, finra_rows):
    """One file carries ~12,000 symbols; only the watchlist is stored."""
    stored = []

    class _DB:
        def upsert_offexchange_volume(self, rows):
            stored.extend(rows)
            return len(rows)

    async def _fake_fetch(self, session_date, client=None):
        return finra_rows

    monkeypatch.setattr(FinraShortVolumeClient, "fetch_day", _fake_fetch)
    tracker = DarkPoolTracker(db=_DB())
    totals = await tracker.sync_range(
        ["AAPL", "NVDA", "005930.KS"], date(2026, 8, 7), date(2026, 8, 7)
    )

    assert totals["sessions"] == 1
    assert {r["ticker"] for r in stored} == {"AAPL", "NVDA"}


# ── SqueezeMetrics DIX / GEX ────────────────────────────────────────────────

def test_dix_parser_emits_two_metrics_per_session():
    text = (FIXTURES / "dix_sample.csv").read_text(encoding="utf-8")
    rows = SqueezeMetricsProvider.parse(text)

    assert len(rows) == 14  # 7 sessions x {dix, gex}
    assert {r["metric"] for r in rows} == {METRIC_DIX, METRIC_GEX}
    assert all(isinstance(r["value"], float) for r in rows)

    latest_dix = [r for r in rows if r["metric"] == METRIC_DIX][-1]
    latest_gex = [r for r in rows if r["metric"] == METRIC_GEX][-1]
    assert latest_dix["value"] == pytest.approx(0.4553315957553551)
    assert latest_gex["value"] == pytest.approx(9051568501.380009)


def test_dix_parser_skips_header_without_counting_it_malformed():
    """The header is expected, not corrupt — it must not raise or log as damage."""
    rows = SqueezeMetricsProvider.parse("date,price,dix,gex\n2026-08-07,1.0,0.5,2.0\n")
    assert len(rows) == 2
    assert rows[0]["session_date"] == "2026-08-07"


def test_regime_published_at_lags_one_day():
    """Both feeds derive from the US close but publish no documented timestamp,
    so they are stamped as public the next day rather than guessed at."""
    rows = SqueezeMetricsProvider.parse("date,price,dix,gex\n2026-08-07,1.0,0.5,2.0\n")
    assert rows[0]["published_at"].startswith("2026-08-08T00:00")


# ── OCC put/call ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def occ_payload():
    return json.loads((FIXTURES / "occ_daily_volume.json").read_text(encoding="utf-8"))


def test_occ_ratio_is_volume_weighted_not_a_mean_of_ratios(occ_payload):
    """Per-venue ratios span 0.56-0.80 across 2%-18% market shares.

    An unweighted mean of them is not the market's put/call ratio and is not
    any published figure either. This fixture's two answers differ by 0.0045 —
    small, but the error is systematic and grows with venue dispersion.
    """
    row = OccVolumeProvider.parse(occ_payload, date(2026, 8, 7))
    venues = occ_payload["entity"]["total_volume"]
    naive_mean = sum(v["ratio"] for v in venues) / len(venues)

    assert row["value"] == pytest.approx(0.6592304064)
    assert row["value"] != pytest.approx(naive_mean, abs=1e-4)
    assert row["metric"] == METRIC_PUT_CALL


def test_occ_covers_every_venue(occ_payload):
    """This is the whole point of using OCC over Cboe, which is one ~18% venue."""
    row = OccVolumeProvider.parse(occ_payload, date(2026, 8, 7))
    assert row["exchanges"] == len(occ_payload["entity"]["total_volume"]) == 19


def test_occ_exchange_names_are_space_padded(occ_payload):
    """Guards the fixture itself: OCC returns 'AMEX ' and 'C2   ', so anything
    keyed off these names needs stripping."""
    names = [v["exchange"] for v in occ_payload["entity"]["total_volume"]]
    assert any(n != n.strip() for n in names)


@pytest.mark.parametrize("payload", [
    {},
    {"entity": {}},
    {"entity": {"total_volume": []}},
    {"entity": {"total_volume": [{"exchange": "X", "calls": 0, "puts": 0}]}},
])
def test_occ_returns_none_on_empty_payloads(payload):
    assert OccVolumeProvider.parse(payload, date(2026, 8, 7)) is None


# ── Option chain aggregation ────────────────────────────────────────────────

def _contract(strike, iv, *, itm=False, bid=1.0, volume=10, oi=100):
    return {
        "strike": strike, "impliedVolatility": iv, "inTheMoney": itm,
        "bid": bid, "ask": bid + 0.5, "volume": volume, "openInterest": oi,
    }


def test_deep_itm_iv_is_excluded():
    """Yahoo quotes 133% IV on a deep-ITM AAPL 260 call against a 313 spot.

    One such row in the average is enough to invert the sign of any skew
    measure, so ITM contracts never contribute to an IV figure.
    """
    calls = [_contract(260.0, 1.333988, itm=True), _contract(320.0, 0.28)]
    row = OptionsSnapshotTracker.aggregate("AAPL", 313.33, [(30, calls, [])])

    assert row["atm_iv"] == pytest.approx(0.28)


@pytest.mark.parametrize("contract,reason", [
    (_contract(320.0, 0.28, bid=0.0), "no bid means no real price"),
    (_contract(320.0, 0.28, volume=0, oi=0), "never traded and nobody holds it"),
    (_contract(999.0, 0.28), "far outside the moneyness band"),
    (_contract(320.0, 0.0), "zero IV is a placeholder, not a quote"),
    (_contract(320.0, 9.9), "900% IV is a broken back-solve"),
])
def test_untrustworthy_quotes_are_excluded_from_iv(contract, reason):
    row = OptionsSnapshotTracker.aggregate("AAPL", 313.33, [(30, [contract], [])])
    assert row["atm_iv"] is None, reason


def test_nan_volume_does_not_poison_totals():
    """yfinance returns pandas NaN — not None — for untraded contracts.

    NaN is truthy, so `row.get("volume") or 0.0` keeps it, one NaN makes the
    running total NaN, and every volume figure for the ticker silently lands as
    NaN with the ratios falling back to None.
    """
    nan = float("nan")
    calls = [_contract(320.0, 0.28, volume=nan), _contract(330.0, 0.29, volume=50)]
    puts = [_contract(300.0, 0.31, volume=20)]
    row = OptionsSnapshotTracker.aggregate("AAPL", 313.33, [(30, calls, puts)])

    assert row["call_volume"] == 50
    assert row["put_volume"] == 20
    assert row["put_call_volume_ratio"] == pytest.approx(0.4)


def test_num_is_nan_safe():
    assert _num(float("nan")) == 0.0
    assert _num(None) == 0.0
    assert _num("garbage") == 0.0
    assert _num("12.5") == 12.5


def test_iv_skew_uses_the_reference_expiry_not_the_front_weekly():
    """Skew legs sit 10%+ out of the money, where a 1-day weekly has no bid.

    Measuring on the front expiry returns None on most sessions; the ~30 DTE
    leg is both the convention and the one that actually has a market.
    """
    front_calls = [_contract(320.0, 0.60)]           # weekly, no wings quoted
    ref_puts = [_contract(280.0, 0.35)]              # ~10% OTM put
    ref_calls = [_contract(350.0, 0.25)]             # ~12% OTM call

    row = OptionsSnapshotTracker.aggregate(
        "AAPL", 313.33, [(1, front_calls, []), (30, ref_calls, ref_puts)]
    )

    assert row["iv_skew"] == pytest.approx(0.10)     # 0.35 put - 0.25 call
    assert row["near_term_iv"] == pytest.approx(0.60)
    assert row["atm_iv"] == pytest.approx(0.30)      # mean of the 30-DTE leg


def test_term_structure_uses_shortest_and_longest():
    chains = [(7, [_contract(320.0, 0.40)], []), (90, [_contract(320.0, 0.25)], [])]
    row = OptionsSnapshotTracker.aggregate("AAPL", 313.33, chains)

    assert row["near_term_iv"] == pytest.approx(0.40)
    assert row["far_term_iv"] == pytest.approx(0.25)
    assert row["expirations_seen"] == 2


def test_select_expirations_targets_dte_not_position():
    """Liquid names list weeklies, so "the first four" spans under two weeks.

    Targeting DTE is what makes near_term_iv and far_term_iv measure different
    things instead of the same thing twice.
    """
    listed = ["2026-08-10", "2026-08-12", "2026-08-14", "2026-08-21",
              "2026-09-11", "2026-10-16", "2026-11-20", "2027-01-15"]
    chosen = OptionsSnapshotTracker.select_expirations(
        listed, targets=(7, 30, 60, 90)
    )

    dtes = [_dte(e) for e in chosen]
    assert len(chosen) == len(set(chosen)), "no expiry fetched twice"
    assert max(dtes) - min(dtes) > 60, "spans a real term structure"


def test_select_expirations_dedupes_thin_chains():
    """A name listing one expiry yields one request, not four identical ones."""
    chosen = OptionsSnapshotTracker.select_expirations(["2026-09-18"])
    assert chosen == ["2026-09-18"]


def test_session_date_is_the_us_trading_day():
    """The job runs at 06:00 KST = 21:00 UTC the previous day, so a UTC date
    would mislabel roughly half the year's snapshots."""
    row = OptionsSnapshotTracker.aggregate("AAPL", 313.33, [(30, [], [])])
    expected = datetime.now(timezone.utc).astimezone(
        ZoneInfo("America/New_York")).date().isoformat()
    assert row["session_date"] == expected
