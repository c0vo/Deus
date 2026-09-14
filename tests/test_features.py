"""
Tests for pipeline.features — the v4 feature panel.

Everything is synthetic and in memory: PanelInputs is built directly, or a tiny
stand-in object plays the database for the loaders. No SQLite, no network.

The load-bearing test is test_no_lookahead. The panel is only worth training on
if a row for session d is exactly what the builder would have produced on the
evening of d, so appending the future — bars, news, dark-pool rows, a regime
print, an insider filing — must leave every existing row bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import features as F

N_BARS = 600


# ── Synthetic inputs ─────────────────────────────────────────────────────────


def _bars(n: int = N_BARS, *, start: str = "2023-01-02", seed: int = 0,
          price: float = 100.0, vol: float = 0.02) -> pd.DataFrame:
    """A business-day random walk with consistent OHLC and positive volume."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    close = price * np.exp(np.cumsum(rng.normal(0.0003, vol, n)))
    open_ = close * np.exp(rng.normal(0.0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, vol / 2, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=index)


def _market(n: int = N_BARS, start: str = "2023-01-02") -> dict[str, pd.DataFrame]:
    vix = _bars(n, start=start, seed=11, price=18.0, vol=0.05)
    vix["volume"] = 0.0      # indices print no volume
    return {
        "SPY": _bars(n, start=start, seed=10, price=400.0, vol=0.01),
        "^VIX": vix,
        "^TNX": _bars(n, start=start, seed=12, price=4.0, vol=0.02),
        "XLK": _bars(n, start=start, seed=13, price=150.0, vol=0.012),
    }


def _news_rows(days, *, seed: int = 1) -> list[dict]:
    """One or two articles per session, each before that session's cutoff."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in days:
        for _ in range(1 + int(rng.integers(0, 2))):
            rows.append({
                "published_at": f"{day.date()}T{int(rng.integers(0, 23)):02d}:10:00+00:00",
                "sentiment_score": float(rng.uniform(-1, 1)),
                "importance_score": float(rng.uniform(0, 10)),
                "suggested_direction": str(rng.choice(["bullish", "bearish", "neutral"])),
                "urgency": str(rng.choice(["low", "medium", "high", "critical"])),
            })
    return rows


def _offexch_rows(days, *, seed: int = 2) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{
        "session_date": str(day.date()),
        "published_at": f"{day.date()}T22:00:00+00:00",   # 18:00 ET, same evening
        "short_volume": float(rng.uniform(0.3, 0.5)) * 1_000_000,
        "short_exempt_volume": 0.0,
        "total_volume": 1_000_000.0,
        "market_codes": "B,Q,N",
        "consolidated_volume": 2_500_000.0,
    } for day in days]


def _regime_rows(days, *, seed: int = 3) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for metric, level in (("dix", 0.45), ("gex", 5e9), ("occ_put_call_ratio", 0.8)):
        for day in days:
            rows.append({
                "metric": metric,
                "session_date": str(day.date()),
                "value": level * float(rng.uniform(0.9, 1.1)),
                # SqueezeMetrics/OCC rows are stamped 00:00 UTC the next day.
                "published_at": f"{(day + pd.Timedelta(days=1)).date()}T00:00:00+00:00",
                "source": "test",
            })
    return rows


def _insider_rows(days, *, seed: int = 4) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{
        "filed_at": f"{day.date()}T23:59:59+00:00",
        "transaction_date": str((day - pd.Timedelta(days=2)).date()),
        "transaction_code": str(rng.choice(["P", "S"])),
        "is_discretionary": 1,
        "insider_name": str(rng.choice(["Ann", "Bob", "Cy"])),
        "value_usd": float(rng.uniform(1e4, 1e6)),
    } for day in days]


def _inputs(n: int = N_BARS, *, coverage_index: int = 300) -> F.PanelInputs:
    """One US tech ticker with every source populated, plus a bare second ticker."""
    bars = {"AAA": _bars(n, seed=1), "BBB": _bars(n, seed=2, price=50.0, vol=0.03)}
    days = bars["AAA"].index
    return F.PanelInputs(
        bars=bars,
        market=_market(n),
        sector_of={"AAA": "Technology", "BBB": "Unknown"},
        news_daily={"AAA": F.aggregate_news_rows(_news_rows(days[250:]))},
        offexch={"AAA": F.prepare_offexch(_offexch_rows(days[100:]), bars["AAA"])},
        insider={"AAA": F.prepare_insider(_insider_rows(days[150::5]))},
        regime=F.prepare_regime(_regime_rows(days)),
        coverage_start=days[coverage_index],
        market_symbol="SPY",
    )


# ── (a) schema and shapes ────────────────────────────────────────────────────


def test_feature_names_are_the_groups_in_order_and_unique():
    assert F.FEATURE_SCHEMA_VERSION == 4
    assert F.HORIZONS == (5, 21, 63, 252)
    assert list(F.FEATURE_GROUPS) == [
        "price", "market", "regime", "smart_money", "sentiment", "calendar", "context"]
    concatenated = [name for group in F.FEATURE_GROUPS.values() for name in group]
    assert F.FEATURE_NAMES == concatenated
    assert F.FEATURE_NAMES != sorted(F.FEATURE_NAMES)   # order is positional, never sorted
    assert len(set(F.FEATURE_NAMES)) == len(F.FEATURE_NAMES)
    assert set(F.CATEGORICAL_FEATURES) <= set(F.FEATURE_NAMES)


def test_ticker_frame_has_every_feature_and_label_column():
    inputs = _inputs()
    frame = F.build_ticker_frame("AAA", inputs)

    assert list(frame.columns) == F.FEATURE_NAMES + F.label_columns(F.HORIZONS)
    assert frame.index.equals(inputs.bars["AAA"].index)
    assert all(frame[name].dtype == float for name in F.FEATURE_NAMES)
    assert not np.isinf(frame[F.FEATURE_NAMES].to_numpy()).any()
    # A long, fully populated history computes every feature on its last row.
    assert frame[F.FEATURE_NAMES].iloc[-1].notna().all()


def test_panel_stacks_tickers_and_drops_the_warmup_year():
    inputs = _inputs()
    panel = F.build_panel(inputs, ["AAA", "BBB"])

    assert list(panel.columns[:2]) == ["ticker", "date"]
    assert list(panel.columns[2:2 + len(F.FEATURE_NAMES)]) == F.FEATURE_NAMES
    assert set(panel["ticker"]) == {"AAA", "BBB"}
    per_ticker = panel.groupby("ticker").size()
    assert (per_ticker == N_BARS - F.WARMUP_ROWS).all()
    first_kept = inputs.bars["AAA"].index[F.WARMUP_ROWS]
    assert panel.loc[panel["ticker"] == "AAA", "date"].min() == first_kept
    assert panel["date"].is_monotonic_increasing


def test_short_history_keeps_rows_once_the_21_day_window_exists():
    inputs = F.PanelInputs(bars={"CRWV": _bars(100, seed=5)}, market=_market())
    panel = F.build_panel(inputs, ["CRWV"])
    # ret_21d and vol_21 first exist on the 22nd bar.
    assert len(panel) == 100 - 21
    assert panel[["ret_21d", "vol_21"]].notna().all().all()


def test_very_short_history_yields_no_rows_and_does_not_raise():
    inputs = F.PanelInputs(bars={"SKHY": _bars(15, seed=6), "AAA": _bars(seed=1)},
                           market=_market())
    panel = F.build_panel(inputs, ["SKHY", "AAA", "MISSING"])
    assert set(panel["ticker"]) == {"AAA"}

    empty = F.build_panel(F.PanelInputs(bars={"SKHY": _bars(15, seed=6)}), ["SKHY"])
    assert empty.empty
    assert list(empty.columns[:2]) == ["ticker", "date"]


# ── (b) the as-of contract ───────────────────────────────────────────────────


def test_no_lookahead():
    """Appending the future must not change a single feature of the past."""
    extra = 30
    full_bars = {"AAA": _bars(N_BARS + extra, seed=1),
                 "BBB": _bars(N_BARS + extra, seed=2, price=50.0, vol=0.03)}
    full_market = _market(N_BARS + extra)
    days = full_bars["AAA"].index
    last_past = days[N_BARS - 1]
    future_days = days[N_BARS:]

    past_news = _news_rows(days[250:N_BARS])
    future_news = [{"published_at": f"{d.date()}T15:00:00+00:00", "sentiment_score": -1.0,
                    "importance_score": 10.0, "suggested_direction": "bearish",
                    "urgency": "critical"} for d in future_days]
    past_offexch = _offexch_rows(days[100:N_BARS])
    future_offexch = _offexch_rows(future_days, seed=99)
    past_regime = _regime_rows(days[:N_BARS - 1])         # published by d+1 00:00 UTC
    future_regime = [{"metric": "dix", "session_date": str(last_past.date()), "value": 0.99,
                      "published_at": f"{(last_past + pd.Timedelta(days=1)).date()}T00:00:00+00:00"}]
    past_insider = _insider_rows(days[150:N_BARS:5])
    future_insider = _insider_rows(future_days[::3], seed=77)

    def build(bars, market, news, offexch, regime, insider):
        inputs = F.PanelInputs(
            bars=bars, market=market,
            sector_of={"AAA": "Technology", "BBB": "Unknown"},
            news_daily={"AAA": F.aggregate_news_rows(news)},
            offexch={"AAA": F.prepare_offexch(offexch, bars["AAA"])},
            insider={"AAA": F.prepare_insider(insider)},
            regime=F.prepare_regime(regime),
            coverage_start=days[300],
            market_symbol="SPY",
        )
        return {t: F.build_ticker_frame(t, inputs) for t in ("AAA", "BBB")}

    past = build({t: b.iloc[:N_BARS] for t, b in full_bars.items()},
                 {s: b.iloc[:N_BARS] for s, b in full_market.items()},
                 past_news, past_offexch, past_regime, past_insider)
    extended = build(full_bars, full_market, past_news + future_news,
                     past_offexch + future_offexch, past_regime + future_regime,
                     past_insider + future_insider)

    for ticker in ("AAA", "BBB"):
        before = past[ticker][F.FEATURE_NAMES]
        after = extended[ticker].loc[before.index, F.FEATURE_NAMES]
        assert len(before) == N_BARS
        pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_rewriting_future_bars_changes_no_past_feature():
    """Replacing every bar after a cut - tickers and market series, not only appending - leaves
    every earlier row of the ticker frames and of the panel bit-identical."""
    cut = 450
    base, rewritten = _inputs(), _inputs()
    vix = _bars(seed=31, price=25.0, vol=0.08)
    vix["volume"] = 0.0
    other = {"AAA": _bars(seed=21, price=80.0), "BBB": _bars(seed=22, price=30.0, vol=0.05),
             "SPY": _bars(seed=30, price=300.0, vol=0.03), "^VIX": vix,
             "^TNX": _bars(seed=32, price=5.0, vol=0.04), "XLK": _bars(seed=33, price=90.0, vol=0.03)}
    for frames in (rewritten.bars, rewritten.market):
        for key, frame in list(frames.items()):
            frames[key] = pd.concat([frame.iloc[:cut], other[key].iloc[cut:]])
    days = base.bars["AAA"].index

    for ticker in ("AAA", "BBB"):
        before = F.build_ticker_frame(ticker, base)
        after = F.build_ticker_frame(ticker, rewritten)
        assert not before["ret_1d"].iloc[cut:].equals(after["ret_1d"].iloc[cut:])
        pd.testing.assert_frame_equal(before.iloc[:cut][F.FEATURE_NAMES], after.iloc[:cut][F.FEATURE_NAMES],
                                      check_exact=True)

    panel_before, panel_after = F.build_panel(base), F.build_panel(rewritten)
    past_before = panel_before[panel_before["date"] < days[cut]].reset_index(drop=True)
    past_after = panel_after[panel_after["date"] < days[cut]].reset_index(drop=True)
    assert len(past_before) > 0
    pd.testing.assert_frame_equal(past_before[["ticker", "date", *F.FEATURE_NAMES]],
                                  past_after[["ticker", "date", *F.FEATURE_NAMES]], check_exact=True)


def test_regime_value_is_invisible_until_its_publication():
    """A session's DIX, stamped 00:00 UTC the next day, reaches the row one session later."""
    inputs = _inputs()
    days = inputs.bars["AAA"].index
    frame = F.build_ticker_frame("AAA", inputs)

    bumped = inputs.regime.copy()
    target = days[400]
    bumped.loc[(bumped["metric"] == "dix") & (bumped["session_date"] == target), "value"] = 5.0
    changed = F.build_ticker_frame("AAA", F.PanelInputs(**{**inputs.__dict__, "regime": bumped}))

    assert changed.loc[target, "dix_z60"] == frame.loc[target, "dix_z60"]
    assert changed.loc[days[401], "dix_z60"] != frame.loc[days[401], "dix_z60"]


# ── (c) NaN policy ───────────────────────────────────────────────────────────


def test_sentiment_is_nan_before_coverage_and_counts_zero_after():
    inputs = _inputs(coverage_index=300)
    days = inputs.bars["BBB"].index
    frame = F.build_ticker_frame("BBB", inputs)      # BBB has no news at all
    sentiment = F.FEATURE_GROUPS["sentiment"]

    before = frame.loc[frame.index < days[300], sentiment]
    after = frame.loc[frame.index >= days[300]]
    assert before.isna().all().all()
    assert (after["news_n_3d"] == 0).all()
    assert (after["news_n_14d"] == 0).all()
    for mean_type in ("sent_1d", "sent_7d", "news_velocity", "imp_avg_7d", "bull_ratio_7d"):
        assert after[mean_type].isna().all()


def test_no_coverage_start_leaves_every_sentiment_feature_nan():
    inputs = _inputs()
    inputs.coverage_start = None
    frame = F.build_ticker_frame("AAA", inputs)
    assert frame[F.FEATURE_GROUPS["sentiment"]].isna().all().all()


def test_non_us_ticker_has_nan_smart_money_even_with_data():
    inputs = _inputs()
    korean = "005930.KS"
    inputs.bars[korean] = inputs.bars["AAA"]
    inputs.offexch[korean] = inputs.offexch["AAA"]
    inputs.insider[korean] = inputs.insider["AAA"]
    frame = F.build_ticker_frame(korean, inputs)

    assert frame[F.FEATURE_GROUPS["smart_money"]].isna().all().all()
    us = F.build_ticker_frame("AAA", inputs)
    assert us["offexch_short_ratio_z20"].notna().any()


def test_us_ticker_without_alt_data_is_nan_not_zero():
    frame = F.build_ticker_frame("BBB", _inputs())
    assert frame[F.FEATURE_GROUPS["smart_money"]].isna().all().all()


def test_insider_features_start_at_the_first_filing():
    inputs = _inputs()
    days = inputs.bars["AAA"].index
    first_filing = days[150]
    frame = F.build_ticker_frame("AAA", inputs)

    assert frame.loc[frame.index < first_filing, "insider_net_30d"].isna().all()
    visible = frame.loc[frame.index >= first_filing]
    assert visible["insider_net_30d"].notna().all()
    assert visible["insider_cluster_30d"].notna().all()
    assert visible["insider_net_30d"].between(-1.0, 1.0).all()
    assert (visible["days_since_insider_buy"].dropna() <= F.DAYS_SINCE_BUY_CAP).all()


def test_stale_dark_pool_readings_are_dropped():
    inputs = _inputs()
    days = inputs.bars["AAA"].index
    rows = _offexch_rows(days[100:300])                # feed stops at session 299
    inputs.offexch["AAA"] = F.prepare_offexch(rows, inputs.bars["AAA"])
    frame = F.build_ticker_frame("AAA", inputs)

    assert frame.loc[days[299], ["offexch_short_ratio_z20", "offexch_share_z20"]].notna().all()
    stale = days[299] + pd.Timedelta(days=F.OFFEXCH_MAX_STALE_DAYS + 1)
    later = frame.loc[frame.index >= stale]
    assert len(later) > 0
    assert later[["offexch_short_ratio_z20", "offexch_short_ratio_mom",
                  "offexch_share_z20"]].isna().all().all()


# ── (d) bars: splits and cleaning ────────────────────────────────────────────


def _raw_split_history(ratio: float = 4.0, at: int = 300):
    adjusted = _bars(seed=21)
    split_day = adjusted.index[at]
    raw = adjusted.copy()
    before = raw.index < split_day
    raw.loc[before, ["open", "high", "low", "close"]] *= ratio
    raw.loc[before, "volume"] /= ratio
    splits = F.splits_series([{"date": str(split_day.date()), "ratio": ratio}])
    return adjusted, raw, splits, split_day


def test_unadjusted_split_is_repaired():
    adjusted, raw, splits, split_day = _raw_split_history()
    bars = F.prepare_bars(raw, splits, ticker="SPLT")

    log_returns = np.log(bars["close"]).diff().abs()
    assert (log_returns.dropna() <= 0.5).all()
    np.testing.assert_allclose(bars["close"].to_numpy(), adjusted["close"].to_numpy())
    np.testing.assert_allclose(bars["volume"].to_numpy(), adjusted["volume"].to_numpy())

    frame = F.build_ticker_frame("SPLT", F.PanelInputs(bars={"SPLT": bars}, market=_market()))
    step = frame["dist_sma20"].diff().abs()
    assert step.loc[split_day] < 0.1                  # continuous across the split
    assert (frame["ret_1d"].abs().dropna() < 10).all()


def test_already_adjusted_history_is_left_alone():
    """Yahoo's bars arrive split-adjusted; adjusting them again would break them."""
    adjusted, _, splits, _ = _raw_split_history()
    bars = F.prepare_bars(adjusted, splits, ticker="ADJ")
    pd.testing.assert_frame_equal(bars, F.prepare_bars(adjusted, None), check_exact=True)


def test_seam_before_the_split_date_is_repaired():
    """A refresh window that began before the split leaves the seam there, not on the split."""
    adjusted, _, splits, split_day = _raw_split_history(ratio=10.0, at=450)
    patched = adjusted.copy()
    older = patched.index < adjusted.index[200]
    patched.loc[older, ["open", "high", "low", "close"]] *= 10.0
    patched.loc[older, "volume"] /= 10.0

    bars = F.prepare_bars(patched, splits)
    np.testing.assert_allclose(bars["close"].to_numpy(), adjusted["close"].to_numpy())


def test_reverse_split_is_repaired():
    adjusted, raw, splits, _ = _raw_split_history(ratio=0.1)
    bars = F.prepare_bars(raw, splits)
    np.testing.assert_allclose(bars["close"].to_numpy(), adjusted["close"].to_numpy())


def test_panel_drops_rows_with_unrepaired_split_residue():
    _, raw, _, split_day = _raw_split_history()
    inputs = F.PanelInputs(bars={"RAW": F.prepare_bars(raw, None)}, market=_market())
    panel = F.build_panel(inputs, ["RAW"])
    assert split_day not in set(panel["date"])
    assert len(panel) == N_BARS - F.WARMUP_ROWS - 1


def test_clean_daily_bars_keeps_the_trailing_daily_segment():
    monthly = pd.date_range("2015-01-01", "2020-12-01", freq="MS")
    daily = pd.bdate_range("2021-01-04", periods=60)
    weekend = pd.DatetimeIndex(["2021-01-09"])            # a Saturday inside the daily run
    index = monthly.append(daily).append(weekend).sort_values()
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                         index=index)

    cleaned = F.clean_daily_bars(frame, ticker="MIXED")
    assert cleaned.index.equals(daily)


def test_clean_daily_bars_tolerates_a_one_week_market_closure():
    index = pd.bdate_range("2001-08-01", "2001-09-10").append(
        pd.bdate_range("2001-09-17", "2001-10-31"))       # NYSE closed 11-14 Sep 2001
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                         index=index)
    assert len(F.clean_daily_bars(frame)) == len(frame)


def test_market_index_prefers_the_longer_clean_history():
    long_history = _bars(seed=31)
    short_history = long_history.iloc[400:]
    assert F.choose_market_symbol({"SPY": short_history, "^GSPC": long_history}) == "^GSPC"
    assert F.choose_market_symbol({"SPY": long_history, "^GSPC": long_history}) == "SPY"
    stale = long_history.iloc[:-30]                        # stopped updating six weeks ago
    assert F.choose_market_symbol({"SPY": short_history, "^GSPC": stale}) == "SPY"
    assert F.choose_market_symbol({}) is None


def test_market_features_are_nan_without_an_index_series():
    inputs = _inputs()
    inputs.market_symbol = None
    frame = F.build_ticker_frame("AAA", inputs)
    for name in ("rel_spy_21d", "beta_63", "corr_63", "spy_ret_21d"):
        assert frame[name].isna().all()
    assert frame["vix_level"].notna().any()


# ── (e) labels ───────────────────────────────────────────────────────────────


def test_labels_are_the_sign_of_the_forward_log_return():
    inputs = _inputs()
    close = inputs.bars["AAA"]["close"]
    frame = F.build_ticker_frame("AAA", inputs)

    expected = np.log(close.shift(-5) / close)
    np.testing.assert_allclose(frame["fwd_ret_5"].to_numpy(), expected.to_numpy(), equal_nan=True)
    labeled = frame["fwd_ret_5"].notna()
    assert (frame.loc[labeled, "y_5"] == (expected[labeled] > 0).astype(float)).all()
    assert frame["y_5"].iloc[-5:].isna().all()
    assert frame["label_date_5"].iloc[-5:].isna().all()
    assert frame["y_5"].iloc[:-5].notna().all()
    assert frame["label_date_5"].iloc[0] == close.index[5]


# ── (f) the news cutoff ──────────────────────────────────────────────────────


def test_article_after_the_cutoff_counts_on_the_next_session():
    inputs = _inputs()
    days = inputs.bars["BBB"].index
    day = days[420]
    assert day.dayofweek < 4                              # the next calendar day is a session
    rows = [
        {"published_at": f"{day.date()}T23:45:00+00:00", "sentiment_score": -0.9,
         "importance_score": 5.0, "suggested_direction": "bearish", "urgency": "high"},
        {"published_at": f"{day.date()}T23:15:00", "sentiment_score": 0.5,
         "importance_score": 5.0, "suggested_direction": "bullish", "urgency": "low"},
    ]
    daily = F.aggregate_news_rows(rows)
    next_day = day + pd.Timedelta(days=1)
    assert list(daily.index) == [day, next_day]

    inputs.news_daily["BBB"] = daily
    frame = F.build_ticker_frame("BBB", inputs)
    assert frame.loc[day, "sent_1d"] == pytest.approx(0.5)
    assert frame.loc[next_day, "sent_1d"] == pytest.approx(-0.9)
    assert frame.loc[day, "news_n_3d"] == 1
    assert frame.loc[next_day, "news_n_3d"] == 2


# ── (g) matrix helpers ───────────────────────────────────────────────────────


def test_to_matrix_and_categorical_indices():
    panel = F.build_panel(_inputs(), ["AAA", "BBB"])
    matrix = F.to_matrix(panel)
    assert matrix.shape == (len(panel), len(F.FEATURE_NAMES))
    assert matrix.dtype == float
    assert np.isnan(matrix).any()                         # NaN preserved, not imputed

    indices = F.categorical_indices()
    assert [F.FEATURE_NAMES[i] for i in indices] == F.CATEGORICAL_FEATURES
    dow = matrix[:, F.FEATURE_NAMES.index("dow")]
    assert set(np.unique(dow)) <= {0.0, 1.0, 2.0, 3.0, 4.0}

    row = panel.iloc[-1][F.FEATURE_NAMES].astype(float)
    assert F.to_matrix(row).shape == (1, len(F.FEATURE_NAMES))


# ── Loaders and the live row, against a stand-in database ────────────────────


class _FakeDb:
    """Just the loader surface pipeline.features reads."""

    def __init__(self, bars: dict[str, pd.DataFrame], splits: dict | None = None):
        self._bars = bars
        self._splits = splits or {}

    def get_price_history(self, ticker, limit=None):
        frame = self._bars.get(ticker.upper())
        if frame is None:
            return []
        return [{"date": d.strftime("%Y-%m-%d"), **{k: float(v) for k, v in row.items()}}
                for d, row in frame.iterrows()]

    def get_price_splits(self, ticker):
        return self._splits.get(ticker.upper(), [])

    def get_ticker_sector(self, ticker):
        return "Technology"

    def get_ticker_news_rows(self, ticker):
        return []

    def get_offexchange_series(self, ticker):
        return []

    def get_insider_series(self, ticker):
        return []

    def get_market_regime_series(self):
        raise RuntimeError("no such table: market_regime_daily")

    def get_news_coverage_start(self):
        return None


def test_load_panel_inputs_reads_market_series_and_survives_a_failing_source():
    market = _market()
    db = _FakeDb({"AAA": _bars(seed=1), **market})
    inputs = F.load_panel_inputs(db, ["aaa"])

    assert list(inputs.bars) == ["AAA"]
    assert inputs.market_symbol == "SPY"
    assert {"SPY", "^VIX", "^TNX", "XLK"} <= set(inputs.market)
    assert inputs.regime.empty                            # the raising loader cost only itself
    assert inputs.coverage_start is None
    panel = F.build_panel(inputs, ["AAA"])
    assert len(panel) == N_BARS - F.WARMUP_ROWS


def test_build_live_row_returns_the_last_session():
    bars = _bars(seed=1)
    row, meta = F.build_live_row(_FakeDb({"AAA": bars, **_market()}), "AAA")

    assert list(row.index) == F.FEATURE_NAMES
    assert meta["asof_date"] == bars.index[-1].strftime("%Y-%m-%d")
    assert meta["bars"] == N_BARS
    assert meta["stale_days"] >= 0
    frame = F.build_ticker_frame("AAA", F.load_panel_inputs(_FakeDb({"AAA": bars, **_market()}), ["AAA"]))
    pd.testing.assert_series_equal(row, frame[F.FEATURE_NAMES].iloc[-1].rename("AAA"),
                                   check_exact=True)


def test_build_live_row_handles_short_and_missing_histories():
    db = _FakeDb({"NEW": _bars(2, seed=7), "ONE": _bars(1, seed=8), **_market()})
    row, meta = F.build_live_row(db, "NEW")
    assert meta["bars"] == 2
    assert row["ret_252d"] != row["ret_252d"]            # NaN: the window cannot exist yet
    assert F.build_live_row(db, "ONE") is None
    assert F.build_live_row(db, "MISSING") is None
