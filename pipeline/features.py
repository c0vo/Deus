"""
Feature panel for the pooled direction model — feature schema v4.

One code path builds both the training panel and the live row.
`build_ticker_frame` turns one ticker's stored history into a frame with one
row per session, and `build_live_row` is the last row of that same frame, so a
feature cannot be computed one way in training and another way live. Every
feature is a vectorized pandas operation over the whole history (rolling
windows, EWMs, merge_asof): a ticker costs one pass, where the v3 predictor
rebuilt the whole vector once per historical date.

As-of contract
--------------
The row for session date d uses only what was knowable when the daily
prediction job runs, 23:30 UTC (08:30 KST the next morning):

  bars           date <= d
  news           articles.published_at <= d 23:30 UTC, joined through
                 ticker_mentions. Never mentioned_at, which is when the
                 classifier reached the article, not when anyone could read it.
  dark pool      offexchange_volume.published_at <= d 23:30 UTC
  market regime  market_regime_daily.published_at <= d 23:30 UTC. DIX/GEX are
                 stamped 00:00 UTC the day after their session, so they reach
                 the features one session late by construction.
  insider        insider_transactions.filed_at <= d, end of day: the
                 disclosure date, never the trade date.
  sector         today's classification. A known look-ahead, accepted:
                 point-in-time sector history is not stored and sectors are
                 rarely re-labelled.
  splits         calendar facts (see "Bars" below).

tests/test_features.py::test_no_lookahead enforces this: appending future bars,
news, dark-pool rows and a regime row leaves every existing row's features
bit-identical. The only forward-looking columns are the labels —
fwd_ret_{h} = log(C[d+h] / C[d]) over the ticker's own session index,
y_{h} = 1 if that is positive, label_date_{h} the session it resolves on — and
rows without a d+h session are unlabeled (the live row always is).

NaN policy
----------
Nothing is imputed. A feature that cannot be computed — a window not yet full,
a source with nothing visible on that date, a source gone stale — is NaN, and
the gradient-boosted model learns what missing means. Zero is reserved for a
measured zero.

The sentiment group is the case where that matters most. It is NaN on every row
dated before `coverage_start`, the first day classified news was collected
densely. Prices go back decades and classified news a few months, so a zero on
those rows would teach the model that "nobody was collecting articles" means
"nothing happened to this company". From coverage_start on, counts are 0 on a
day with no news and the mean-type features are NaN, which keeps "no coverage"
and "no news" distinguishable.

Market series
-------------
spy_* / rel_spy_* / beta_63 / corr_63 read whichever of SPY and ^GSPC has the
earlier clean daily history (a candidate whose bars stopped updating is
skipped); the choice is recorded as PanelInputs.market_symbol. SPY's stored
history was front-loaded with monthly bars by Yahoo's range=max downgrade while
^GSPC's is clean, so the index series can differ between databases. The two
track each other to within the dividend, ~0.005% a day, which no feature here
resolves. ^VIX and ^TNX come from the same price_history table.

Bars
----
Stored bars are cleaned first (`clean_daily_bars`): weekend-dated rows are
dropped and only the trailing run of true daily bars is kept, because monthly
bars from the range=max downgrade sit in front of the daily ones for some
symbols.

Splits are handled by repair, not blanket adjustment. Yahoo serves bars already
split-adjusted as of the day they are fetched, and price_history is INSERT OR
REPLACE over trailing windows, so after a split the table holds re-fetched rows
at the new scale next to older rows still at the old one — and that seam falls
where a refresh window began, not on the split date. Dividing everything before
a split by its ratio would corrupt every history fetched after the split, which
after a full backfill is all of them. `adjust_for_splits` instead looks for a
one-session jump matching a stored split ratio (or a run of consecutive ones)
dated before that split, and rescales only the rows before a seam it finds; a
history with no seam comes back untouched. As a last line of defence
`build_panel` drops, and logs, any row whose raw one-day log return exceeds 0.5.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from data.tickers import US, classify_market
from data.watchlist import ETF_TICKERS, SECTOR_ETF_MAP, SECTOR_ETFS, sector_etf_for
from pipeline.technical_rating import TechnicalRatingTracker, _ema, _rma

log = get_logger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────────

# Stamped into every model artifact's filename by the predictor. Bump it with
# any change to FEATURE_NAMES or to a feature's definition.
FEATURE_SCHEMA_VERSION = 4

# Prediction horizons in sessions, matching the predictor's HORIZON_LABELS.
HORIZONS: tuple[int, ...] = (5, 21, 63, 252)

# When run_daily_predictions fires (08:30 KST). Anything published by then is
# visible to that session's row.
NEWS_CUTOFF_UTC = time(23, 30)

CATEGORICAL_FEATURES: list[str] = ["sector_id", "dow"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "price": [
        "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
        "mom_12_1",
        "dist_sma20", "dist_sma50", "dist_sma200", "sma50_200",
        "rsi_14", "macd_hist_pct", "bb_pos", "bb_width", "atr_pct",
        "vol_21", "vol_63", "vol_ratio",
        "dd_63", "dist_52w_high", "dist_52w_low",
        "volume_z20", "log_dollar_vol",
        "gap_1d", "intraday_1d", "up_days_10", "skew_63",
    ],
    "market": [
        "rel_spy_5d", "rel_spy_21d", "rel_spy_63d",
        "beta_63", "corr_63",
        "rel_sector_21d",
        "spy_ret_5d", "spy_ret_21d", "spy_ret_63d", "spy_dist_sma200", "spy_vol_21",
        "vix_level", "vix_z63", "vix_chg_5d",
        "tnx_chg_5d", "tnx_chg_21d",
    ],
    "regime": ["dix_z60", "gex_z60", "pcr_z60"],
    "smart_money": [
        "offexch_short_ratio_z20", "offexch_short_ratio_mom", "offexch_share_z20",
        "insider_buy_ratio_90d", "insider_net_30d", "insider_cluster_30d",
        "days_since_insider_buy",
    ],
    "sentiment": [
        "sent_1d", "sent_3d", "sent_7d", "sent_mom",
        "news_n_3d", "news_n_14d", "news_velocity",
        "imp_avg_7d", "imp_max_7d", "bull_ratio_7d", "urg_max_7d",
    ],
    "calendar": ["dow", "days_to_month_end"],
    "context": ["sector_id", "is_etf", "hist_len_years"],
}

# The model's column order: the groups concatenated in the order above. Never
# sorted() — an artifact stores this list and is scored against it by position.
FEATURE_NAMES: list[str] = [name for group in FEATURE_GROUPS.values() for name in group]

# ── Tunables ─────────────────────────────────────────────────────────────────

# SPY first: on equal history it wins, being the series the features are named for.
MARKET_INDEX_CANDIDATES: tuple[str, ...] = ("SPY", "^GSPC")
VIX_SYMBOL = "^VIX"
TNX_SYMBOL = "^TNX"
# A market candidate whose last bar trails the other's by more than this has
# stopped updating, and would blank the live row's market features.
MARKET_MAX_STALE_DAYS = 10

# Longest calendar gap between consecutive daily bars. The 9/11 closure is
# exactly 7 (Mon 10 Sep -> Mon 17 Sep 2001); a monthly bar is ~30.
MAX_SESSION_GAP_DAYS = 7

# build_panel drops up to a year of warm-up rows per ticker, but always keeps
# the most recent year (or everything once ret_21d/vol_21 exist, if shorter).
WARMUP_ROWS = 252
MIN_LIVE_BARS = 2
# A raw one-day |log return| above this is unrepaired split residue, not a day.
SPLIT_RESIDUE_LOG_RETURN = 0.5

# Split seam matching, in log space: a seam's jump must land within this of
# -log(ratio). A ratio closer to 1 than SPLIT_MIN_LOG_RATIO (about 1.16:1) is
# indistinguishable from an ordinary session and is never matched on its own.
SPLIT_MATCH_TOLERANCE = 0.10
SPLIT_MIN_LOG_RATIO = 0.15

MACD_WARMUP_BARS = 35            # EMA26 + signal EMA9: the seed's influence has decayed

REGIME_METRICS: dict[str, str] = {
    "dix": "dix_z60",
    "gex": "gex_z60",
    "occ_put_call_ratio": "pcr_z60",
}
REGIME_Z_WINDOW = 60
REGIME_MAX_STALE_DAYS = 10

OFFEXCH_Z_WINDOW = 20
OFFEXCH_MOM_WINDOW = 5
OFFEXCH_MAX_STALE_DAYS = 7

INSIDER_RATIO_WINDOW = 90
INSIDER_NET_WINDOW = 30
INSIDER_CLUSTER_WINDOW = 30
DAYS_SINCE_BUY_CAP = 365.0

URGENCY_CODES: dict[str, float] = {"low": 0.0, "medium": 1.0, "high": 2.0, "critical": 3.0}

# Per-day news aggregates. Sums and counts rather than means, so a multi-day
# window re-averages over articles instead of averaging daily averages.
NEWS_DAILY_COLUMNS: list[str] = [
    "n", "sent_sum", "sent_cnt", "imp_sum", "imp_cnt", "imp_max", "bull", "bear", "urg_max",
]
_NEWS_MAX_COLUMNS = ("imp_max", "urg_max")

# One integer per sector, keyed by its tracking ETF so the aliases in
# SECTOR_ETF_MAP ("finance", "financial services") share a code. Ordered by
# first appearance in the map, which makes the codes stable across runs.
SECTOR_CODES: dict[str, int] = {
    etf: code for code, etf in enumerate(dict.fromkeys(SECTOR_ETF_MAP.values()))
}

# Funds have no yfinance sector. A sector ETF belongs to its own sector; the
# broad index funds to "index".
_FUND_SECTORS: dict[str, str] = {
    **{fund: "index" for fund in ETF_TICKERS},
    **{etf: key for key, etf in reversed(list(SECTOR_ETF_MAP.items())) if etf in SECTOR_ETFS},
}

_BAR_COLUMNS = ("open", "high", "low", "close", "volume")
_CUTOFF_OFFSET = pd.Timedelta(hours=NEWS_CUTOFF_UTC.hour, minutes=NEWS_CUTOFF_UTC.minute)
_NS = "datetime64[ns]"
_NS_UTC = "datetime64[ns, UTC]"

_warned: set[tuple] = set()


# ── Inputs ───────────────────────────────────────────────────────────────────

def _empty_regime() -> pd.DataFrame:
    return pd.DataFrame({
        "metric": pd.Series(dtype=object),
        "session_date": pd.Series(dtype=_NS),
        "available_at": pd.Series(dtype=_NS_UTC),
        "value": pd.Series(dtype=float),
    })


@dataclass
class PanelInputs:
    """Everything the feature builder reads, loaded from SQLite once.

    Every frame is already in the shape the builder consumes, so tests build
    this directly instead of going through a database:

      bars, market    DatetimeIndex of naive session dates (sorted, unique) with
                      float open/high/low/close/volume, cleaned and split-repaired
                      (`prepare_bars`). `market` holds SPY, ^GSPC, ^VIX, ^TNX and
                      the sector ETFs the tickers need.
      sector_of       ticker -> sector name as stored (normalized on use).
      news_daily      ticker -> `aggregate_news_rows` output: index news_date,
                      columns NEWS_DAILY_COLUMNS.
      offexch         ticker -> `prepare_offexch` output: session_date,
                      available_at (UTC), short_ratio, share.
      insider         ticker -> `prepare_insider` output: filed_date,
                      transaction_date, code, discretionary, value, name.
      regime          `prepare_regime` output, long form: metric, session_date,
                      available_at (UTC), value.
      splits          ticker -> ratios indexed by effective session (`splits_series`).
      coverage_start  first day of dense news coverage; None leaves every
                      sentiment feature NaN.
      market_symbol   the `market` key the spy_* features read, or None.
    """

    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    market: dict[str, pd.DataFrame] = field(default_factory=dict)
    sector_of: dict[str, str] = field(default_factory=dict)
    news_daily: dict[str, pd.DataFrame] = field(default_factory=dict)
    offexch: dict[str, pd.DataFrame] = field(default_factory=dict)
    insider: dict[str, pd.DataFrame] = field(default_factory=dict)
    regime: pd.DataFrame = field(default_factory=_empty_regime)
    splits: dict[str, pd.Series] = field(default_factory=dict)
    coverage_start: Optional[pd.Timestamp] = None
    market_symbol: Optional[str] = MARKET_INDEX_CANDIDATES[0]


# ── Small helpers ────────────────────────────────────────────────────────────

def _warn_once(event: str, **fields: Any) -> None:
    """Log a warning the first time this exact event and payload is seen.

    Loaders run once per ticker per training run and once per live prediction;
    a missing table or a polluted SPY history would otherwise repeat every time.
    """
    key = (event, tuple(sorted((k, str(v)) for k, v in fields.items())))
    if key in _warned:
        return
    _warned.add(key)
    log.warning(event, **fields)


def _dedupe(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in symbols or ():
        symbol = str(raw or "").strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _as_day(value: Any) -> pd.Timestamp:
    """A date-like value as a naive midnight Timestamp (UTC date if tz-aware)."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _day_index(values: Any) -> pd.DatetimeIndex:
    """Anything date-like -> naive, midnight, nanosecond DatetimeIndex.

    Nanoseconds explicitly: pandas 3 infers microsecond resolution from strings,
    and merge_asof refuses keys whose resolutions differ.
    """
    parsed = pd.to_datetime(pd.Index(values), utc=True, format="mixed", errors="coerce")
    return pd.DatetimeIndex(parsed).tz_localize(None).normalize().astype(_NS)


def _utc_series(values: pd.Series) -> pd.Series:
    """Timestamps in any stored ISO spelling -> UTC nanosecond Series (NaT if unparseable)."""
    parsed = pd.to_datetime(values, utc=True, format="mixed", errors="coerce")
    return pd.Series(parsed, index=values.index).astype(_NS_UTC)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    """A text column as stripped Python strings, "" where missing."""
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    values = [("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
              for v in frame[column].tolist()]
    return pd.Series(values, index=frame.index, dtype=object)


def _nan_series(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.nan, index=idx, dtype=float)


def _safe_load(fn: Callable[..., Any], *args: Any, source: str, default: Any) -> Any:
    """Call a database loader; on failure log once and fall back to `default`.

    A missing table (an old or read-only database) or a transient error in one
    source must cost that source's features, not the whole panel.
    """
    try:
        return fn(*args)
    except Exception as e:
        _warn_once("features.load_failed", source=source,
                   error=f"{type(e).__name__}: {e}")
        return default


def _signed_log_scale(value: Any, cap: float = 10.0) -> Any:
    """Compress signed dollar amounts onto roughly [-1, 1].

    The predictor's v3 helper, vectorized: sign(x) * min(log10(1 + |x|) / cap, 1).
    Insider trade sizes span six orders of magnitude, from a $20k director
    purchase to a $500m block sale; a signed log keeps order and direction
    while flattening the scale. Accepts a scalar or an array.
    """
    arr = np.asarray(value, dtype=float)
    out = np.sign(arr) * np.minimum(np.log10(1.0 + np.abs(arr)) / cap, 1.0)
    return float(out) if out.ndim == 0 else out


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Standard score of each value against the trailing window that ends on it.

    Population std, as the v3 `_z_score` used. NaN until the window is full and
    on a flat window, where no deviation can be measured.
    """
    rolling = series.rolling(window, min_periods=window)
    std = rolling.std(ddof=0)
    return ((series - rolling.mean()) / std).where(std > 0)


def _cutoffs(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Each session date's visibility cutoff: d 23:30 UTC."""
    return (idx.tz_localize("UTC") + _CUTOFF_OFFSET).astype(_NS_UTC)


def _asof(idx: pd.DatetimeIndex, cutoffs: pd.DatetimeIndex, series: pd.DataFrame,
          max_stale_days: int) -> pd.Series:
    """The newest `value` published by each row's cutoff, NaN if stale or absent.

    `series` has available_at (UTC), session_date and value. On an
    available_at tie the later session wins. A match whose session is more than
    `max_stale_days` older than the row is dropped: a reading from weeks ago
    presents as a live one and is worse than none.
    """
    right = series.dropna(subset=["available_at", "session_date"])
    if right.empty:
        return _nan_series(idx)
    right = right.sort_values(["available_at", "session_date"], kind="stable")
    right = right.assign(available_at=right["available_at"].astype(_NS_UTC))
    left = pd.DataFrame({"cutoff": cutoffs, "row_date": idx})
    merged = pd.merge_asof(left, right[["available_at", "session_date", "value"]],
                           left_on="cutoff", right_on="available_at",
                           direction="backward")
    age = (merged["row_date"] - merged["session_date"]).dt.days
    values = merged["value"].where(age <= max_stale_days)
    return pd.Series(values.to_numpy(dtype=float), index=idx)


# ── Bars ─────────────────────────────────────────────────────────────────────

def _normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a bar frame to the canonical shape; a no-op on one already in it."""
    if (isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is None
            and frame.index.dtype == _NS and frame.index.is_monotonic_increasing
            and frame.index.is_unique and list(frame.columns) == list(_BAR_COLUMNS)
            and all(frame[c].dtype == float for c in _BAR_COLUMNS)):
        return frame

    out = frame.copy()
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index("date")
    out.index = _day_index(out.index)
    out = out[out.index.notna()].sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="last")]
    data = {c: (pd.to_numeric(out[c], errors="coerce").astype(float) if c in out.columns
                else pd.Series(np.nan, index=out.index, dtype=float))
            for c in _BAR_COLUMNS}
    result = pd.DataFrame(data, index=out.index)
    result.index.name = None
    return result


def clean_daily_bars(frame: pd.DataFrame, *, ticker: Optional[str] = None,
                     max_gap_days: int = MAX_SESSION_GAP_DAYS,
                     keep_weekends: bool = False) -> pd.DataFrame:
    """Keep only the trailing run of true daily bars.

    Drops weekend-dated rows (unless `keep_weekends`, for symbols that trade
    then), then everything before the last gap of more than `max_gap_days`
    calendar days between consecutive bars. That removes the monthly bars Yahoo
    returns for range=max, which sit in front of the daily history for some
    symbols, and any older fragment separated from the present by a hole.
    Expects a sorted DatetimeIndex. Logs once per ticker when rows are dropped.
    """
    if frame is None or len(frame) == 0:
        return frame
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index(kind="stable")

    kept = frame if keep_weekends else frame[pd.DatetimeIndex(frame.index).dayofweek < 5]
    weekend_rows = len(frame) - len(kept)

    before_segment = 0
    if len(kept) >= 2:
        days = pd.DatetimeIndex(kept.index).values.astype("datetime64[D]")
        gaps = np.diff(days).astype(np.int64)
        breaks = np.flatnonzero(gaps > max_gap_days)
        if breaks.size:
            before_segment = int(breaks[-1]) + 1
            kept = kept.iloc[before_segment:]

    if ticker and (weekend_rows or before_segment):
        _warn_once("features.bars_cleaned", ticker=ticker,
                   rows_dropped=weekend_rows + before_segment,
                   weekend_rows=weekend_rows,
                   segment_start=kept.index[0].strftime("%Y-%m-%d") if len(kept) else None)
    return kept


def splits_series(rows: Optional[Iterable[dict]]) -> pd.Series:
    """`get_price_splits` rows -> ratios indexed by effective session, oldest first.

    Ratios that are non-positive, non-finite or exactly 1 are dropped.
    """
    rows = list(rows or [])
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype=_NS))
    frame = pd.DataFrame(rows)
    ratios = _numeric(frame, "ratio").to_numpy()
    index = _day_index(frame["date"] if "date" in frame.columns else [None] * len(frame))
    series = pd.Series(ratios, index=index, dtype=float)
    keep = np.isfinite(series.to_numpy()) & (series.to_numpy() > 0) \
        & (series.to_numpy() != 1.0) & np.asarray(series.index.notna())
    series = series[keep].sort_index(kind="stable")
    return series[~series.index.duplicated(keep="last")]


def adjust_for_splits(bars: pd.DataFrame, splits: Optional[pd.Series], *,
                      ticker: Optional[str] = None) -> pd.DataFrame:
    """Repair split seams in stored bars; a history without one is returned as-is.

    Stored bars carry whatever split adjustment Yahoo applied on the day each
    row was last fetched (see the module docstring), so the scale can change
    between two consecutive rows anywhere before a split. Because every fetch
    covers a window ending at the fetch, the rows' fetch times rise with their
    dates, and the splits an older row is still unadjusted for are always the
    most recent ones: going back in time, a seam adds the next-older run of
    splits. This walks the one-session jumps from newest to oldest and accepts
    one as a seam when it matches -log of the product of such a run (within
    SPLIT_MATCH_TOLERANCE), with every split in the run dated after the older
    row. Rows before the seam are rescaled onto the newest row's scale: prices
    divided by the ratio product, volume multiplied by it.

    Absolute scale is irrelevant to the features (all ratios and log returns),
    so a split dated after the last bar, or before the first, changes nothing.
    """
    if bars is None or len(bars) < 2 or splits is None or len(splits) == 0:
        return bars
    dates = bars.index
    events = splits[(splits.index > dates[0]) & (splits.index <= dates[-1])].sort_index()
    if events.empty:
        return bars

    split_days = events.index.values
    log_ratio = np.log(events.to_numpy(dtype=float))
    close = bars["close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        jumps = np.diff(np.log(close))
    floor = max(SPLIT_MIN_LOG_RATIO - SPLIT_MATCH_TOLERANCE, 0.0)
    candidates = np.flatnonzero(np.abs(np.nan_to_num(jumps, nan=0.0)) >= floor)

    bar_days = dates.values
    pending = len(log_ratio)          # events[:pending] are not yet matched
    seams: list[tuple[int, float]] = []  # (last row of the older segment, log ratio)
    for i in candidates[::-1]:
        if pending == 0:
            break
        older_day = bar_days[i]
        best: Optional[tuple[int, float, float]] = None
        total = 0.0
        for j in range(pending - 1, -1, -1):
            if split_days[j] <= older_day:
                break
            total += log_ratio[j]
            if abs(total) < SPLIT_MIN_LOG_RATIO:
                continue
            miss = abs(jumps[i] + total)
            if miss <= min(SPLIT_MATCH_TOLERANCE, 0.4 * abs(total)) and (
                    best is None or miss < best[1]):
                best = (j, miss, total)
        if best is not None:
            pending = best[0]
            seams.append((int(i), best[2]))

    if not seams:
        return bars

    log_scale = np.zeros(len(close))
    for last_older, total in seams:
        log_scale[: last_older + 1] += total
    scale = np.exp(log_scale)
    out = bars.copy()
    for column in ("open", "high", "low", "close"):
        out[column] = out[column].to_numpy(dtype=float) / scale
    out["volume"] = out["volume"].to_numpy(dtype=float) * scale
    if ticker:
        log.info("features.split_seam_repaired", ticker=ticker,
                 seams=[{"first_new_scale_session": dates[i + 1].strftime("%Y-%m-%d"),
                         "ratio": round(float(np.exp(total)), 6)} for i, total in seams])
    return out


def prepare_bars(frame: Optional[pd.DataFrame], splits: Optional[pd.Series] = None, *,
                 ticker: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Stored OHLCV -> the canonical bar frame the builder reads, or None if empty.

    Normalizes the frame, drops non-positive closes, keeps the trailing daily
    segment and repairs split seams, in that order.
    """
    if frame is None or len(frame) == 0:
        return None
    bars = _normalize_bars(frame)
    bars = bars[bars["close"] > 0]
    bars = clean_daily_bars(bars, ticker=ticker)
    if bars is None or len(bars) == 0:
        return None
    return adjust_for_splits(bars, splits, ticker=ticker)


def choose_market_symbol(market: dict[str, pd.DataFrame],
                         candidates: tuple[str, ...] = MARKET_INDEX_CANDIDATES) -> Optional[str]:
    """The index series the market features read: earliest clean daily start wins.

    Candidates need two bars and must be current to within
    MARKET_MAX_STALE_DAYS of the freshest candidate. Ties go to the earlier
    candidate. None when no candidate is usable.
    """
    usable = {s: market[s] for s in candidates
              if market.get(s) is not None and len(market[s]) >= 2}
    if not usable:
        return None
    newest = max(frame.index[-1] for frame in usable.values())
    fresh = [s for s in candidates if s in usable
             and (newest - usable[s].index[-1]).days <= MARKET_MAX_STALE_DAYS]
    return min(fresh, key=lambda s: usable[s].index[0])


# ── Alternative data preparation ─────────────────────────────────────────────

def _empty_news_daily() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float) for c in NEWS_DAILY_COLUMNS},
                        index=pd.DatetimeIndex([], dtype=_NS, name="news_date"))


def _news_dates(published: pd.Series) -> pd.Series:
    """The first session date whose 23:30 UTC cutoff an article makes.

    ceil(published - 23:30) rather than floor(published + 30 min): the two agree
    except at exactly 23:30:00, which belongs to d under `published <= d 23:30`.
    """
    shifted = (published - _CUTOFF_OFFSET).dt.ceil("D")
    return shifted.dt.tz_localize(None).astype(_NS)


def aggregate_news_rows(rows: Optional[Iterable[dict]]) -> pd.DataFrame:
    """`get_ticker_news_rows` rows -> one row per news_date of sums and counts.

    Columns are NEWS_DAILY_COLUMNS: article count, sentiment sum and count,
    importance sum, count and max, bullish and bearish counts, and the max
    urgency code (low=0 .. critical=3). Rows with an unparseable timestamp are
    dropped.
    """
    rows = list(rows or [])
    if not rows:
        return _empty_news_daily()
    frame = pd.DataFrame(rows)
    if "published_at" not in frame.columns:
        return _empty_news_daily()
    published = _utc_series(frame["published_at"])
    keep = published.notna().to_numpy()
    if not keep.any():
        return _empty_news_daily()
    frame = frame[keep]
    published = published[keep]

    sentiment = _numeric(frame, "sentiment_score")
    importance = _numeric(frame, "importance_score")
    direction = _text(frame, "suggested_direction").str.lower()
    urgency = _text(frame, "urgency").str.lower().map(URGENCY_CODES)

    daily = pd.DataFrame({
        "news_date": _news_dates(published).to_numpy(),
        "n": np.ones(len(frame)),
        "sent_sum": sentiment.fillna(0.0).to_numpy(),
        "sent_cnt": sentiment.notna().to_numpy(dtype=float),
        "imp_sum": importance.fillna(0.0).to_numpy(),
        "imp_cnt": importance.notna().to_numpy(dtype=float),
        "imp_max": importance.to_numpy(),
        "bull": (direction == "bullish").to_numpy(dtype=float),
        "bear": (direction == "bearish").to_numpy(dtype=float),
        "urg_max": pd.to_numeric(urgency, errors="coerce").to_numpy(dtype=float),
    })
    grouped = daily.groupby("news_date").agg(
        n=("n", "sum"), sent_sum=("sent_sum", "sum"), sent_cnt=("sent_cnt", "sum"),
        imp_sum=("imp_sum", "sum"), imp_cnt=("imp_cnt", "sum"), imp_max=("imp_max", "max"),
        bull=("bull", "sum"), bear=("bear", "sum"), urg_max=("urg_max", "max"),
    )
    grouped.index = pd.DatetimeIndex(grouped.index).astype(_NS)
    grouped.index.name = "news_date"
    return grouped[NEWS_DAILY_COLUMNS].astype(float)


def _empty_offexch() -> pd.DataFrame:
    return pd.DataFrame({
        "session_date": pd.Series(dtype=_NS),
        "available_at": pd.Series(dtype=_NS_UTC),
        "short_ratio": pd.Series(dtype=float),
        "share": pd.Series(dtype=float),
    })


def _split_multiplier(dates: pd.DatetimeIndex, splits: Optional[pd.Series],
                      last_date: pd.Timestamp) -> np.ndarray:
    """For each date, the product of split ratios effective after it (through last_date).

    Converts an as-traded share count on that date onto the scale of the
    newest bar.
    """
    multiplier = np.ones(len(dates))
    if splits is None or len(splits) == 0 or len(dates) == 0:
        return multiplier
    events = splits[splits.index <= last_date].sort_index()
    if events.empty:
        return multiplier
    suffix = np.append(np.cumprod(events.to_numpy(dtype=float)[::-1])[::-1], 1.0)
    position = np.searchsorted(events.index.values, dates.values, side="right")
    return suffix[position]


def prepare_offexch(rows: Optional[Iterable[dict]], bars: Optional[pd.DataFrame] = None,
                    splits: Optional[pd.Series] = None) -> pd.DataFrame:
    """`get_offexchange_series` rows -> session_date, available_at, short_ratio, share.

    short_ratio = short volume / off-exchange volume. share = off-exchange
    volume / consolidated volume, where the consolidated volume comes from the
    prepared `bars` when given: FINRA reports as-traded shares while the stored
    bars are on the newest split scale, so FINRA's figure is moved onto that
    scale first. Without bars it falls back to the row's own
    consolidated_volume. Either is NaN where the denominator is missing or zero.
    """
    rows = list(rows or [])
    if not rows:
        return _empty_offexch()
    frame = pd.DataFrame(rows)
    if "session_date" not in frame.columns or "published_at" not in frame.columns:
        return _empty_offexch()
    sessions = _day_index(frame["session_date"])
    total = _numeric(frame, "total_volume").to_numpy()
    short = _numeric(frame, "short_volume").to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        short_ratio = np.where(total > 0, short / total, np.nan)
        if bars is not None and len(bars):
            bar_volume = bars["volume"].reindex(sessions).to_numpy(dtype=float)
            traded = bar_volume / _split_multiplier(sessions, splits, bars.index[-1])
            share = np.where((traded > 0) & (total > 0), total / traded, np.nan)
        else:
            consolidated = _numeric(frame, "consolidated_volume").to_numpy()
            share = np.where((consolidated > 0) & (total > 0), total / consolidated, np.nan)

    out = pd.DataFrame({
        "session_date": sessions,
        "available_at": _utc_series(frame["published_at"]).to_numpy(),
        "short_ratio": short_ratio,
        "share": share,
    })
    out["available_at"] = out["available_at"].astype(_NS_UTC)
    out = out.dropna(subset=["session_date", "available_at"])
    return out.sort_values("session_date", kind="stable").reset_index(drop=True)


def _empty_insider() -> pd.DataFrame:
    return pd.DataFrame({
        "filed_date": pd.Series(dtype=_NS),
        "transaction_date": pd.Series(dtype=_NS),
        "code": pd.Series(dtype=object),
        "discretionary": pd.Series(dtype=bool),
        "value": pd.Series(dtype=float),
        "name": pd.Series(dtype=object),
    })


def prepare_insider(rows: Optional[Iterable[dict]]) -> pd.DataFrame:
    """`get_insider_series` rows -> filed_date, transaction_date, code, discretionary, value, name.

    filed_date is the UTC calendar day of filed_at (EDGAR stamps a filing at
    23:59:59 UTC on its filing date). value is the absolute dollar size.
    """
    rows = list(rows or [])
    if not rows:
        return _empty_insider()
    frame = pd.DataFrame(rows)
    if "filed_at" not in frame.columns:
        return _empty_insider()
    tx_dates = frame["transaction_date"] if "transaction_date" in frame.columns \
        else pd.Series([None] * len(frame), index=frame.index)
    out = pd.DataFrame({
        "filed_date": _day_index(frame["filed_at"]),
        "transaction_date": _day_index(tx_dates),
        "code": _text(frame, "transaction_code").str.upper().to_numpy(dtype=object),
        "discretionary": (_numeric(frame, "is_discretionary").fillna(0.0) > 0).to_numpy(),
        "value": _numeric(frame, "value_usd").abs().fillna(0.0).to_numpy(),
        "name": _text(frame, "insider_name").to_numpy(dtype=object),
    })
    out = out.dropna(subset=["filed_date"])
    return out.sort_values("filed_date", kind="stable").reset_index(drop=True)


def prepare_regime(rows: Optional[Iterable[dict]]) -> pd.DataFrame:
    """`get_market_regime_series` rows -> long frame: metric, session_date, available_at, value."""
    rows = list(rows or [])
    if not rows:
        return _empty_regime()
    frame = pd.DataFrame(rows)
    if not {"metric", "session_date", "published_at"} <= set(frame.columns):
        return _empty_regime()
    out = pd.DataFrame({
        "metric": _text(frame, "metric").to_numpy(dtype=object),
        "session_date": _day_index(frame["session_date"]),
        "available_at": _utc_series(frame["published_at"]).to_numpy(),
        "value": _numeric(frame, "value").to_numpy(),
    })
    out["available_at"] = out["available_at"].astype(_NS_UTC)
    out = out.dropna(subset=["session_date", "available_at", "value"])
    return out.sort_values(["metric", "session_date"], kind="stable").reset_index(drop=True)


# ── Feature groups ───────────────────────────────────────────────────────────

def _price_features(bars: pd.DataFrame) -> dict[str, pd.Series]:
    close = bars["close"]
    open_ = bars["open"].where(bars["open"] > 0)
    high, low = bars["high"], bars["low"]
    volume = bars["volume"].where(bars["volume"] > 0)

    log_close = np.log(close)
    log_ret = log_close.diff()
    vol_21 = log_ret.rolling(21, min_periods=21).std()
    vol_63 = log_ret.rolling(63, min_periods=63).std()

    out: dict[str, pd.Series] = {}
    # Vol-scaled momentum: a 5% month means something different for KO and TSLA.
    for k in (1, 5, 21, 63, 126, 252):
        out[f"ret_{k}d"] = (log_close - log_close.shift(k)) / (vol_21 * np.sqrt(k))
    out["mom_12_1"] = (log_close.shift(21) - log_close.shift(252)) / (vol_21 * np.sqrt(231))

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    std20 = close.rolling(20, min_periods=20).std()
    out["dist_sma20"] = close / sma20 - 1.0
    out["dist_sma50"] = close / sma50 - 1.0
    out["dist_sma200"] = close / sma200 - 1.0
    out["sma50_200"] = sma50 / sma200 - 1.0

    # Wilder RSI with TradingView's guards, in Pine's order: no losses is 100
    # even when there are no gains either.
    change = close.diff()
    gain = _rma(change.clip(lower=0.0), 14)
    loss = _rma((-change).clip(lower=0.0), 14)
    rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    out["rsi_14"] = rsi.mask(gain == 0, 0.0).mask(loss == 0, 100.0)

    macd = _ema(close, 12) - _ema(close, 26)
    histogram = macd - _ema(macd, 9)
    warmed = pd.Series(np.arange(len(close)) >= MACD_WARMUP_BARS, index=close.index)
    out["macd_hist_pct"] = (histogram / close).where(warmed)

    out["bb_pos"] = (close - sma20) / (2.0 * std20)
    out["bb_width"] = 4.0 * std20 / sma20

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    out["atr_pct"] = _rma(true_range, 14) / close

    out["vol_21"] = vol_21
    out["vol_63"] = vol_63
    out["vol_ratio"] = vol_21 / vol_63

    out["dd_63"] = close / close.rolling(63, min_periods=63).max() - 1.0
    out["dist_52w_high"] = close / close.rolling(252, min_periods=252).max() - 1.0
    out["dist_52w_low"] = close / close.rolling(252, min_periods=252).min() - 1.0

    out["volume_z20"] = _rolling_z(np.log(volume), 20)
    out["log_dollar_vol"] = np.log((close * volume).rolling(20, min_periods=20).mean())

    out["gap_1d"] = open_ / prev_close - 1.0
    out["intraday_1d"] = close / open_ - 1.0
    up = (log_ret > 0).astype(float).where(log_ret.notna())
    out["up_days_10"] = up.rolling(10, min_periods=10).mean()
    out["skew_63"] = log_ret.rolling(63, min_periods=63).skew()
    return out


def _market_features(symbol: str, bars: pd.DataFrame, vol_21: pd.Series,
                     inputs: PanelInputs) -> dict[str, pd.Series]:
    idx = bars.index
    close = bars["close"]
    out = {name: _nan_series(idx) for name in FEATURE_GROUPS["market"]}

    index_symbol = inputs.market_symbol
    index_bars = inputs.market.get(index_symbol) if index_symbol else None
    if index_bars is None or len(index_bars) < 2:
        _warn_once("features.market_index_missing", market_symbol=index_symbol,
                   available=sorted(inputs.market))
    else:
        m_close = _normalize_bars(index_bars)["close"]
        m_log = np.log(m_close)
        m_vol_21 = m_log.diff().rolling(21, min_periods=21).std()
        for k in (5, 21, 63):
            out[f"spy_ret_{k}d"] = ((m_log - m_log.shift(k)) / (m_vol_21 * np.sqrt(k))).reindex(idx)
        out["spy_dist_sma200"] = (m_close / m_close.rolling(200, min_periods=200).mean() - 1.0).reindex(idx)
        out["spy_vol_21"] = m_vol_21.reindex(idx)

        # Relative moves on the sessions both series traded; nothing is filled.
        joined = pd.concat({"t": close, "m": m_close}, axis=1, join="inner")
        t_log, j_log = np.log(joined["t"]), np.log(joined["m"])
        scale = vol_21.reindex(joined.index)
        for k in (5, 21, 63):
            rel = ((t_log - t_log.shift(k)) - (j_log - j_log.shift(k))) / (scale * np.sqrt(k))
            out[f"rel_spy_{k}d"] = rel.reindex(idx)
        t_ret, j_ret = t_log.diff(), j_log.diff()
        covariance = t_ret.rolling(63, min_periods=63).cov(j_ret)
        variance = j_ret.rolling(63, min_periods=63).var()
        out["beta_63"] = (covariance / variance).where(variance > 0).reindex(idx)
        out["corr_63"] = t_ret.rolling(63, min_periods=63).corr(j_ret).reindex(idx)

    sector_etf = sector_etf_for(inputs.sector_of.get(symbol))
    sector_bars = inputs.market.get(sector_etf) if sector_etf and sector_etf != symbol else None
    if sector_bars is not None and len(sector_bars) >= 2:
        e_close = _normalize_bars(sector_bars)["close"]
        joined = pd.concat({"t": close, "e": e_close}, axis=1, join="inner")
        t_log, e_log = np.log(joined["t"]), np.log(joined["e"])
        out["rel_sector_21d"] = ((t_log - t_log.shift(21)) - (e_log - e_log.shift(21))).reindex(idx)

    vix_bars = inputs.market.get(VIX_SYMBOL)
    if vix_bars is not None and len(vix_bars):
        vix = _normalize_bars(vix_bars)["close"]
        out["vix_level"] = vix.reindex(idx)
        out["vix_z63"] = _rolling_z(vix, 63).reindex(idx)
        out["vix_chg_5d"] = np.log(vix / vix.shift(5)).reindex(idx)

    tnx_bars = inputs.market.get(TNX_SYMBOL)
    if tnx_bars is not None and len(tnx_bars):
        yield_pct = _normalize_bars(tnx_bars)["close"]
        out["tnx_chg_5d"] = (yield_pct - yield_pct.shift(5)).reindex(idx)
        out["tnx_chg_21d"] = (yield_pct - yield_pct.shift(21)).reindex(idx)
    return out


def _regime_features(idx: pd.DatetimeIndex, regime: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
    out = {name: _nan_series(idx) for name in REGIME_METRICS.values()}
    if regime is None or len(regime) == 0:
        return out
    cutoffs = _cutoffs(idx)
    for metric, name in REGIME_METRICS.items():
        rows = regime[regime["metric"] == metric]
        if rows.empty:
            continue
        rows = rows.sort_values("session_date", kind="stable")
        rows = rows.drop_duplicates("session_date", keep="last").reset_index(drop=True)
        # z on the series' own session order, then made visible by publication.
        z = _rolling_z(rows["value"].astype(float), REGIME_Z_WINDOW)
        series = rows[["available_at", "session_date"]].assign(value=z.to_numpy())
        out[name] = _asof(idx, cutoffs, series, REGIME_MAX_STALE_DAYS)
    return out


def _darkpool_features(idx: pd.DatetimeIndex, offexch: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
    names = ("offexch_short_ratio_z20", "offexch_short_ratio_mom", "offexch_share_z20")
    out = {name: _nan_series(idx) for name in names}
    if offexch is None or len(offexch) == 0:
        return out
    frame = offexch.dropna(subset=["session_date", "available_at"])
    frame = frame.sort_values("session_date", kind="stable")
    frame = frame.drop_duplicates("session_date", keep="last")
    cutoffs = _cutoffs(idx)

    # Each leg is windowed over its own valid sessions, as v3 did: a session
    # with no consolidated volume skips the share window without costing the
    # short-ratio window a slot.
    short = frame[frame["short_ratio"].notna()].reset_index(drop=True)
    if len(short):
        ratio = short["short_ratio"].astype(float)
        base = short[["available_at", "session_date"]]
        z = _rolling_z(ratio, OFFEXCH_Z_WINDOW)
        # "Unusual today" and "trending" are different readings; a ratio can sit
        # inside its band for a week while walking steadily across it.
        momentum = (ratio.rolling(OFFEXCH_MOM_WINDOW, min_periods=OFFEXCH_MOM_WINDOW).mean()
                    - ratio.rolling(OFFEXCH_Z_WINDOW, min_periods=OFFEXCH_Z_WINDOW).mean())
        out["offexch_short_ratio_z20"] = _asof(idx, cutoffs, base.assign(value=z.to_numpy()),
                                               OFFEXCH_MAX_STALE_DAYS)
        out["offexch_short_ratio_mom"] = _asof(idx, cutoffs, base.assign(value=momentum.to_numpy()),
                                               OFFEXCH_MAX_STALE_DAYS)

    share = frame[frame["share"].notna()].reset_index(drop=True)
    if len(share):
        z = _rolling_z(share["share"].astype(float), OFFEXCH_Z_WINDOW)
        out["offexch_share_z20"] = _asof(
            idx, cutoffs, share[["available_at", "session_date"]].assign(value=z.to_numpy()),
            OFFEXCH_MAX_STALE_DAYS)
    return out


def _distinct_in_window(idx: pd.DatetimeIndex, filed: pd.Series, names: pd.Series,
                        window_days: int) -> np.ndarray:
    """Distinct names filed in (d - window_days, d] for each row date d."""
    result = np.zeros(len(idx))
    if len(filed) == 0:
        return result
    days = filed.to_numpy().astype("datetime64[D]")
    order = np.argsort(days, kind="stable")
    days = days[order]
    who = np.asarray(names, dtype=object)[order]
    span = np.timedelta64(int(window_days), "D")
    counts: Counter = Counter()
    lo = hi = 0
    for i, day in enumerate(idx.values.astype("datetime64[D]")):
        while hi < len(days) and days[hi] <= day:
            counts[who[hi]] += 1
            hi += 1
        while lo < hi and days[lo] <= day - span:
            counts[who[lo]] -= 1
            if counts[who[lo]] == 0:
                del counts[who[lo]]
            lo += 1
        result[i] = len(counts)
    return result


def _insider_features(idx: pd.DatetimeIndex, insider: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
    names = ("insider_buy_ratio_90d", "insider_net_30d", "insider_cluster_30d",
             "days_since_insider_buy")
    out = {name: _nan_series(idx) for name in names}
    if insider is None or len(insider) == 0:
        return out

    # Before the first visible filing the series had not started for this
    # ticker: that is missing, not "no insider trading".
    first_visible = insider["filed_date"].min()
    visible = pd.Series(idx >= first_visible, index=idx)
    if not visible.any():
        return out

    calendar = pd.date_range(min(first_visible, idx[0]), idx[-1], freq="D")
    # Grants, option exercises and tax withholding are compensation mechanics;
    # only open-market buys and sales carry a view.
    disc = insider[insider["discretionary"].astype(bool)]
    is_buy = (disc["code"] == "P").to_numpy()
    value = disc["value"].to_numpy(dtype=float)
    flows = pd.DataFrame({
        "buy": np.where(is_buy, value, 0.0),
        "sell": np.where(is_buy, 0.0, value),
        "net": np.where(is_buy, value, -value),
        "n_buy": is_buy.astype(float),
        "n_sell": (~is_buy).astype(float),
        "n": np.ones(len(disc)),
    }, index=pd.DatetimeIndex(disc["filed_date"]))
    daily = flows.groupby(level=0).sum().reindex(calendar, fill_value=0.0)

    def window(column: str, days: int) -> pd.Series:
        return daily[column].rolling(days, min_periods=1).sum()

    # Counts gate the dollar sums: a window whose trades have all rolled out can
    # keep a floating-point residue that would otherwise read as a tiny ratio.
    buy_90 = window("buy", INSIDER_RATIO_WINDOW).where(window("n_buy", INSIDER_RATIO_WINDOW) > 0, 0.0)
    sell_90 = window("sell", INSIDER_RATIO_WINDOW).where(window("n_sell", INSIDER_RATIO_WINDOW) > 0, 0.0)
    total_90 = buy_90 + sell_90
    ratio = (buy_90 / total_90).where(total_90 > 0)
    net_30 = window("net", INSIDER_NET_WINDOW).where(window("n", INSIDER_NET_WINDOW) > 0, 0.0)

    out["insider_buy_ratio_90d"] = ratio.reindex(idx).where(visible)
    out["insider_net_30d"] = pd.Series(_signed_log_scale(net_30.to_numpy()),
                                       index=calendar).reindex(idx).where(visible)

    buys = disc[is_buy]
    named = buys[buys["name"].astype(str) != ""]
    out["insider_cluster_30d"] = pd.Series(
        _distinct_in_window(idx, named["filed_date"], named["name"], INSIDER_CLUSTER_WINDOW),
        index=idx).where(visible)

    dated = buys.dropna(subset=["transaction_date"]).sort_values("filed_date", kind="stable")
    if len(dated):
        latest = pd.DataFrame({
            "filed_date": dated["filed_date"].astype(_NS).to_numpy(),
            "last_buy": dated["transaction_date"].cummax().astype(_NS).to_numpy(),
        })
        merged = pd.merge_asof(pd.DataFrame({"row_date": idx}), latest,
                               left_on="row_date", right_on="filed_date", direction="backward")
        elapsed = (merged["row_date"] - merged["last_buy"]).dt.days.to_numpy(dtype=float)
        out["days_since_insider_buy"] = pd.Series(
            np.clip(elapsed, 0.0, DAYS_SINCE_BUY_CAP), index=idx).where(visible)
    return out


def _smart_money_features(symbol: str, idx: pd.DatetimeIndex,
                          inputs: PanelInputs) -> dict[str, pd.Series]:
    out = {name: _nan_series(idx) for name in FEATURE_GROUPS["smart_money"]}
    # FINRA and SEC Form 4 cover US listings only; everything else stays NaN.
    if classify_market(symbol) != US:
        return out
    out.update(_darkpool_features(idx, inputs.offexch.get(symbol)))
    out.update(_insider_features(idx, inputs.insider.get(symbol)))
    return out


def _sentiment_features(idx: pd.DatetimeIndex, news_daily: Optional[pd.DataFrame],
                        coverage_start: Any) -> dict[str, pd.Series]:
    out = {name: _nan_series(idx) for name in FEATURE_GROUPS["sentiment"]}
    if coverage_start is None:
        return out
    coverage = _as_day(coverage_start)
    if idx[-1] < coverage:
        return out

    has_news = news_daily is not None and len(news_daily) > 0
    start = min(idx[0], news_daily.index.min()) if has_news else idx[0]
    calendar = pd.date_range(start, idx[-1], freq="D")
    if has_news:
        daily = news_daily.reindex(calendar)
        sums = [c for c in NEWS_DAILY_COLUMNS if c not in _NEWS_MAX_COLUMNS]
        daily[sums] = daily[sums].fillna(0.0)
    else:
        daily = pd.DataFrame(0.0, index=calendar, columns=NEWS_DAILY_COLUMNS)
        daily[list(_NEWS_MAX_COLUMNS)] = np.nan

    def total(column: str, days: int) -> pd.Series:
        return daily[column].rolling(days, min_periods=1).sum()

    def peak(column: str, days: int) -> pd.Series:
        return daily[column].rolling(days, min_periods=1).max()

    def mean_sentiment(days: int) -> pd.Series:
        count = total("sent_cnt", days)
        return (total("sent_sum", days) / count).where(count > 0)

    # Calendar windows (d-k, d] over news dates, so weekend news lands on Monday.
    n_3, n_14 = total("n", 3), total("n", 14)
    sent_1, sent_7 = mean_sentiment(1), mean_sentiment(7)
    imp_count = total("imp_cnt", 7)
    bull, bear = total("bull", 7), total("bear", 7)
    frame = pd.DataFrame({
        "sent_1d": sent_1,
        "sent_3d": mean_sentiment(3),
        "sent_7d": sent_7,
        "sent_mom": sent_1 - sent_7,
        "news_n_3d": n_3,
        "news_n_14d": n_14,
        "news_velocity": ((n_3 / 3.0) / (n_14 / 14.0)).where(n_14 > 0),
        "imp_avg_7d": (total("imp_sum", 7) / imp_count).where(imp_count > 0),
        "imp_max_7d": peak("imp_max", 7),
        "bull_ratio_7d": (bull / (bull + bear)).where((bull + bear) > 0),
        "urg_max_7d": peak("urg_max", 7),
    }, index=calendar).reindex(idx)

    covered = pd.Series(idx >= coverage, index=idx)
    for name in FEATURE_GROUPS["sentiment"]:
        out[name] = frame[name].where(covered)
    return out


def _calendar_features(idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    days = idx.values.astype("datetime64[D]")
    month_end = (idx + pd.offsets.MonthEnd(0)).values.astype("datetime64[D]")
    one_day = np.timedelta64(1, "D")
    # Weekdays left in the month after d. A weekday calendar, not the ticker's
    # own future bars: those do not exist yet on the live row, and training on
    # them would give the model a value it can never see in production.
    remaining = np.busday_count(days + one_day, month_end + one_day)
    return {
        "dow": pd.Series(idx.dayofweek.to_numpy(dtype=float), index=idx),
        "days_to_month_end": pd.Series(remaining.astype(float), index=idx),
    }


def _context_features(symbol: str, idx: pd.DatetimeIndex,
                      inputs: PanelInputs) -> dict[str, pd.Series]:
    etf = sector_etf_for(inputs.sector_of.get(symbol))
    code = SECTOR_CODES.get(etf) if etf else None
    is_fund = symbol in ETF_TICKERS or symbol.startswith("^")
    return {
        "sector_id": pd.Series(float(code) if code is not None else np.nan, index=idx),
        "is_etf": pd.Series(1.0 if is_fund else 0.0, index=idx),
        # Years of stored history: an IPO-recency proxy.
        "hist_len_years": pd.Series((idx - idx[0]).days.to_numpy(dtype=float) / 365.25, index=idx),
    }


def label_columns(horizons: Iterable[int] = HORIZONS) -> list[str]:
    """The label columns build_ticker_frame appends, in order."""
    return [c for h in horizons for c in (f"fwd_ret_{h}", f"y_{h}", f"label_date_{h}")]


def _labels(close: pd.Series, horizons: tuple[int, ...]) -> pd.DataFrame:
    log_close = np.log(close)
    session_dates = pd.Series(close.index, index=close.index)
    columns: dict[str, pd.Series] = {}
    for h in horizons:
        forward = log_close.shift(-h) - log_close
        columns[f"fwd_ret_{h}"] = forward
        columns[f"y_{h}"] = (forward > 0).astype(float).where(forward.notna())
        columns[f"label_date_{h}"] = session_dates.shift(-h)
    return pd.DataFrame(columns, index=close.index)


def _empty_ticker_frame(horizons: tuple[int, ...]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {name: pd.Series(dtype=float) for name in FEATURE_NAMES}
    for h in horizons:
        columns[f"fwd_ret_{h}"] = pd.Series(dtype=float)
        columns[f"y_{h}"] = pd.Series(dtype=float)
        columns[f"label_date_{h}"] = pd.Series(dtype=_NS)
    return pd.DataFrame(columns, index=pd.DatetimeIndex([], dtype=_NS, name="date"))


# ── Public builders ──────────────────────────────────────────────────────────

def build_ticker_frame(ticker: str, inputs: PanelInputs,
                       horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """One ticker's full feature history: a row per stored session.

    Index is the session date (named "date"); columns are FEATURE_NAMES followed
    by fwd_ret_{h}, y_{h}, label_date_{h} for each horizon. No rows are
    dropped here — warm-up rows carry NaN — so the last row is always the
    latest session, which is what the live row reads.
    """
    symbol = str(ticker or "").strip().upper()
    horizons = tuple(int(h) for h in horizons)
    raw = inputs.bars.get(symbol)
    if raw is None or len(raw) == 0:
        return _empty_ticker_frame(horizons)

    bars = _normalize_bars(raw)
    if len(bars) == 0:
        return _empty_ticker_frame(horizons)
    idx = bars.index

    columns: dict[str, pd.Series] = {}
    price = _price_features(bars)
    columns.update(price)
    columns.update(_market_features(symbol, bars, price["vol_21"], inputs))
    columns.update(_regime_features(idx, inputs.regime))
    columns.update(_smart_money_features(symbol, idx, inputs))
    columns.update(_sentiment_features(idx, inputs.news_daily.get(symbol), inputs.coverage_start))
    columns.update(_calendar_features(idx))
    columns.update(_context_features(symbol, idx, inputs))

    frame = pd.DataFrame(
        {name: columns[name].reindex(idx).to_numpy(dtype=float) for name in FEATURE_NAMES},
        index=idx,
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if horizons:
        frame = pd.concat([frame, _labels(bars["close"], horizons)], axis=1)
    # rename_axis, not `frame.index.name = ...`: the index object can be the
    # caller's own bars index, and renaming it in place would edit their inputs.
    return frame.rename_axis("date")


def build_panel(inputs: PanelInputs, tickers: Optional[Iterable[str]] = None,
                horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """Every ticker's frame stacked into one training panel, sorted by (date, ticker).

    Columns: ticker, date, FEATURE_NAMES, then the label columns. Per ticker
    it drops warm-up rows — up to WARMUP_ROWS, never so many that fewer than
    WARMUP_ROWS remain, and always every row before ret_21d and vol_21 exist —
    and any row whose raw one-day log return exceeds SPLIT_RESIDUE_LOG_RETURN
    (logged). A ticker too short to produce a row is logged and skipped;
    `tickers=None` means every ticker in `inputs.bars`.
    """
    horizons = tuple(int(h) for h in horizons)
    symbols = _dedupe(tickers if tickers is not None else inputs.bars.keys())
    parts: list[pd.DataFrame] = []
    skipped: list[str] = []

    for symbol in symbols:
        frame = build_ticker_frame(symbol, inputs, horizons)
        ready = (frame["ret_21d"].notna() & frame["vol_21"].notna()).to_numpy()
        if not ready.any():
            log.info("features.ticker_too_short", ticker=symbol, bars=len(frame))
            skipped.append(symbol)
            continue

        first_ready = int(np.argmax(ready))
        drop = max(first_ready, min(WARMUP_ROWS, len(frame) - WARMUP_ROWS))
        kept = frame.iloc[drop:]

        close = _normalize_bars(inputs.bars[symbol])["close"]
        jump = np.log(close).diff().abs().reindex(kept.index)
        residue = (jump > SPLIT_RESIDUE_LOG_RETURN).to_numpy()
        if residue.any():
            dates = [d.strftime("%Y-%m-%d") for d in kept.index[residue]]
            log.warning("features.split_residue_rows_dropped", ticker=symbol,
                        rows=len(dates), dates=dates[:10])
            kept = kept[~residue]
        if kept.empty:
            skipped.append(symbol)
            continue

        kept = kept.reset_index()
        kept.insert(0, "ticker", symbol)
        parts.append(kept)

    columns = ["ticker", "date", *FEATURE_NAMES, *label_columns(horizons)]
    if not parts:
        log.info("features.panel_built", tickers=0, rows=0, skipped=skipped)
        empty = _empty_ticker_frame(horizons).reset_index()
        empty.insert(0, "ticker", pd.Series(dtype=object))
        return empty[columns]

    panel = pd.concat(parts, ignore_index=True)[columns]
    panel = panel.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    log.info("features.panel_built", tickers=len(parts), rows=len(panel), skipped=skipped)
    return panel


def _load_sectors(db: Any, symbols: list[str]) -> dict[str, str]:
    """Sector per symbol: funds from the static map, the rest from ticker_info.

    `get_ticker_sector` falls back to yfinance on a cache miss and caches the
    answer, so a new symbol costs one lookup ever. Index and crypto symbols
    have no sector and are never looked up.
    """
    sectors: dict[str, str] = {}
    for symbol in symbols:
        if symbol in _FUND_SECTORS:
            sectors[symbol] = _FUND_SECTORS[symbol]
        elif symbol.startswith("^") or symbol.endswith("-USD"):
            sectors[symbol] = ""
        else:
            sectors[symbol] = _safe_load(db.get_ticker_sector, symbol,
                                         source="ticker_sector", default="") or ""
    return sectors


def _load_bars(db: Any, symbol: str) -> tuple[Optional[pd.DataFrame], pd.Series]:
    splits = splits_series(_safe_load(db.get_price_splits, symbol,
                                      source="price_splits", default=[]))
    raw = _safe_load(TechnicalRatingTracker.load_bars, db, symbol,
                     source="price_history", default=None)
    return prepare_bars(raw, splits, ticker=symbol), splits


def load_panel_inputs(db: Any, tickers: Iterable[str], *, start: Optional[str] = None) -> PanelInputs:
    """Read everything the builder needs for `tickers` from the database, once.

    One query per table per ticker, plus the market series (SPY, ^GSPC, ^VIX,
    ^TNX and each ticker's sector ETF), the regime series and the news coverage
    start. A source that fails to load is logged and left empty rather than
    failing the panel. `start` ('YYYY-MM-DD') trims every bar frame to sessions
    on or after it, after the market index has been chosen and dark-pool shares
    computed on the full history; rolling windows then warm up, and
    hist_len_years counts, from `start`. Leave it None for production training
    and the live row, so both see the same history.
    """
    symbols = _dedupe(tickers)
    inputs = PanelInputs(market_symbol=None)
    inputs.sector_of = _load_sectors(db, symbols)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        bars, splits = _load_bars(db, symbol)
        if bars is None:
            log.info("features.no_bars", ticker=symbol)
            continue
        bars_by_symbol[symbol] = bars
        inputs.splits[symbol] = splits
        inputs.news_daily[symbol] = aggregate_news_rows(
            _safe_load(db.get_ticker_news_rows, symbol, source="news", default=[]))
        if classify_market(symbol) == US:
            inputs.offexch[symbol] = prepare_offexch(
                _safe_load(db.get_offexchange_series, symbol, source="offexchange", default=[]),
                bars, splits)
            inputs.insider[symbol] = prepare_insider(
                _safe_load(db.get_insider_series, symbol, source="insider", default=[]))

    wanted = [*MARKET_INDEX_CANDIDATES, VIX_SYMBOL, TNX_SYMBOL]
    wanted += [etf for etf in (sector_etf_for(inputs.sector_of.get(s)) for s in symbols) if etf]
    market: dict[str, pd.DataFrame] = {}
    for symbol in _dedupe(wanted):
        if symbol in bars_by_symbol:
            market[symbol] = bars_by_symbol[symbol]
            continue
        bars, _ = _load_bars(db, symbol)
        if bars is not None:
            market[symbol] = bars

    inputs.market_symbol = choose_market_symbol(market)
    if inputs.market_symbol is None:
        _warn_once("features.market_index_missing", market_symbol=None,
                   candidates=list(MARKET_INDEX_CANDIDATES))

    inputs.regime = prepare_regime(
        _safe_load(db.get_market_regime_series, source="market_regime", default=[]))
    coverage = _safe_load(db.get_news_coverage_start, source="news_coverage", default=None)
    inputs.coverage_start = _as_day(coverage) if coverage else None

    if start:
        first = _as_day(start)
        bars_by_symbol = {s: b[b.index >= first] for s, b in bars_by_symbol.items()}
        bars_by_symbol = {s: b for s, b in bars_by_symbol.items() if len(b)}
        market = {s: b[b.index >= first] for s, b in market.items()}
        market = {s: b for s, b in market.items() if len(b)}
    inputs.bars = bars_by_symbol
    inputs.market = market

    log.info("features.inputs_loaded", tickers=len(inputs.bars), requested=len(symbols),
             market=sorted(inputs.market), market_symbol=inputs.market_symbol,
             coverage_start=(inputs.coverage_start.strftime("%Y-%m-%d")
                             if inputs.coverage_start is not None else None))
    return inputs


def build_live_row(db: Any, ticker: str) -> Optional[tuple[pd.Series, dict]]:
    """The newest session's features for one ticker, plus where they came from.

    Returns (row, meta): `row` is a float Series indexed by FEATURE_NAMES (NaN
    where a window is still too short), `meta` is {"asof_date", "bars",
    "stale_days"} — the session the row describes, how many clean daily bars
    stood behind it, and how many calendar days old that session is. Returns
    None, logging why, when fewer than MIN_LIVE_BARS clean bars exist.
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    inputs = load_panel_inputs(db, [symbol])
    bars = inputs.bars.get(symbol)
    n_bars = 0 if bars is None else len(bars)
    if n_bars < MIN_LIVE_BARS:
        log.warning("features.live_row_unavailable", ticker=symbol, bars=n_bars,
                    reason=f"fewer than {MIN_LIVE_BARS} clean daily bars")
        return None

    frame = build_ticker_frame(symbol, inputs, horizons=())
    row = frame[FEATURE_NAMES].iloc[-1].astype(float)
    row.name = symbol
    asof = frame.index[-1]
    today = datetime.now(timezone.utc).date()
    meta = {
        "asof_date": asof.strftime("%Y-%m-%d"),
        "bars": int(n_bars),
        "stale_days": int((today - asof.date()).days),
    }
    return row, meta


def to_matrix(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    """Feature columns as a float matrix in FEATURE_NAMES order, NaN preserved.

    A Series (a live row) becomes a single-row matrix.
    """
    if isinstance(frame, pd.Series):
        return frame[FEATURE_NAMES].to_numpy(dtype=float).reshape(1, -1)
    return frame[FEATURE_NAMES].to_numpy(dtype=float)


def categorical_indices() -> list[int]:
    """Positions of CATEGORICAL_FEATURES in FEATURE_NAMES, for HistGradientBoosting."""
    return [FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES]
