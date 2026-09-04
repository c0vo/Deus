"""
Deus — Crowding / "rumour stage" scoring

Answers the second half of the buy-the-rumour question. The thesis engine finds
companies positioned at a bottleneck; this decides whether the market has
already worked that out.

Two independent axes, deliberately not collapsed into one:

  conviction — how load-bearing is this company to the chain? Driven by the
               confidence of the causal node above it, the share of its revenue
               actually exposed to the bottleneck, and how substitutable it is.
  crowding   — how priced-in is it? Driven by price run-up, distance from the
               52-week high, volume, and news attention.

  edge_score = conviction * (1 - crowding)

High conviction and still quiet. A name at 0.9 conviction and 0.85 crowding is
the one to sell, and it sorts to the bottom on its own.

The hard part is missing data, and it is the normal case rather than the
exception: a freshly discovered third-hop supplier has no mention history
(ticker_mentions holds 486 tickers, 270 of them with a single row) and no price
history (price_history covers the watchlist). "Nobody is talking about it" is
*legitimately* the EARLY signal, so absence cannot be treated as an error — but
it must not be allowed to manufacture a confident score either.

The resolution is to renormalise over observed components and report coverage
alongside the score, rather than zero-filling absent ones. A candidate at
crowding 0.15 with coverage 0.9 and one at crowding 0.15 with coverage 0.2 are
different claims, and the caller can see which is which. With no price data at
all the stage is UNKNOWN, never EARLY.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.logging_config import get_logger
from config.settings import settings
from data.database import Database

log = get_logger(__name__)

# Weights sum to 1.0. Price signals dominate because they are dense and can be
# fetched on demand for any ticker; mention signals are sparse in this corpus
# and smart-money signals only exist for US names already being synced.
COMPONENT_WEIGHTS: dict[str, float] = {
    "price_runup_3m": 0.22,
    "dist_from_52w_high": 0.18,
    "price_runup_1m": 0.10,
    "volume_surge": 0.10,
    "mention_accel": 0.20,
    "mention_absolute": 0.10,
    "offexch_share_dev": 0.05,
    "insider_net": 0.05,
}

# Below this, the evidence is too thin to call something CROWDED.
MIN_COVERAGE_FOR_HIGH_STAGE = 0.35

STAGE_EARLY = "EARLY"
STAGE_BUILDING = "BUILDING"
STAGE_CROWDED = "CROWDED"
STAGE_POST_NEWS = "POST_NEWS"
STAGE_UNKNOWN = "UNKNOWN"


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class CrowdingResult(dict):
    """Plain dict subclass so callers can json-serialise it directly."""


def score_conviction(
    node_confidence: float,
    exposure_pct: Optional[float],
    substitutability: str,
    evidence_count: int,
) -> float:
    """How much this company matters to the chain, independent of its price.

    exposure_pct is the field that stops the engine degenerating into naming
    megacaps: a pure-play moves on the thesis, while a conglomerate dilutes the
    same tailwind below its own noise floor.
    """
    exposure = clamp01((exposure_pct or 0.0) / 100.0)
    scarcity = {
        "sole_source": 1.0,
        "duopoly": 0.75,
        "oligopoly": 0.5,
        "commoditized": 0.15,
    }.get((substitutability or "").strip().lower(), 0.4)
    evidence = clamp01(evidence_count / 3.0)
    return round(
        0.35 * clamp01(node_confidence)
        + 0.30 * exposure
        + 0.20 * scarcity
        + 0.15 * evidence,
        4,
    )


class CrowdingScorer:
    """Computes the rumour stage for one candidate. No LLM calls."""

    def __init__(self, db: Database):
        self.db = db

    # ── Components ───────────────────────────────────────────────────────

    def _price_components(self, ticker: str) -> tuple[dict, dict]:
        """Returns (components, raw) from price_history. Values or None."""
        rows = self._price_rows(ticker)
        comp: dict[str, Optional[float]] = {
            "price_runup_1m": None,
            "price_runup_3m": None,
            "dist_from_52w_high": None,
            "volume_surge": None,
        }
        raw: dict = {"price": None, "ret_1m": None, "ret_3m": None,
                     "dist_from_52w_high": None, "volume_ratio": None}
        if len(rows) < 22:
            return comp, raw

        closes = [r["close"] for r in rows if r["close"]]
        if len(closes) < 22:
            return comp, raw
        last = closes[-1]
        raw["price"] = last

        ret_1m = (last - closes[-22]) / closes[-22] if closes[-22] else None
        if ret_1m is not None:
            raw["ret_1m"] = ret_1m
            # A 25% month is a fully-noticed move.
            comp["price_runup_1m"] = clamp01(ret_1m / 0.25)

        if len(closes) >= 64:
            ret_3m = (last - closes[-64]) / closes[-64] if closes[-64] else None
            if ret_3m is not None:
                raw["ret_3m"] = ret_3m
                comp["price_runup_3m"] = clamp01(ret_3m / 0.60)

        window = closes[-252:] if len(closes) >= 252 else closes
        high, low = max(window), min(window)
        # Only meaningful once the name has actually traded a range. A stock
        # that has barely moved sits at its 52-week high by default, and
        # scoring that as "fully crowded" would read a flat, ignored ticker as
        # a finished move.
        if high and low and (high - low) / low >= 0.05:
            gap = (high - last) / high
            raw["dist_from_52w_high"] = gap
            # At the high = fully crowded; 30% below = not.
            comp["dist_from_52w_high"] = clamp01(1.0 - gap / 0.30)

        vols = [r["volume"] for r in rows if r["volume"]]
        if len(vols) >= 120:
            recent = sum(vols[-20:]) / 20.0
            base = sum(vols[-120:]) / 120.0
            if base > 0:
                ratio = recent / base
                raw["volume_ratio"] = ratio
                comp["volume_surge"] = clamp01((ratio - 1.0) / 1.5)
        return comp, raw

    def _price_rows(self, ticker: str) -> list[dict]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT date, close, volume FROM price_history "
                "WHERE ticker = ? ORDER BY date ASC",
                (ticker.upper(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def _mention_components(
        self, ticker: str, as_of: Optional[str] = None
    ) -> tuple[dict, dict]:
        """Attention level and acceleration from ticker_mentions.

        Frequently absent by design — the table is shallow and most tickers
        appear once — so both components return None rather than 0 when the
        baseline is too thin to support a ratio.
        """
        counts = self.db.get_ticker_mention_counts(
            [ticker.upper()],
            recent_days=settings.thesis_recent_window_days,
            base_days=settings.thesis_baseline_window_days,
            as_of=as_of,
        ).get(ticker.upper(), {})
        recent = counts.get("recent", 0)
        base = counts.get("base", 0)
        total_recent = counts.get("total_recent", 0)
        total_base = counts.get("total_base", 0)

        comp: dict[str, Optional[float]] = {"mention_accel": None, "mention_absolute": None}
        raw = {"mentions_recent": recent, "mentions_base": base, "mention_accel": None}

        if base >= settings.thesis_min_baseline_mentions and total_recent and total_base:
            share_recent = (recent + 0.5) / total_recent
            share_base = (base + 0.5) / total_base
            if share_base > 0:
                accel = share_recent / share_base
                raw["mention_accel"] = accel
                # 8x share growth reads as fully noticed.
                comp["mention_accel"] = clamp01(math.log2(max(accel, 1e-6)) / 3.0)

        if base or recent:
            comp["mention_absolute"] = clamp01((recent + base) / 20.0)
        return comp, raw

    def _smart_money_components(self, ticker: str) -> dict:
        """Dark-pool share deviation and insider buying.

        Both push *against* crowding: quiet institutional accumulation and
        insider purchases are what being early looks like.
        """
        comp: dict[str, Optional[float]] = {"offexch_share_dev": None, "insider_net": None}
        try:
            series = self.db.get_offexchange_series(ticker)
            shares = [
                r["total_volume"] / r["consolidated_volume"]
                for r in series
                if r.get("consolidated_volume") and r.get("total_volume")
            ]
            if len(shares) >= 25:
                recent = sum(shares[-5:]) / 5.0
                base = sum(shares[-25:]) / 25.0
                comp["offexch_share_dev"] = clamp01((recent - base) / 0.10)
        except Exception as e:
            log.debug("crowding.offexch_failed", ticker=ticker, error=str(e))

        try:
            rows = self.db.get_insider_series(ticker)
            cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
            recent = [
                r for r in rows
                if (r.get("filed_at") or "") >= cutoff and r.get("is_discretionary")
            ]
            if recent:
                buys = sum(1 for r in recent if (r.get("transaction_code") or "") == "P")
                # Inverted: heavy insider buying argues the crowd has NOT arrived.
                comp["insider_net"] = clamp01(1.0 - buys / len(recent))
        except Exception as e:
            log.debug("crowding.insider_failed", ticker=ticker, error=str(e))
        return comp

    # ── Aggregation ──────────────────────────────────────────────────────

    def score(
        self,
        ticker: str,
        prior_snapshot: Optional[dict] = None,
        as_of: Optional[str] = None,
    ) -> CrowdingResult:
        """Crowding score, coverage and rumour stage for one ticker."""
        price_comp, raw = self._price_components(ticker)
        mention_comp, mention_raw = self._mention_components(ticker, as_of=as_of)
        smart_comp = self._smart_money_components(ticker)

        components = {**price_comp, **mention_comp, **smart_comp}
        raw.update(mention_raw)

        observed = {k: v for k, v in components.items() if v is not None}
        weight_seen = sum(COMPONENT_WEIGHTS[k] for k in observed)
        weight_all = sum(COMPONENT_WEIGHTS.values())
        coverage = round(weight_seen / weight_all, 4) if weight_all else 0.0

        if not observed or weight_seen <= 0:
            crowding = None
        else:
            # Renormalise over what was actually observed. Zero-filling would
            # make an unmeasured name indistinguishable from a genuinely quiet
            # one, which is the whole distinction this feature rests on.
            crowding = round(
                sum(COMPONENT_WEIGHTS[k] * v for k, v in observed.items()) / weight_seen,
                4,
            )

        has_price = price_comp.get("price_runup_1m") is not None
        stage = self._classify(crowding, coverage, has_price, raw, prior_snapshot)

        return CrowdingResult(
            ticker=ticker.upper(),
            crowding=crowding,
            coverage=coverage,
            stage=stage,
            components=components,
            raw=raw,
        )

    @staticmethod
    def _classify(
        crowding: Optional[float],
        coverage: float,
        has_price: bool,
        raw: dict,
        prior: Optional[dict],
    ) -> str:
        """Bucket a score into a rumour stage.

        POST_NEWS is a *transition*, not a threshold: it needs a prior snapshot
        showing attention rolling over. Peak-and-decelerating is the sell
        signal; peak-and-still-climbing is not, and nothing can be POST_NEWS on
        its first ever scoring run.
        """
        if crowding is None or not has_price:
            return STAGE_UNKNOWN

        if coverage < MIN_COVERAGE_FOR_HIGH_STAGE:
            # Not enough evidence to accuse a name of being crowded.
            return STAGE_EARLY if crowding < 0.25 else STAGE_BUILDING

        if crowding >= 0.75 and prior:
            prior_accel = prior.get("mention_accel")
            curr_accel = raw.get("mention_accel")
            rolling_over = (
                prior_accel is not None
                and curr_accel is not None
                and curr_accel < prior_accel
            )
            if rolling_over and (raw.get("ret_1m") or 0.0) < 0:
                return STAGE_POST_NEWS

        if crowding < 0.25:
            return STAGE_EARLY
        if crowding < 0.50:
            return STAGE_BUILDING
        return STAGE_CROWDED


def edge_score(conviction: float, crowding: Optional[float]) -> Optional[float]:
    """Conviction discounted by how much of the move has already happened.

    None when crowding is unknown: an edge that cannot be measured should be
    absent from the ranking rather than optimistically assumed.
    """
    if crowding is None:
        return None
    return round(clamp01(conviction) * (1.0 - clamp01(crowding)), 4)
