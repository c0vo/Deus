"""
Deus — Emergent Theme Detection

Finds what is *starting* to be talked about, which is a different question from
what is being talked about most.

TrendForecaster.generate_macro_themes() ranks the top-15 articles by importance
score. That selects, by construction, for the loudest and most-priced-in
stories: when AI capex is the top story it returns NVDA. Useful for scenarios
on names you already hold, useless for finding a trade before it is one.

This module instead clusters recent article embeddings and ranks clusters by
*acceleration* — volume in a recent window against a longer baseline — so a
theme rising from near-silence outranks one already at peak volume.

Two things here are load-bearing and were both established by measurement
against this corpus rather than chosen by intuition:

1. Clustering runs on MEAN-CENTERED vectors. Raw Gemini embeddings on this
   corpus have a cosine floor near 0.52 (mean 0.519, p50 0.512, p90 0.583), so
   any raw threshold under ~0.6 is below the 90th percentile of *random* pairs
   and groups everything. After subtracting the corpus mean and renormalising,
   random pairs sit at ~0.00 (p90 0.095) and the threshold becomes meaningful.

2. Cluster membership is decided against a FIXED seed vector, never a running
   centroid mean. Averaging drags each centroid toward the corpus mean, which
   then matches everything — the classic collapse, and it happens even at
   thresholds that look conservative.

Acceleration is share-of-voice normalised for a third measured reason: ingest
here is bursty (weekly article volume over the last two months ran 855, 339, 1,
522, 32, 0, 644). A raw count ratio between two windows would mostly report
whether the worker process happened to be running.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from config.llm import complete, is_llm_configured, parse_structured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.database import Database
from data.models import ThemeNote

log = get_logger(__name__)

# Laplace-style smoothing on the share-of-voice ratio. Keeps a cluster that
# went from 0 to 2 articles from posting an infinite acceleration.
_SOV_SMOOTHING = 0.5


@dataclass
class ThemeSeed:
    """A candidate theme, before any expensive reasoning is spent on it."""

    seed_kind: str  # 'auto' | 'user'
    fingerprint: str
    title: str
    summary: str
    article_ids: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    # Tickers the seed articles already name. Handed to the decomposition step
    # as the explicitly priced-in set it must reason *past*, not toward.
    consensus_tickers: list[str] = field(default_factory=list)
    acceleration: Optional[float] = None
    acceleration_basis: str = "sov"  # 'sov' | 'thin' | 'user'
    count_recent: int = 0
    count_base: int = 0
    mean_importance: float = 0.0
    source_count: int = 0

    def to_thesis_row(self) -> dict:
        """Shape expected by Database.insert_thesis."""
        return {
            "title": self.title,
            "summary": self.summary,
            "seed_kind": self.seed_kind,
            "seed_fingerprint": self.fingerprint,
            "consensus_tickers": self.consensus_tickers,
            "acceleration": self.acceleration,
            "acceleration_basis": self.acceleration_basis,
            "article_count_recent": self.count_recent,
            "article_count_base": self.count_base,
            "evidence": [
                {"article_id": aid, "headline": h}
                for aid, h in zip(self.article_ids[:12], self.headlines[:12])
            ],
        }


def share_of_voice_accel(
    n_recent: int,
    total_recent: int,
    n_base: int,
    total_base: int,
    min_corpus: int = 0,
) -> tuple[Optional[float], str]:
    """Acceleration as a ratio of shares, not of counts.

    Returns (value, basis). A value of None with basis 'thin' means the corpus
    was too small in the baseline window to say anything — deliberately
    distinct from 0.0, which would mean "measured, and flat".
    """
    if total_base < min_corpus or total_recent <= 0 or total_base <= 0:
        return None, "thin"
    share_recent = (n_recent + _SOV_SMOOTHING) / total_recent
    share_base = (n_base + _SOV_SMOOTHING) / total_base
    if share_base <= 0:
        return None, "thin"
    return share_recent / share_base, "sov"


def _fingerprint(article_ids: list[str]) -> str:
    """Stable id for a cluster across runs.

    Centroids are not comparable between runs because the space is
    mean-centered and the window moves, so identity is carried by the member
    articles instead: the top few ids, sorted, hashed.
    """
    top = sorted(article_ids)[:5]
    return hashlib.sha1("|".join(top).encode("utf-8")).hexdigest()[:16]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ThemeDetector:
    """Clusters recent articles and ranks the clusters by acceleration."""

    def __init__(self, db: Database):
        self.db = db

    # ── Clustering ───────────────────────────────────────────────────────

    def cluster_recent(self, as_of: Optional[datetime] = None) -> list[ThemeSeed]:
        """Pure numpy, zero LLM cost. Safe to call from a worker thread."""
        now = as_of or datetime.now(timezone.utc)
        base_days = settings.thesis_baseline_window_days
        recent_days = settings.thesis_recent_window_days
        base_cutoff = (now - timedelta(days=base_days)).isoformat()
        recent_cutoff = now - timedelta(days=recent_days)

        rows = self.db.get_embeddings_since(
            base_cutoff,
            exclude_social=True,
            limit=settings.thesis_cluster_max_vectors,
        )
        if len(rows) < settings.thesis_cluster_min_size:
            log.info("theme_detector.corpus_too_small", articles=len(rows))
            return []

        vectors = np.stack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        ).astype(np.float32)
        vectors = self._center_and_normalize(vectors)

        # Seed order matters: the most important article in a topic should be
        # the one that defines it, not whichever happened to be fetched first.
        importance = np.array([r["importance_score"] or 0.0 for r in rows], dtype=np.float32)
        order = np.argsort(-importance)
        labels = self._greedy_assign(vectors, order, settings.thesis_cluster_similarity)

        total_recent = self.db.count_articles_in_window(
            recent_cutoff.isoformat(), now.isoformat()
        )
        total_base = self.db.count_articles_in_window(base_cutoff, recent_cutoff.isoformat())

        seeds: list[ThemeSeed] = []
        for cluster_id in range(labels.max() + 1 if len(labels) else 0):
            members = np.where(labels == cluster_id)[0]
            if len(members) < settings.thesis_cluster_min_size:
                continue
            seed = self._build_seed(
                [rows[i] for i in members], recent_cutoff, total_recent, total_base
            )
            if seed is not None:
                seeds.append(seed)

        seeds.sort(key=self._rank_key, reverse=True)
        log.info(
            "theme_detector.clustered",
            articles=len(rows),
            clusters=len(seeds),
            total_recent=total_recent,
            total_base=total_base,
        )
        return seeds

    @staticmethod
    def _center_and_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise, subtract the corpus mean, re-normalise.

        The centering step is what makes the similarity threshold mean
        anything — see the module docstring.
        """
        vectors = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9
        )
        vectors = vectors - vectors.mean(axis=0, keepdims=True)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    @staticmethod
    def _greedy_assign(
        vectors: np.ndarray, order: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Assign each vector to the first seed it is close enough to.

        Similarity is measured against the seed vector itself and the seed is
        never updated. A running centroid mean drifts toward the corpus mean
        and then absorbs everything.
        """
        labels = -np.ones(len(vectors), dtype=int)
        seed_rows: list[int] = []
        for idx in order:
            if labels[idx] >= 0:
                continue
            if seed_rows:
                sims = vectors[idx] @ vectors[np.array(seed_rows)].T
                best = int(np.argmax(sims))
                if sims[best] >= threshold:
                    labels[idx] = best
                    continue
            labels[idx] = len(seed_rows)
            seed_rows.append(int(idx))
        return labels

    def _build_seed(
        self,
        members: list[dict],
        recent_cutoff: datetime,
        total_recent: int,
        total_base: int,
    ) -> Optional[ThemeSeed]:
        """Turn one cluster into a ThemeSeed, or drop it."""
        recent = [m for m in members if self._published(m) >= recent_cutoff]
        n_recent, n_base = len(recent), len(members) - len(recent)

        accel, basis = share_of_voice_accel(
            n_recent, total_recent, n_base, total_base,
            min_corpus=settings.thesis_min_corpus_articles,
        )

        ranked = sorted(members, key=lambda m: -(m["importance_score"] or 0.0))
        tickers, sectors = self._collect_tags(ranked)
        mean_importance = float(
            np.mean([m["importance_score"] or 0.0 for m in ranked])
        )

        # A single chatty feed must not be able to manufacture a theme. This is
        # the filter that removes Alpha Vantage's 13F churn ("X Purchases 48,973
        # Shares of Y"), which clusters very tightly because it is formulaic,
        # arrives in bursts, and would otherwise top the acceleration ranking.
        distinct_sources = len({self._source_of(m) for m in ranked})
        if distinct_sources < settings.thesis_cluster_min_sources:
            return None

        # Second net, on content rather than provenance. Per the project's own
        # importance calibration, 0-2 is noise and 3-4 is minor single-name news;
        # a theme worth spending a reasoning call on should clear both.
        if mean_importance < settings.thesis_cluster_min_importance:
            return None

        return ThemeSeed(
            seed_kind="auto",
            fingerprint=_fingerprint([m["id"] for m in ranked]),
            title="",  # filled in by name_clusters()
            summary="",
            article_ids=[m["id"] for m in ranked],
            headlines=[m["headline"] for m in ranked],
            sectors=sectors,
            consensus_tickers=tickers,
            acceleration=accel,
            acceleration_basis=basis,
            count_recent=n_recent,
            count_base=n_base,
            mean_importance=mean_importance,
            source_count=distinct_sources,
        )

    @staticmethod
    def _source_of(row: dict) -> str:
        """The feed an article came from.

        Reads source_name rather than parsing the id prefix: id formats differ
        per adapter (rss_cnbc_<hash> but av_<hash>), so prefix-splitting
        returned the whole id for some sources and silently disabled the
        distinct-source floor.
        """
        return (row.get("source_name") or row.get("source_type") or "").strip()

    @staticmethod
    def _published(row: dict) -> datetime:
        try:
            dt = datetime.fromisoformat(row["published_at"])
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @staticmethod
    def _collect_tags(rows: list[dict]) -> tuple[list[str], list[str]]:
        """Union the classifier's ticker and sector tags across the cluster."""
        tickers: list[str] = []
        sectors: list[str] = []
        for r in rows:
            for column, sink in (("affected_tickers", tickers), ("affected_sectors", sectors)):
                try:
                    for v in json.loads(r.get(column) or "[]"):
                        if v and v not in sink:
                            sink.append(v)
                except (json.JSONDecodeError, TypeError):
                    continue
        return tickers[:12], sectors[:8]

    @staticmethod
    def _rank_key(seed: ThemeSeed) -> tuple:
        """Accelerating first, then loud, then important.

        Seeds with unknown acceleration sort below measured ones rather than
        being treated as zero — an unmeasurable theme is not a flat theme.
        """
        return (
            1 if seed.acceleration is not None else 0,
            (seed.acceleration or 0.0) * float(np.log1p(seed.count_recent)),
            seed.mean_importance,
        )

    # ── Naming ───────────────────────────────────────────────────────────

    async def name_clusters(self, seeds: list[ThemeSeed]) -> list[ThemeSeed]:
        """One batched Gemini call names every cluster. Never raises."""
        if not is_llm_configured() or not settings.model_thesis_extract or not seeds:
            return seeds

        prompt = self._build_naming_prompt(seeds)

        try:
            with track_llm(self.db, settings.model_thesis_extract, "thesis_naming") as u:
                u.response = response = await complete(
                    model=settings.model_thesis_extract,
                    prompt=prompt,
                    schema=list[ThemeNote],
                    reasoning="low",
                )
            notes = response.parsed
            if not isinstance(notes, list):
                notes = parse_structured(response.text, list[ThemeNote])
            named = {}
            for n in notes:
                cid = (n.get("cluster_id") if isinstance(n, dict) else getattr(n, "cluster_id", "")) or ""
                title = (n.get("title") if isinstance(n, dict) else getattr(n, "title", "")) or ""
                summary = (n.get("summary") if isinstance(n, dict) else getattr(n, "summary", "")) or ""
                named[cid.strip()] = (title.strip(), summary.strip())
        except Exception as e:
            log.warning("theme_detector.naming_failed", error=str(e))
            named = {}

        for i, seed in enumerate(seeds):
            title, summary = named.get(str(i), ("", ""))
            # Falling back to the top headline keeps an unnamed cluster usable
            # rather than discarding a perfectly good seed over a failed call.
            seed.title = title or (seed.headlines[0][:80] if seed.headlines else "Untitled theme")
            seed.summary = summary
        return seeds

    @staticmethod
    def _build_naming_prompt(seeds: list[ThemeSeed]) -> str:
        parts = [THEME_NAMING_PROMPT_HEADER]
        for i, seed in enumerate(seeds):
            parts.append(f"Cluster {i}:")
            parts.extend(f"- {h}" for h in seed.headlines[:6])
            parts.append("")
        return "\n".join(parts)

    # ── Entry point ──────────────────────────────────────────────────────

    async def collect(
        self, limit: int = 5, as_of: Optional[datetime] = None
    ) -> list[ThemeSeed]:
        """Detect, name and rank candidate themes."""
        seeds = await asyncio.to_thread(self.cluster_recent, as_of)
        if not seeds:
            return []

        # Only spend the naming call on what could plausibly be used.
        seeds = self._dedupe(seeds)[: max(limit * 2, limit)]
        seeds = await self.name_clusters(seeds)
        return seeds[:limit]

    @staticmethod
    def _dedupe(seeds: list[ThemeSeed], threshold: float = 0.3) -> list[ThemeSeed]:
        """Drop clusters that substantially overlap a higher-ranked one."""
        kept: list[ThemeSeed] = []
        seen: list[set] = []
        for seed in seeds:
            ids = set(seed.article_ids)
            if any(_jaccard(ids, prior) >= threshold for prior in seen):
                continue
            kept.append(seed)
            seen.append(ids)
        return kept

    def seed_from_text(self, text: str) -> ThemeSeed:
        """Build a seed from a user-supplied topic.

        A first-class entry point, not a fallback: this corpus ingests ~20
        articles a day, which is often too thin to surface a specific theme by
        clustering alone.
        """
        cleaned = (text or "").strip()
        return ThemeSeed(
            seed_kind="user",
            fingerprint=_fingerprint([cleaned.lower()]),
            title=cleaned[:80],
            summary="",
            acceleration=None,
            acceleration_basis="user",
        )


THEME_NAMING_PROMPT_HEADER = (
    "You are a financial news editor. Below are clusters of related headlines.\n"
    "For EACH cluster, return the cluster_id exactly as given, plus:\n"
    "- title: a short, specific theme name (under 60 characters). Name the "
    "underlying development, not the loudest headline. Prefer "
    '"Grid interconnect queues delay datacenter builds" over "Big Tech news".\n'
    "- summary: two plain-text sentences on what is developing and why it "
    "matters commercially.\n"
    "Do NOT use markdown, HTML or emojis. Return one entry per cluster.\n"
)
