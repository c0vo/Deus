"""Tests for emergent theme detection.

These lock in behaviour that was established by measuring the real corpus, not
by intuition — every constant here has a counterexample behind it. See the
module docstring in pipeline/theme_detector.py.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.theme_detector import (
    ThemeDetector,
    _fingerprint,
    share_of_voice_accel,
)

NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)
DIM = 64


def make_rows(groups, recent_days=3, base_days=40, importance=6.0, sources=None,
              seed=0, id_prefix="a"):
    """Synthetic articles clustered around `groups` random centres.

    Each group is (n_recent, n_base). Vectors are a group centre plus small
    noise, then shifted by a large shared offset — which is what real Gemini
    embeddings look like: an anisotropic cloud with a high similarity floor.

    Note every fixture here uses at least three groups. Mean-centering
    subtracts the corpus centroid, so a corpus containing exactly one topic has
    that topic *as* its mean and centering flattens it into noise. Real corpora
    are not like this — the live database yields ~400 clusters — but a
    single-group fixture is degenerate and will not survive the filters.
    """
    rng = np.random.default_rng(seed)
    shared_offset = rng.normal(size=DIM) * 5.0  # the anisotropy
    rows, idx = [], 0
    for gi, (n_recent, n_base) in enumerate(groups):
        centre = rng.normal(size=DIM)
        for is_recent, count in ((True, n_recent), (False, n_base)):
            for _ in range(count):
                vec = centre + rng.normal(size=DIM) * 0.05 + shared_offset
                vec = vec.astype(np.float32)
                age = recent_days if is_recent else base_days
                published = (NOW - timedelta(days=age)).isoformat()
                src = (sources or ["cnbc", "reuters", "wsj"])[idx % 3]
                rows.append({
                    "id": f"{id_prefix}{idx}", "headline": f"group {gi} article {idx}",
                    "published_at": published, "importance_score": importance,
                    "source_type": "rss", "source_name": src,
                    "affected_tickers": "[]", "affected_sectors": "[]",
                    "embedding": vec.tobytes(),
                })
                idx += 1
    return rows


def detector_with(rows, total_recent=500, total_base=2000):
    db = MagicMock()
    db.get_embeddings_since.return_value = rows
    db.count_articles_in_window.side_effect = [total_recent, total_base]
    return ThemeDetector(db)


# ── Share-of-voice acceleration ──────────────────────────────────────────


def test_acceleration_is_scale_invariant_for_large_counts():
    """The burstiness fix.

    Weekly ingest volume in the real corpus runs 855, 339, 1, 522, 32, 0, 644.
    A raw count ratio would mostly report whether the worker was running, so
    the same *shares* over a 10x larger corpus must give the same answer.
    """
    a, _ = share_of_voice_accel(1000, 10_000, 2000, 40_000, min_corpus=10)
    assert a == pytest.approx(2.0, abs=0.01)


def test_acceleration_is_none_below_the_corpus_floor():
    """None ('not computable') is deliberately distinct from 0.0 ('flat')."""
    value, basis = share_of_voice_accel(5, 50, 1, 40, min_corpus=10_000)
    assert value is None
    assert basis == "thin"


def test_acceleration_detects_a_riser_and_a_fader():
    rising, _ = share_of_voice_accel(20, 200, 5, 800, min_corpus=10)
    fading, _ = share_of_voice_accel(1, 200, 60, 800, min_corpus=10)
    assert rising > 1.0
    assert fading < 1.0


def test_tiny_counts_are_damped_not_infinite():
    value, _ = share_of_voice_accel(2, 100, 0, 400, min_corpus=10)
    assert value is not None and value < 100


# ── Clustering ───────────────────────────────────────────────────────────


def test_centering_prevents_cluster_collapse():
    """The correction that made clustering work at all.

    Raw Gemini embeddings on the real corpus have a cosine floor near 0.52
    (mean 0.519, p90 0.583), so a threshold under ~0.6 groups random pairs.
    Centering moves random pairs to ~0.00 and makes the threshold meaningful.
    """
    rows = make_rows([(5, 5), (5, 5), (5, 5)])
    vectors = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    raw = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    centered = ThemeDetector._center_and_normalize(vectors)

    def offdiag_mean(m):
        sim = m @ m.T
        iu = np.triu_indices_from(sim, k=1)
        return float(sim[iu].mean())

    assert offdiag_mean(raw) > 0.5      # anisotropic: everything looks similar
    assert abs(offdiag_mean(centered)) < 0.2   # centred: only real topics match


def test_fixed_seed_centroid_recovers_distinct_groups():
    """Three synthetic topics must yield three clusters, not one blob.

    Assigning against a fixed seed rather than a running centroid mean is what
    prevents the centroid drifting toward the corpus mean and absorbing
    everything.
    """
    rows = make_rows([(6, 6), (6, 6), (6, 6)])
    vectors = ThemeDetector._center_and_normalize(
        np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    )
    order = np.arange(len(rows))
    labels = ThemeDetector._greedy_assign(vectors, order, threshold=0.30)
    assert len(set(labels.tolist())) == 3


def test_cluster_requires_multiple_sources():
    """One chatty feed must not be able to manufacture a theme.

    This is the filter that removes Alpha Vantage's 13F churn, which clusters
    tightly because it is formulaic and always looks like it is accelerating.
    """
    rows = make_rows([(6, 6), (6, 6), (6, 6)], sources=["alpha_vantage"] * 3)
    seeds = detector_with(rows).cluster_recent(as_of=NOW)
    assert seeds == []


def test_cluster_requires_minimum_importance():
    rows = make_rows([(6, 6), (6, 6), (6, 6)], importance=1.0)  # noise band
    seeds = detector_with(rows).cluster_recent(as_of=NOW)
    assert seeds == []


def test_valid_cluster_survives_and_carries_counts():
    rows = make_rows([(6, 6), (6, 6), (6, 6)])
    seeds = detector_with(rows).cluster_recent(as_of=NOW)
    assert len(seeds) == 3
    seed = seeds[0]
    assert seed.count_recent == 6
    assert seed.count_base == 6
    assert seed.source_count >= 3
    assert seed.acceleration is not None


def test_riser_outranks_a_larger_flat_cluster():
    """The whole point: emerging beats loud.

    A small cluster rising from near-silence must outrank a bigger one whose
    volume is mostly historic — the latter is the story that is already priced.
    """
    rows = (
        make_rows([(8, 0)], seed=1, id_prefix="r")          # the riser
        + make_rows([(2, 30)], seed=2, id_prefix="f")       # the faded story
        + make_rows([(4, 4)], seed=3, id_prefix="n")        # a neutral third topic
    )
    seeds = detector_with(rows).cluster_recent(as_of=NOW)
    assert len(seeds) >= 2
    top = seeds[0]
    assert top.count_recent > top.count_base
    # The faded cluster is larger but must not win.
    faded = [s for s in seeds if s.count_base >= 20]
    assert faded and seeds.index(faded[0]) > 0


def test_empty_corpus_returns_no_seeds():
    assert detector_with([]).cluster_recent(as_of=NOW) == []


# ── Seed identity ────────────────────────────────────────────────────────


def test_fingerprint_is_stable_regardless_of_order():
    assert _fingerprint(["b", "a", "c"]) == _fingerprint(["c", "b", "a"])


def test_fingerprint_differs_between_topics():
    assert _fingerprint(["a", "b"]) != _fingerprint(["c", "d"])


def test_dedupe_drops_overlapping_clusters():
    from pipeline.theme_detector import ThemeSeed

    a = ThemeSeed("auto", "f1", "A", "", article_ids=["1", "2", "3", "4"])
    b = ThemeSeed("auto", "f2", "B", "", article_ids=["1", "2", "3", "5"])
    c = ThemeSeed("auto", "f3", "C", "", article_ids=["9", "8", "7", "6"])
    kept = ThemeDetector._dedupe([a, b, c])
    assert [s.title for s in kept] == ["A", "C"]


def test_user_seed_is_marked_and_has_no_acceleration():
    """A hand-typed topic has no cluster behind it, so acceleration is unknown
    rather than zero."""
    seed = ThemeDetector(MagicMock()).seed_from_text("China restricts gallium exports")
    assert seed.seed_kind == "user"
    assert seed.acceleration is None
    assert seed.acceleration_basis == "user"
    assert "gallium" in seed.title


def test_thesis_row_shape_matches_insert_contract():
    rows = make_rows([(6, 6), (6, 6), (6, 6)])
    seed = detector_with(rows).cluster_recent(as_of=NOW)[0]
    row = seed.to_thesis_row()
    for key in ("title", "seed_kind", "seed_fingerprint", "acceleration",
                "acceleration_basis", "article_count_recent", "evidence"):
        assert key in row
