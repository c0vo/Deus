"""
Deus — Application Settings

Loads all configuration from .env using Pydantic Settings.
Module-level singleton: `from config.settings import settings`
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default RSS feeds, as "name|url" or "name|url|max_items".
#
# Two entries were dead and returning nothing: Reuters retired its public RSS
# (feeds.reuters.com no longer resolves) and is dropped, and WSJ moved off
# feeds.a.dj.com — that host is frozen at Jan 2025 — to feeds.content.dowjones.io.
#
# Per-feed max_items is the cost dial: every fetched article is embedded before
# the noise pre-filter runs, so broad feeds are capped tighter than the ones that
# actually name tickers.
DEFAULT_RSS_FEEDS: list[str] = [
    # ── US markets ───────────────────────────────────────────────────────
    "cnbc|https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "yahoo_finance|https://finance.yahoo.com/news/rssindex",
    "wsj_markets|https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "wsj_us_business|https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness|20",
    "wsj_tech|https://feeds.content.dowjones.io/public/rss/RSSWSJD|20",
    "marketwatch|https://feeds.content.dowjones.io/public/rss/mw_topstories|20",
    "nyt_business|https://rss.nytimes.com/services/xml/rss/nyt/Business.xml|20",
    "google_news_business|https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
    # Policy — low volume, high importance (FOMC statements land here).
    "fed_press|https://www.federalreserve.gov/feeds/press_all.xml|15",
    # ── Korea (English-language outlets, so no translation step is needed) ──
    "korea_times_economy|https://feed.koreatimes.co.kr/k/economy.xml|25",
    "korea_times_business|https://feed.koreatimes.co.kr/k/business.xml|25",
]


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    # One key for every provider. Which provider serves a given function is the
    # prefix of that function's slug below ("google/…", "deepseek/…"), so a
    # function can be moved between providers here without touching any code.
    openrouter_api_key: str = ""
    # Per-request ceiling. The debate and thesis calls run at xhigh reasoning
    # effort and legitimately take minutes, so the usual 60s is far too tight;
    # the SDK's own 600s default is too loose to notice a wedged route.
    llm_timeout_seconds: float = 240.0

    # Every model is unset by default and must be chosen against OpenRouter's
    # live price list — see .env.example for the full table and the reasoning
    # tier each function runs at. An empty value disables that function and is
    # reported by name at startup (see `preflight_models`), rather than failing
    # later on the first call.
    model_classifier: str = ""
    model_classifier_fallback: str = ""
    model_reddit_sentiment: str = ""
    model_reddit_sentiment_fallback: str = ""
    model_ranker: str = ""
    model_router: str = ""
    model_chat_shallow: str = ""
    model_chat_complex: str = ""
    # Single-object JSON extraction — IPO details, calendar events, web-search
    # summaries. Split out from the classifier key it used to share: same shape
    # of work, but a fraction of the volume, so it is worth pricing separately.
    model_extract: str = ""
    # Grading matured predictions against SPY.
    model_reflection: str = ""
    # Batch note-writing. These four shared one key before and are unrelated,
    # which made the cost of any one of them impossible to read off the
    # dashboard.
    model_trending: str = ""
    model_daily_advisor: str = ""
    model_trend_outlook: str = ""
    # Grades whether the context we retrieved actually answers the question.
    # Two callers, one decision: the chat graph's "search the web?" and the
    # price alert's "do we have a real catalyst, or a generic one?". Falls back
    # to model_router when unset, since it is the same size of job as routing.
    model_grader: str = ""
    # The weekly tip digest — seasonal precedents plus the coming week's events.
    model_weekly_tip: str = ""
    # The Bull/Bear debate is configured separately from the trend forecaster,
    # which is the other consumer of a reasoning-tier model, so the two can be
    # dialled independently.
    model_debate: str = ""
    model_reasoner: str = ""
    model_trader: str = ""
    model_predictor_narrative: str = ""
    model_market_scanner: str = ""
    model_sector_analyzer: str = ""

    # The one model that is NOT free to change. Stored vectors are 3072-dim and
    # compared directly against new ones, so a model of any other width
    # silently corrupts dedup, RAG and thesis grounding. The embedder rejects
    # off-width vectors rather than storing them; swapping this deliberately
    # means re-embedding the whole corpus.
    model_embedding: str = "google/gemini-embedding-001"

    # ── Telegram ─────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── News APIs (optional) ─────────────────────────────────────────────
    finnhub_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # ── SEC EDGAR (insider trades, >5% stakes) ───────────────────────────
    # No API key exists, but EDGAR returns 403 to any request whose User-Agent
    # lacks a contact address. Format: "Name email@example.com". Empty disables
    # the insider jobs the same way an empty API key disables Finnhub.
    sec_user_agent: str = ""
    # How far back the first insider backfill reaches. Form 4 coverage is dense,
    # so this is the dominant cost of the initial sync.
    insider_backfill_days: int = 1095  # 3 years

    # ── Off-exchange (dark pool) volume — FINRA ──────────────────────────
    # No API key exists; the file is a public CDN object. The dial that matters
    # is depth, and it is deliberately small: a full backfill is ~1,300
    # sequential requests over 30-60 minutes, which would stall the orchestrator
    # if it ran on first boot. This is the cold-start window for the scheduled
    # job only — full depth is scripts/manual/backfill_darkpool.py.
    darkpool_backfill_days: int = 30

    # ── Option-chain snapshots ───────────────────────────────────────────
    # yfinance exposes no options history, so a usable panel only accrues one
    # session per day and can never be backfilled — which is why this is on by
    # default even though no feature reads the table yet. Turn it off if
    # yfinance rate-limiting becomes a problem; the cost is elapsed calendar
    # time, which is the one thing that cannot be recovered later.
    options_snapshot_enabled: bool = True

    # ── Analyst consensus & price targets ────────────────────────────────
    # Same no-history constraint as the option chain above: Yahoo exposes only
    # today's consensus, so the panel accrues one session per day and cannot be
    # backfilled. On by default for that reason. Turn it off if yfinance
    # rate-limiting bites — it shares a host with the price feed and the option
    # snapshot, and a block affects all three.
    analyst_ratings_enabled: bool = True

    # ── Technical ratings (TradingView methodology) ───────────────────────
    # Pure computation over stored price_history — no network, no API key, and
    # fully backfillable, unlike the two above. The reason it can still be
    # switched off is cost of a different kind: 26 indicators across three
    # timeframes for every tracked ticker, which is real CPU on a phone.
    technical_rating_enabled: bool = True

    # ── Web Search (optional — enables real-time news for agent debates) ──
    tavily_api_key: str = ""
    web_search_provider: str = "tavily"
    web_search_max_results: int = 5
    # Identical chat turns re-billed Tavily on every ask. Short enough that a
    # breaking story is not served stale, long enough to cover a conversation.
    web_search_cache_seconds: int = 900

    # ── Thesis Engine (causal chains, second-order beneficiaries) ────────
    # Finds an emerging theme, decomposes it into the bottlenecks it creates,
    # and names the companies at each chokepoint — then scores how priced-in
    # each one already is, because the first-order beneficiary is the news,
    # not the rumour.
    thesis_enabled: bool = True
    thesis_per_run: int = 1
    thesis_max_nodes: int = 12
    thesis_max_hops: int = 3
    thesis_max_children_per_node: int = 3
    # The main cost dial: one Tavily search per searched leaf.
    thesis_max_search_nodes: int = 6
    thesis_max_candidates_per_node: int = 4
    # Shared between the xhigh reasoning block and the answer, exactly as
    # debate_max_output_tokens is. 4000 was not enough for either half: the
    # chain alone is up to thesis_max_nodes bottlenecks each carrying a claim,
    # mechanism and falsifier, and the thinking that precedes it routinely ran
    # the budget to zero, so decompose returned empty content and the run ended
    # with nothing persisted.
    thesis_max_output_tokens: int = 24000
    thesis_expiry_days: int = 45
    # Grounding each hop against our own corpus. Free — the embeddings are
    # already paid for by ingest — so unlike the web search this runs for every
    # node at every hop rather than only the leaves.
    thesis_evidence_per_node: int = 3
    # Nearest-neighbour over ~2.5k articles returns the k closest rows however
    # unrelated they are, so a claim with no coverage would still be handed
    # three citations. Raw (uncentered) vectors, so this is comparable to
    # dedup_similarity_threshold rather than to thesis_cluster_similarity.
    #
    # Measured over 719k pairs of this corpus, unrelated articles are NOT near
    # zero — the median pair scores 0.524 and p99 is 0.732, because Gemini
    # embeddings are not centered (the same fact that makes theme_detector
    # center before thresholding). A floor of 0.45 would therefore admit 96% of
    # random pairs and filter nothing. 0.65 sits at p95 of the noise while
    # staying below the 0.76-0.79 that genuinely related articles score, which
    # leaves headroom for a claim matching an article across registers.
    thesis_evidence_min_similarity: float = 0.65
    # How far back grounding will reach. Wider than the theme windows: a
    # bottleneck can be well documented by an article older than the theme
    # that surfaced it.
    thesis_evidence_lookback_days: int = 180

    # Clustering runs on MEAN-CENTERED embeddings, so this threshold is not
    # comparable to dedup_similarity_threshold (0.70 on raw vectors).
    # Measured on 1,039 real articles from this corpus: raw Gemini embeddings
    # have a similarity floor near 0.52 (mean 0.519, p90 0.583), so a raw
    # threshold below ~0.6 groups random pairs. After centering, random pairs
    # sit at ~0.00 (p90 0.095) and 0.30 yields ~60 clusters of >=4 members
    # with a largest cluster of 20 — coherent topics rather than one blob.
    thesis_cluster_similarity: float = 0.30
    thesis_cluster_min_size: int = 4
    # A cluster must span this many distinct feeds. Without it, Alpha Vantage's
    # 13F churn ("X Purchases 48,973 Shares of Y", avg importance 2.3) tops the
    # acceleration ranking: it is formulaic, so it clusters tightly, and it
    # arrives in bursts, so it always looks like it is accelerating.
    thesis_cluster_min_sources: int = 3
    # Second net, on content. The project's importance calibration puts 0-2 at
    # noise and 3-4 at minor single-name news.
    thesis_cluster_min_importance: float = 3.5
    thesis_cluster_max_vectors: int = 3000
    thesis_recent_window_days: int = 14
    thesis_baseline_window_days: int = 90
    # Below this many articles in the baseline window, acceleration is reported
    # as unknown rather than computed. Ingest is bursty — weekly volume over
    # the last two months ran 855, 339, 1, 522, 32, 0, 644 — and a ratio taken
    # over a near-empty baseline measures worker uptime, not attention.
    thesis_min_corpus_articles: int = 150
    # ticker_mentions is only ~47 days deep, and 270 of its 486 tickers have a
    # single mention, so the mention-acceleration component is absent far more
    # often than present. That is expected; the crowding score renormalises
    # around whatever is missing rather than zero-filling it.
    thesis_min_baseline_mentions: int = 3
    thesis_rescore_max_tickers: int = 60
    thesis_ticker_cache_days: int = 30
    thesis_ticker_cache_negative_days: int = 7
    # Auto-track discovered names below this crowding score into hot_tickers,
    # so ingest starts accumulating the history the next re-score needs.
    thesis_promote_max_crowding: float = 0.30
    thesis_promote_min_conviction: float = 0.60
    # Accountability. 45 days is long enough for a second-order name to have
    # re-rated but short enough that the chain that justified it still holds;
    # SPY is the benchmark because a rising tide is not an edge.
    thesis_review_min_age_days: int = 45
    thesis_review_benchmark: str = "SPY"
    thesis_review_batch: int = 40
    # Own model keys so the thesis tier can be dialled down to flash
    # independently of the trend forecaster. The reasoner decomposes the causal
    # chain; the extractor only pulls company names out of search results, so
    # it can run several tiers cheaper.
    model_thesis_reasoner: str = ""
    model_thesis_extract: str = ""

    # ── Nitter / X-Twitter ───────────────────────────────────────────────
    nitter_instances: str = "nitter.net,nitter.privacydev.net,nitter.poast.org"
    nitter_accounts: str = "DeItaone,realDonaldTrump"

    # ── Reddit ───────────────────────────────────────────────────────────
    reddit_subreddits: str = "wallstreetbets,stocks,investing,smallstreetbets"

    # ── RSS feeds ────────────────────────────────────────────────────────
    # Entries are "name|url" or "name|url|max_items". Declared as a list so
    # pydantic-settings parses it as JSON from the env, which the other
    # comma-separated settings above cannot do safely here: feed URLs carry
    # query strings, and a ',' would split mid-URL while a '#' would be read
    # as a comment by python-dotenv.
    rss_feeds: list[str] = DEFAULT_RSS_FEEDS

    # ── Schedule ─────────────────────────────────────────────────────────
    # How often the ETL pipeline (fetch → embed → classify → rank) runs. Every
    # per-cycle LLM cost scales linearly with this, so it is the single biggest
    # spend dial in the system. Replaces the old fetch_interval_hours, which
    # nothing ever read — the interval was hardcoded at the call site instead.
    pipeline_interval_minutes: int = 15
    # How long the worker waits before its first pipeline cycle. A full
    # fetch → classify → embed → rank cycle used to fire the instant the
    # process started, saturating a phone SoC exactly while the dashboard was
    # being loaded for the first time.
    pipeline_startup_delay_seconds: int = 90
    # Same idea for the missed-daily-job catch-up, which can trigger
    # multi-agent LLM debates and a Yahoo call per unresolved prediction.
    startup_catchup_delay_seconds: int = 300
    # How often the worker refreshes quotes for /api/markets. One HTTP GET per
    # tracked ticker, no LLM cost, so this can be far more frequent than the
    # pipeline cycle.
    price_refresh_seconds: int = 60
    # When the prioritized daily brief goes out, in the configured timezone.
    briefing_hour: int = 5
    briefing_minute: int = 0
    # When the weekly digest goes out, same timezone. Sunday evening: late
    # enough that the week is unambiguously over, early enough to be read
    # before Monday's US open.
    digest_day: str = "sun"
    digest_hour: int = 20
    # Default misfire window for every scheduled job. APScheduler's own default
    # is ONE SECOND, so a job whose fire time passes while the event loop is
    # busy is dropped rather than run late — the normal case on a phone that
    # Android Doze keeps suspending. Daily and weekly jobs override this with a
    # far wider window; see DAILY_MISFIRE_GRACE_SECONDS in
    # orchestrator/scheduler.py.
    job_misfire_grace_seconds: int = 300

    # ── Price alerts (market scanner) ─────────────────────────────────────
    # Asymmetric on purpose: a tracked position falling is actionable at a
    # smaller move than anything else the scanner watches, so a tracked drop
    # alerts at 3% while every other move — either direction, tracked or not —
    # keeps the original 5% rule.
    alert_drop_pct_tracked: float = 3.0
    alert_move_pct: float = 5.0
    # Re-alert step. One alert per ticker per day, then again only once the move
    # has deepened by this much: -3% then -6%, not -3.0, -3.1, -3.2.
    alert_escalation_step_pct: float = 3.0
    # Volume alert threshold, as a multiple of the 20-session average volume.
    alert_volume_multiple: float = 3.0
    # Web results pulled when our own corpus cannot explain a tracked drop.
    # Separate from web_search_max_results because this one fires unattended.
    alert_web_search_max_results: int = 5

    # ── Seasonality & daily advisory ──────────────────────────────────────
    # Benchmarks the weekly tip computes its precedents against, and the minimum
    # depth of price history before a seasonal statistic is claimed at all — a
    # "September is historically weak" line drawn from four years is noise
    # dressed up as a precedent.
    seasonality_benchmarks: str = "SPY,QQQ"
    seasonality_min_years: int = 10
    # The daily stance note is one cheap batched call, but a material change can
    # justify re-running the full Bull/Bear debate for a ticker. These bound
    # that: how many debates a day, and how important the news has to be.
    advisor_rerun_max_per_day: int = 2
    advisor_rerun_min_importance: float = 8.0

    # ── Classification ───────────────────────────────────────────────────
    # Articles per classify_batch call. The ~1.1k-token guidance preamble is
    # sent once per call, so this is the amortisation factor. Raising it saves
    # more input tokens but widens the blast radius of one malformed response.
    classify_batch_size: int = 10

    # Output budget per article in a batch, multiplied by the batch size. A
    # classification object is ~150 tokens; the headroom is because JSON mode
    # truncates into unparseable output rather than degrading gracefully.
    classify_max_output_tokens_per_article: int = 400

    # Backlog draining. Classification used to run only inside the fetch-bound
    # pipeline cycle, which pinned its capacity to ingest with no margin and let
    # one persistently failing batch sit at the head of the queue indefinitely.
    # These bound a dedicated pass instead: how often it runs, how many rows it
    # may take per run, how many chunks may be in flight at once, how many times
    # a single row may fail before it is left alone, and how far back it reaches
    # at all — older rows are marked stale rather than paid for.
    classify_backlog_interval_minutes: int = 5
    classify_per_run_limit: int = 60
    classify_concurrency: int = 2
    classify_max_attempts: int = 3
    classify_max_age_days: int = 30

    # Ranking returns only {"id", "importance_score"} per article.
    rank_max_output_tokens_per_article: int = 80

    # Single-object JSON extraction (IPO details, upcoming events).
    extraction_max_output_tokens: int = 500

    # Forward window for the Finnhub earnings calendar, in days. The calendar
    # page navigates by month, so this needs to cover more than the next cycle.
    event_scan_days_ahead: int = 90

    # ── Multi-agent debate ───────────────────────────────────────────────
    # Per-turn completion budget for the Bull/Bear researchers. This is NOT a
    # prose-length estimate: the debate runs at `xhigh` reasoning effort with
    # thinking enabled, and DeepSeek bills the thinking block against the same
    # completion budget as the answer. At the old 2000 the reasoning routinely
    # consumed the whole allowance and the visible turn came back clipped
    # mid-word — or, on a round-2 turn whose prompt carries the full debate
    # history and so reasons longest, empty. Measured prose is ~830 tokens;
    # the rest of this is headroom for the reasoning that precedes it.
    # `pipeline/trend_forecaster.py` shares this setting, where a truncated
    # response is a hard json.loads failure rather than a clipped sentence.
    debate_max_output_tokens: int = 8000

    # Attempts for a call that failed transiently. Deterministic failures
    # (unparseable JSON at temperature 0) are never retried — see
    # config.llm.is_transient.
    llm_max_retries: int = 3

    # ── Embedding ────────────────────────────────────────────────────────
    # Texts sent per embed_content request. The API takes a list, so this is
    # request batching, not a token saving — it collapses one round-trip per
    # article into one per batch.
    embed_batch_size: int = 100

    # ── Deduplication ────────────────────────────────────────────────────
    # Candidate matches are bounded to a publish-date window: it stops an old
    # story suppressing a current one, and keeps the vector scan small.
    dedup_window_days: int = 3
    dedup_similarity_threshold: float = 0.70
    dedup_backfill_batch: int = 200

    # ── IPO tracking ─────────────────────────────────────────────────────
    # A "listed" IPO stops being watchlist material after this many days, and
    # an extracted IPO date this far in the past means we matched a story about
    # an already-public company.
    ipo_listed_retention_days: int = 30
    ipo_max_backdate_days: int = 90
    # Window for the Finnhub IPO calendar. It is mostly backward-looking, so a
    # little history is what makes the lane non-empty. Keep the look-back inside
    # ipo_max_backdate_days or retire_stale deletes what this just ingested.
    ipo_scan_days_back: int = 30
    ipo_scan_days_ahead: int = 90
    # How long after its date a listed IPO still shows on the watchlist, so
    # yesterday's listing reads as a completed event rather than vanishing.
    ipo_show_listed_days: int = 1

    # ── Geo tagging ──────────────────────────────────────────────────────
    geo_backfill_batch: int = 500

    # ── Observability ────────────────────────────────────────────────────
    # Whether llm_usage_log stores the full prompt and response of every call.
    # Off by default: the classifier and ranker log on every batch, so leaving
    # this on archives the entire prompt history and dominates database growth.
    # Failed calls keep their payload regardless — see config.usage.track_llm.
    log_llm_payloads: bool = False

    # ── System ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    # Matches DB_PATH in .env and the file that actually holds the data. The
    # old "storage/deus.db" default silently created an empty second database
    # whenever .env was not loaded.
    db_path: str = "storage/scrooge.db"
    timezone: str = "Asia/Seoul"
    sqlite_vec_path: str = ""
    # Run the ingest pipeline and Telegram bot inside the API process instead
    # of worker.py. Off by default: they share the API's event loop when on,
    # so a pipeline cycle stalls HTTP requests. Convenient for local
    # development, wrong for the Termux deployment.
    enable_in_process_worker: bool = False

    # ── API Server ───────────────────────────────────────────────────────
    # Stays 0.0.0.0 on Termux: the Android Tailscale client delivers inbound
    # packets on a tun interface, so a loopback-only bind would make the phone
    # unreachable from every other device. trusted_networks below is what
    # keeps that from also exposing the app to whatever WiFi the phone joined.
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Remote access / security ─────────────────────────────────────────
    # Empty passphrase disables the login gate entirely, which keeps local
    # development friction-free. The Termux .env is the only place it is set.
    dashboard_passphrase: str = ""
    dashboard_session_days: int = 30
    # Comma-separated CIDRs allowed to reach the app at all. Defaults cover
    # loopback plus Tailscale's IPv4 CGNAT range and IPv6 ULA prefix. An empty
    # string disables the check (any source may connect).
    trusted_networks: str = "127.0.0.0/8,::1/128,100.64.0.0/10,fd7a:115c:a1e0::/48"
    # Comma-separated CORS origins. Empty means same-origin only, which is
    # what the Termux deployment needs — FastAPI serves the static export and
    # the API from the same port, so no cross-origin request is ever legitimate.
    # Set this only to run the Next dev server against a remote backend.
    cors_origins: str = ""

    # ── Derived helpers (not from env) ───────────────────────────────────

    @property
    def trusted_network_list(self) -> list[str]:
        """Parse comma-separated CIDRs allowed to reach the API."""
        return [n.strip() for n in self.trusted_networks.split(",") if n.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins. Empty means same-origin only."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def auth_enabled(self) -> bool:
        """Whether the login gate is active. Off when no passphrase is set."""
        return bool(self.dashboard_passphrase.strip())

    @property
    def nitter_instance_list(self) -> list[str]:
        """Parse comma-separated Nitter instances into a list."""
        return [i.strip() for i in self.nitter_instances.split(",") if i.strip()]

    @property
    def nitter_account_list(self) -> list[str]:
        """Parse comma-separated Nitter accounts into a list."""
        return [a.strip() for a in self.nitter_accounts.split(",") if a.strip()]

    @property
    def reddit_subreddit_list(self) -> list[str]:
        """Parse comma-separated subreddits into a list."""
        return [s.strip() for s in self.reddit_subreddits.split(",") if s.strip()]

    @property
    def rss_feed_list(self) -> list[tuple[str, str, int]]:
        """Parse rss_feeds into (name, url, max_items) triples.

        Malformed entries are skipped rather than raising — one bad line in .env
        should not stop the whole pipeline from starting.
        """
        parsed: list[tuple[str, str, int]] = []
        for raw in self.rss_feeds:
            parts = [p.strip() for p in str(raw).split("|")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            try:
                max_items = int(parts[2]) if len(parts) > 2 and parts[2] else 30
            except ValueError:
                max_items = 30
            parsed.append((parts[0], parts[1], max_items))
        return parsed

    def has_key(self, key_name: str) -> bool:
        """Check if a specific API key is configured (non-empty)."""
        return bool(getattr(self, key_name, ""))


# Module-level singleton — import this everywhere
settings = Settings()


# ── Startup validation ───────────────────────────────────────────────────

# Retired provider-prefixed names, and what replaced each one. This map is the
# only thing standing between a stale .env and a silently mis-running app:
# `extra="ignore"` above means an obsolete GEMINI_MODEL_RANKER is dropped
# without complaint, so the app would boot clean and run on an empty default.
_LEGACY_MODEL_ENV: dict[str, str] = {
    "GEMINI_API_KEY": "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY": "OPENROUTER_API_KEY",
    "GEMINI_MODEL_CLASSIFIER": "MODEL_CLASSIFIER_FALLBACK",
    "DEEPSEEK_MODEL_CLASSIFIER": "MODEL_CLASSIFIER",
    "GEMINI_MODEL_REDDIT_SENTIMENT": "MODEL_REDDIT_SENTIMENT_FALLBACK",
    "DEEPSEEK_MODEL_REDDIT_SENTIMENT": "MODEL_REDDIT_SENTIMENT",
    "GEMINI_MODEL_RANKER": "MODEL_RANKER",
    "GEMINI_MODEL_ROUTER": "MODEL_ROUTER",
    "GEMINI_MODEL_CHAT": "MODEL_CHAT_COMPLEX",
    "GEMINI_MODEL_CHAT_SHALLOW": "MODEL_CHAT_SHALLOW",
    "GEMINI_MODEL_CHAT_COMPLEX": "MODEL_CHAT_COMPLEX",
    "DEEPSEEK_MODEL_DEBATE": "MODEL_DEBATE",
    "DEEPSEEK_MODEL_REASONER": "MODEL_REASONER",
    "DEEPSEEK_MODEL_THESIS": "MODEL_THESIS_REASONER",
    "GEMINI_MODEL_THESIS": "MODEL_THESIS_EXTRACT",
}

# Setting name → the feature that stops working when it is empty. Phrased as
# user-visible consequences, because that is what makes an unset model
# actionable at 5am when the daily brief came out short.
_MODEL_FEATURES: dict[str, str] = {
    "model_classifier": "news classification (event type, sentiment, tickers)",
    "model_classifier_fallback": "classification retry on a second model",
    "model_reddit_sentiment": "Reddit comment sentiment",
    "model_reddit_sentiment_fallback": "Reddit sentiment retry on a second model",
    "model_ranker": "importance scoring — without it nothing reaches the brief or alerts",
    "model_router": "chat shallow/complex routing",
    "model_extract": "IPO, calendar-event and web-search extraction",
    "model_reflection": "grading matured predictions into reflection_log",
    "model_trending": "trending ticker summaries",
    "model_daily_advisor": "the daily advisor batch note",
    "model_trend_outlook": "sector outlook and macro themes",
    "model_grader": "context sufficiency grading for chat and price alerts "
                    "(falls back to MODEL_ROUTER)",
    "model_weekly_tip": "weekly tip digest",
    "model_chat_shallow": "fast dashboard chat replies",
    "model_chat_complex": "reasoning-tier dashboard chat replies",
    "model_debate": "Bull/Bear debate",
    "model_reasoner": "trend forecasting",
    "model_trader": "trader synthesis at the end of a debate",
    "model_predictor_narrative": "plain-English rationale on ML predictions",
    "model_market_scanner": "price-swing and earnings-whisper alerts",
    "model_sector_analyzer": "sector rationales",
    "model_thesis_reasoner": "thesis causal-chain decomposition",
    "model_thesis_extract": "company extraction from thesis searches",
    "model_embedding": "embeddings — without it dedup, RAG and thesis grounding all stop",
}


def preflight_models() -> list[str]:
    """
    Report LLM configuration problems at startup instead of at first call.

    Returns the list of unset model settings, so a caller can decide whether an
    empty pipeline is acceptable. Called once from `main.py` and `worker.py`.
    """
    import os

    from config.logging_config import get_logger

    log = get_logger(__name__)

    # `env_file` above is relative to the working directory, so a worker
    # started from anywhere but the repo root reads no file and every setting
    # silently falls back to its empty default. That presents downstream as a
    # dozen "MODEL_X is not set" warnings for settings the .env does define,
    # so say plainly which file was read before reporting anything missing.
    env_path = Path(str(Settings.model_config.get("env_file", ".env"))).resolve()
    if env_path.exists():
        log.info("config.env_file", path=str(env_path))
    else:
        log.error(
            "config.env_file_missing",
            path=str(env_path),
            cwd=os.getcwd(),
            impact="no .env was read; every key and MODEL_* setting is empty",
            hint="start main.py / worker.py from the repo root",
        )

    if not settings.openrouter_api_key:
        log.error(
            "config.missing_api_key",
            key="OPENROUTER_API_KEY",
            impact="every LLM call will fail — including the thesis engine, "
                   "which reports this as its own model being unset",
            hint="get one at https://openrouter.ai/keys",
        )

    # Both sources, because they fail differently. pydantic-settings parses
    # .env directly without exporting anything to the process environment, so
    # checking os.environ alone misses the stale file that is the whole reason
    # this check exists.
    declared: dict[str, str] = {}
    try:
        from dotenv import dotenv_values

        declared.update({k: v for k, v in dotenv_values(".env").items() if v})
    except Exception:  # no python-dotenv, or no .env — os.environ still applies
        pass
    declared.update({k: v for k, v in os.environ.items() if v})

    for old, new in _LEGACY_MODEL_ENV.items():
        if declared.get(old):
            log.warning(
                "config.legacy_setting",
                old=old,
                new=new,
                action="ignored — this project now routes everything through OpenRouter",
            )

    unset = [name for name in _MODEL_FEATURES if not getattr(settings, name, "")]
    for name in unset:
        log.warning(
            "config.model_unset",
            setting=name.upper(),
            disables=_MODEL_FEATURES[name],
            hint="pick a slug at https://openrouter.ai/models",
        )

    if not unset:
        log.info("config.models_ready", count=len(_MODEL_FEATURES))

    return unset
