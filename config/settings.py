"""
Deus — Application Settings

Loads all configuration from .env using Pydantic Settings.
Module-level singleton: `from config.settings import settings`
"""

from __future__ import annotations

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
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_model_classifier: str = "gemini-2.5-flash-lite"
    gemini_model_reddit_sentiment: str = "gemini-2.5-flash-lite"
    deepseek_model_classifier: str = "deepseek-v4-flash"
    deepseek_model_reddit_sentiment: str = "deepseek-v4-flash"
    deepseek_model_reasoner: str = "deepseek-v4-pro"
    gemini_model_ranker: str = "gemini-3.1-flash-lite"
    gemini_model_chat: str = "gemini-3-flash-preview"
    gemini_model_router: str = "gemini-2.5-flash-lite"
    gemini_model_chat_shallow: str = "gemini-3.1-flash-lite"
    gemini_model_chat_complex: str = "gemini-3-flash-preview"

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

    # ── Web Search (optional — enables real-time news for agent debates) ──
    tavily_api_key: str = ""
    web_search_provider: str = "tavily"
    web_search_max_results: int = 5

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
    briefing_hour: int = 5
    briefing_minute: int = 0
    digest_day: str = "sun"
    digest_hour: int = 20

    # ── Classification ───────────────────────────────────────────────────
    # Articles per classify_batch call. The ~1.1k-token guidance preamble is
    # sent once per call, so this is the amortisation factor. Raising it saves
    # more input tokens but widens the blast radius of one malformed response.
    classify_batch_size: int = 10

    # Output budget per article in a batch, multiplied by the batch size. A
    # classification object is ~150 tokens; the headroom is because JSON mode
    # truncates into unparseable output rather than degrading gracefully.
    classify_max_output_tokens_per_article: int = 400

    # Ranking returns only {"id", "importance_score"} per article.
    rank_max_output_tokens_per_article: int = 80

    # Single-object JSON extraction (IPO details, upcoming events).
    extraction_max_output_tokens: int = 500

    # Forward window for the Finnhub earnings calendar, in days. The calendar
    # page navigates by month, so this needs to cover more than the next cycle.
    event_scan_days_ahead: int = 90

    # ── Multi-agent debate ───────────────────────────────────────────────
    # Per-turn output budget for the Bull/Bear researchers. Measured average is
    # ~830 completion tokens; this leaves room for a longer turn without
    # letting a runaway response go unbounded on the priciest model in use.
    # A truncated debate turn is user-visible, so keep this generous.
    debate_max_output_tokens: int = 2000

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

    # ── Geo tagging ──────────────────────────────────────────────────────
    geo_backfill_batch: int = 500

    # ── System ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    db_path: str = "storage/deus.db"
    timezone: str = "Asia/Seoul"
    sqlite_vec_path: str = ""

    # ── API Server ───────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Derived helpers (not from env) ───────────────────────────────────

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
