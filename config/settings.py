"""
Project Scrooge V2 — Application Settings

Loads all configuration from .env using Pydantic Settings.
Module-level singleton: `from config.settings import settings`
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ── Web Search (optional — enables real-time news for agent debates) ──
    tavily_api_key: str = ""
    web_search_provider: str = "tavily"
    web_search_max_results: int = 5

    # ── Nitter / X-Twitter ───────────────────────────────────────────────
    nitter_instances: str = "nitter.net,nitter.privacydev.net,nitter.poast.org"
    nitter_accounts: str = "DeItaone,realDonaldTrump"

    # ── Reddit ───────────────────────────────────────────────────────────
    reddit_subreddits: str = "wallstreetbets,stocks,investing,smallstreetbets"

    # ── Schedule ─────────────────────────────────────────────────────────
    fetch_interval_hours: int = 2
    briefing_hour: int = 5
    briefing_minute: int = 0
    digest_day: str = "sun"
    digest_hour: int = 20

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

    # ── Geo tagging ──────────────────────────────────────────────────────
    geo_backfill_batch: int = 500

    # ── System ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    db_path: str = "storage/scrooge.db"
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

    def has_key(self, key_name: str) -> bool:
        """Check if a specific API key is configured (non-empty)."""
        return bool(getattr(self, key_name, ""))


# Module-level singleton — import this everywhere
settings = Settings()
