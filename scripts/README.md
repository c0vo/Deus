# Scripts

## `manual/` — Manual Validation Scripts

These scripts are NOT automated tests. They hit real APIs (DeepSeek, Gemini, Telegram, Reddit, yfinance) and require API keys to be configured. Use them for ad-hoc debugging and manual validation.

- `test_news_sources.py` — Test Finnhub and Alpha Vantage API sources
- `test_send_alert.py` — Send a test Telegram alert
- `test_telegram_bot.py` — Verify Telegram bot connection
- `test_reddit_pipeline.py` — Full Reddit fetch → classify pipeline
- `test_reddit_fetch.py` — Test Reddit API fetching
- `test_embed_api.py` — Test Gemini embedding API
- `test_live_prefilter.py` — Test prefilter against live Reddit data
- `test_ws_chat.py` — Test WebSocket chat endpoint
- `verify_api.py` — Full API surface integration test
- `test_pipeline_cycle.py` — Run one full pipeline cycle

## `manual/` — Direction model (pooled predictor, feature schema v4)

No API keys needed; all but `null_ship_rate.py` read and write the local database.

- `train_pooled_models.py` — Train every horizon's pooled model once (what the Sunday retrain does) and print the skill table. Use after deploying schema v4 or any schema bump.
- `train_experiments.py` — Desktop experiment runner against a database copy: `run`, `sweep`, `control`, `baselines`, `coverage`, `leaderboard`. Writes `storage/experiments/` and `LEADERBOARD.md`.
- `null_ship_rate.py` — Null calibration and power of the ship rule on simulated panels, through the production training path (no database). Writes `storage/experiments/null_ship_rate.md`; `--smoke` or `--quick` for a short run.
- `regrade_predictions.py` — Re-grade resolved predictions on stored sessions; dry run unless `--apply`.
- `backfill_price_history.py` — Deep daily OHLCV (and `--splits`) for the model universes.
- `clean_price_history.py` — Find and repair non-daily rows in `price_history`.

## Safe Test Runners

- `run_safe_tests.ps1` — PowerShell script to run only safe tests (Windows)
- `run_safe_tests.sh` — Bash script to run only safe tests (Linux/Termux)
