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
- `test_predictor_e2e.py` — End-to-end predictor validation with real data

## Safe Test Runners

- `run_safe_tests.ps1` — PowerShell script to run only safe tests (Windows)
- `run_safe_tests.sh` — Bash script to run only safe tests (Linux/Termux)
