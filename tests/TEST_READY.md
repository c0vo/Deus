# E2E Test Suite Ready

## Test Runner
- Command: `python -m pytest tests/e2e` (to run using mock backend)
- Real server command: `E2E_SERVER_URL=http://localhost:8000 python -m pytest tests/e2e` (to run against real backend)
- Expected: all tests pass with exit code 0

## Coverage Summary
| Tier | Count | Description |
|------|------:|-------------|
| 1. Feature Coverage | 45 | Happy-path coverage (5 per feature) |
| 2. Boundary & Corner | 45 | Edge case & validation coverage (5 per feature) |
| 3. Cross-Feature | 9 | Concurrent stream & state sync combinations |
| 4. Real-World Application | 5 | High-fidelity workflow scenarios |
| **Total** | **104** | |

## Feature Checklist
| Feature | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------|:------:|:------:|:------:|:------:|
| Watchlist CRUD | 5 | 5 | ✓ | ✓ |
| Live Market Grid | 5 | 5 | ✓ | ✓ |
| SSE Predict Stream | 5 | 5 | ✓ | ✓ |
| SSE Chat Stream | 5 | 5 | ✓ | ✓ |
| News Briefings | 5 | 5 | ✓ | ✓ |
| Charts/Indicators | 5 | 5 | ✓ | ✓ |
| Backtesting Simulator | 5 | 5 | ✓ | ✓ |
| Accuracy & Reflections | 5 | 5 | ✓ | ✓ |
| Token & Status | 5 | 5 | ✓ | ✓ |
