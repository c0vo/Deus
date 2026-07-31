# E2E Test Infra: Deus

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + BVA (Boundary Value Analysis) + Pairwise Combinatorial + Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 (Happy) | Tier 2 (Boundary) | Tier 3 (Cross) | Tier 4 (Workload) |
|---|---------|----------------------|:--------------:|:-----------------:|:--------------:|:-----------------:|
| 1 | Watchlist CRUD | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 2 | Live Market Grid | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 3 | SSE Predict Stream | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 4 | SSE Chat Stream | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 5 | News Briefings | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 6 | Charts/Indicators | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 7 | Backtesting Simulator | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 8 | Accuracy & Reflections | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |
| 9 | Token & Status | ORIGINAL_REQUEST R1 | 5 | 5 | Yes | Yes |

## Test Architecture
- **Test Runner**: pytest with `pytest-asyncio` for asynchronous execution. Run using `python -m pytest tests/e2e`.
- **HTTP Client**: Uses `httpx.AsyncClient`. By default, tests use `httpx.ASGITransport(app)` wrapping the mock backend. If the environment variable `E2E_SERVER_URL` is set (e.g. `http://localhost:8000`), tests target the real running backend server.
- **Directory Layout**:
  ```
  tests/
  └── e2e/
      ├── conftest.py
      ├── mock_backend.py
      ├── test_tier1_feature_coverage.py
      ├── test_tier2_boundary_corner.py
      ├── test_tier3_cross_feature.py
      └── test_tier4_real_world.py
  ```

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | The Researcher's Workflow | Watchlist, Markets, News, Charts, Backtest | Medium |
| 2 | The Analyst's Prediction | Watchlist, Predict SSE, Chat SSE | Medium |
| 3 | Performance Review | Accuracy, Reflections, Status/Usage | Medium |
| 4 | Asset Rotation | Watchlist, Markets, Predict SSE, Backtest | High |
| 5 | System Diagnostics | Predict SSE, Accuracy, Status/Usage | Medium |

## Coverage Thresholds
- Tier 1: 5 * 9 = 45 happy path cases
- Tier 2: 5 * 9 = 45 boundary/corner cases
- Tier 3: 9 cross-feature cases
- Tier 4: 5 real-world workloads
- **Total: 104 tests**
