<#
.SYNOPSIS
Run only the safe, fast, automated tests that don't require API keys.
.DESCRIPTION
This script runs the subset of tests that are real pytest tests:
- Prompt and schema validation (fast, no network)
- Classifier and ranker response parsing (mocked LLMs)
- Core pipeline tests (mocked LLMs)
- Database tests (temporary SQLite)
- SSE streaming tests (string-only)
- E2E tests (against mock backend)
- Existing well-maintained tests

It EXCLUDES:
- Manual scripts in scripts/manual/
- Tests that require API keys or live services
#>

$SafeTests = @(
    "tests/test_prompts.py",
    "tests/test_schemas.py",
    "tests/test_classifier.py",
    "tests/test_ranker.py",
    "tests/test_agents.py",
    "tests/test_aggregator.py",
    "tests/test_embedder.py",
    "tests/test_predictor.py",
    "tests/test_chat_orchestrator.py",
    "tests/test_market_scanner.py",
    "tests/test_sqlite_vec.py",
    "tests/test_sse_streaming.py",
    "tests/test_database.py",
    "tests/test_frontend.py",
    "tests/e2e/"
)

python -m pytest @SafeTests -v --tb=short

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ All safe tests passed!" -ForegroundColor Green
} else {
    Write-Host "`n❌ Some tests failed." -ForegroundColor Red
}
exit $LASTEXITCODE
