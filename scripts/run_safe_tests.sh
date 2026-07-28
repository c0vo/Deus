#!/bin/bash
# Run only the safe, fast, automated tests that don't require API keys.
set -e

python -m pytest \
    tests/test_prompts.py \
    tests/test_schemas.py \
    tests/test_classifier.py \
    tests/test_ranker.py \
    tests/test_agents.py \
    tests/test_aggregator.py \
    tests/test_embedder.py \
    tests/test_predictor.py \
    tests/test_chat_orchestrator.py \
    tests/test_market_scanner.py \
    tests/test_sqlite_vec.py \
    tests/test_sse_streaming.py \
    tests/test_database.py \
    tests/test_frontend.py \
    tests/e2e/ \
    -v --tb=short

echo ""
if [ $? -eq 0 ]; then
    echo "✅ All safe tests passed!"
else
    echo "❌ Some tests failed."
    exit 1
fi
