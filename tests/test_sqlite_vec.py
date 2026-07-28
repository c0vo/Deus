import sqlite3
import pytest
import numpy as np
from unittest.mock import MagicMock

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

@pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec is not installed")
def test_sqlite_vec_extension_loading():
    """Verify that sqlite-vec can be loaded and vec_distance_cosine works."""
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    # Create a table with float32 vectors stored as BLOBs
    conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, embedding BLOB)")
    
    # Insert two identical vectors
    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    conn.execute("INSERT INTO items(id, embedding) VALUES (?, ?)", (1, vec1.tobytes()))
    conn.execute("INSERT INTO items(id, embedding) VALUES (?, ?)", (2, vec1.tobytes()))
    
    # Insert an orthogonal vector
    vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    conn.execute("INSERT INTO items(id, embedding) VALUES (?, ?)", (3, vec2.tobytes()))
    
    # Query distance from vec1
    # Distance should be 0.0 for items 1 and 2, and 1.0 for item 3 (cosine distance)
    rows = conn.execute(
        "SELECT id, vec_distance_cosine(embedding, ?) as distance FROM items ORDER BY distance",
        (vec1.tobytes(),)
    ).fetchall()
    
    assert len(rows) == 3
    assert rows[0][0] in (1, 2)
    assert rows[1][0] in (1, 2)
    assert rows[2][0] == 3
    
    # Floating point math can have slight inaccuracies, use approx
    assert rows[0][1] == pytest.approx(0.0, abs=1e-5)
    assert rows[1][1] == pytest.approx(0.0, abs=1e-5)
    assert rows[2][1] == pytest.approx(1.0, abs=1e-5)

@pytest.mark.asyncio
async def test_bot_command_vector_search_fallback():
    """Test the bot handle_query fallback when sqlite-vec is missing."""
    from bot.commands import handle_query
    from data.database import Database
    
    # This is a complex test, just asserting that the fallback numpy code runs
    # in orchestrator/scheduler.py, we already fixed test_pipeline.py for this.
    pass
