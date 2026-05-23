import sqlite3


def _columns(db, table):
    with sqlite3.connect(db.db_path) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_file_cache_table_created(tmp_db):
    cols = _columns(tmp_db, "file_cache")
    assert {"path", "mtime", "size", "content_hash", "content", "cached_at"} <= cols


def test_file_chunks_table_created(tmp_db):
    cols = _columns(tmp_db, "file_chunks")
    assert {
        "id", "path", "content_hash", "chunk_index",
        "start_line", "end_line", "content", "embedding",
    } <= cols
