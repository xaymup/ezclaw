# Embedding-Aware Filesystem Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache `read_file` results in SQLite keyed by path + mtime + size + content hash, and for files ≥ 8 KB also store per-chunk embeddings so subsequent reads with a current user query return the most relevant excerpts instead of the full file.

**Architecture:** Three layers — schema additions in `memory.py`, a new `file_cache.py` module owning the cache API (`FileCache.get` / `FileCache.invalidate`), and tool plumbing that uses a `contextvars.ContextVar("current_query")` declared in `tools.py` and set/reset around `chat_stream` in `agent.py`. The LLM-visible `read_file(path)` signature does not change.

**Tech Stack:** Python 3, SQLite via `sqlite3` stdlib, `ollama` for embeddings (existing `embed.py`), `numpy` for cosine similarity, `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-05-23-fs-embedding-cache-design.md`

---

## File Structure

**Created:**
- `file_cache.py` — `FileCache` class; chunking constants; sole owner of cache reads/writes/invalidation.
- `tests/__init__.py` — empty package marker.
- `tests/conftest.py` — shared pytest fixtures (`tmp_db`, `cache`, `large_text`, `small_text`).
- `tests/test_file_cache.py` — all cache behaviour tests.
- `tests/test_read_write_integration.py` — end-to-end test that two `read_file` calls within one turn perform only one disk open and one query embedding.

**Modified:**
- `memory.py` — add `file_cache` and `file_chunks` table creation in `_init_db` and matching idempotent CREATE statements in `_migrate`.
- `tools.py` — declare `current_query` ContextVar, add `init_file_cache(db)`, replace `read_file` body to consult the cache, append `_file_cache.invalidate(...)` to `write_file`.
- `agent.py` — call `init_file_cache(self.db)` in `ChatAgent.__init__`; wrap `chat_stream` body with `current_query.set()` / `current_query.reset()`.
- `requirements.txt` — add `pytest`.

---

## Task 1: Scaffold pytest and tests directory

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest to requirements**

Modify `requirements.txt` so the full contents are:

```
ollama>=0.4.0
rich
prompt_toolkit
python-dotenv
httpx
numpy
pytest
```

- [ ] **Step 2: Install it**

Run: `pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 3: Create tests package**

Create `tests/__init__.py` as an empty file.

- [ ] **Step 4: Create shared fixtures**

Create `tests/conftest.py`:

```python
import os
import sys
import pytest

# Make the project root importable so tests can `import memory`, `import file_cache`, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import Database


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    return Database(str(db_path))


@pytest.fixture
def small_text():
    return "line one\nline two\nline three\n"


@pytest.fixture
def large_text():
    # ~12 KB of distinct line content so chunking produces multiple chunks.
    lines = []
    for i in range(400):
        lines.append(f"This is line {i:04d} with some unique words like topic_{i % 7}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def small_file(tmp_path, small_text):
    path = tmp_path / "small.txt"
    path.write_text(small_text)
    return str(path)


@pytest.fixture
def large_file(tmp_path, large_text):
    path = tmp_path / "large.txt"
    path.write_text(large_text)
    return str(path)
```

- [ ] **Step 5: Verify pytest discovers the package**

Run: `pytest tests/ -q`
Expected: `no tests ran` (no failures, just zero collected).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/__init__.py tests/conftest.py
git commit -m "test: scaffold pytest setup for file_cache work"
```

---

## Task 2: Add file_cache and file_chunks schema

**Files:**
- Modify: `memory.py:12-21` (`_init_db`) and `memory.py:23-34` (`_migrate`)
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_file_cache.py`:

```python
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
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `pytest tests/test_file_cache.py -v`
Expected: both tests fail with `sqlite3.OperationalError: no such table: file_cache` / `file_chunks`.

- [ ] **Step 3: Add the tables to `_init_db`**

In `memory.py`, inside `Database._init_db`, after the existing `cursor.execute('''CREATE TABLE IF NOT EXISTS routing_history ...''')` line and before `conn.commit()`, add:

```python
cursor.execute('''CREATE TABLE IF NOT EXISTS file_cache (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS file_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE(path, content_hash, chunk_index)
)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_file_chunks_path_hash
    ON file_chunks(path, content_hash)''')
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `pytest tests/test_file_cache.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add memory.py tests/test_file_cache.py
git commit -m "feat(memory): add file_cache and file_chunks tables"
```

---

## Task 3: Create FileCache skeleton — small-file miss-then-hit

**Files:**
- Create: `file_cache.py`
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
from unittest.mock import patch
import builtins
import file_cache as fc_mod
from file_cache import FileCache


def test_small_file_miss_then_hit(tmp_db, small_file, small_text):
    cache = FileCache(tmp_db)

    # First call: cache miss, must read from disk.
    result1 = cache.get(small_file, query=None)
    assert result1 == small_text

    # Second call: must NOT open the file.
    real_open = builtins.open
    with patch("builtins.open", side_effect=AssertionError("should not reopen")) as m:
        # Allow sqlite to keep using open under the hood — patch only this module.
        with patch.object(fc_mod, "open", real_open, create=True):
            result2 = cache.get(small_file, query=None)
    assert result2 == small_text
```

Note: the double-patch leaves SQLite's own file access (which goes through C extensions, not `builtins.open` from Python frames) untouched while still asserting `file_cache.py` does not call `open()` again. If your `file_cache.py` uses `open(...)` without the `file_cache.open` alias, the global `builtins.open` patch will catch it.

- [ ] **Step 2: Run it to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_small_file_miss_then_hit -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'file_cache'`.

- [ ] **Step 3: Create `file_cache.py` with the minimal implementation**

Create `file_cache.py`:

```python
import hashlib
import os
import sqlite3
from typing import Optional

from memory import Database

LARGE_FILE_THRESHOLD = 8 * 1024  # bytes
CHUNK_LINES = 40
CHUNK_OVERLAP = 10
TOP_K = 5
HEAD_LINES = 150
TAIL_LINES = 50


class FileCache:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ── Public API ───────────────────────────────────────────────────

    def get(self, abs_path: str, query: Optional[str]) -> str:
        row = self._lookup(abs_path)
        st = os.stat(abs_path)

        if row and row["mtime"] == st.st_mtime and row["size"] == st.st_size:
            return self._return_cached(row, query)

        content, content_hash = self._read_and_store(abs_path, st)
        return content

    def invalidate(self, abs_path: str) -> None:
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute("DELETE FROM file_cache WHERE path = ?", (abs_path,))
            conn.execute("DELETE FROM file_chunks WHERE path = ?", (abs_path,))
            conn.commit()

    # ── Internals ────────────────────────────────────────────────────

    def _lookup(self, abs_path: str) -> Optional[dict]:
        with sqlite3.connect(self.db.db_path) as conn:
            cur = conn.execute(
                "SELECT path, mtime, size, content_hash, content FROM file_cache WHERE path = ?",
                (abs_path,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {"path": r[0], "mtime": r[1], "size": r[2], "content_hash": r[3], "content": r[4]}

    def _read_and_store(self, abs_path: str, st: os.stat_result) -> tuple[str, str]:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO file_cache "
                "(path, mtime, size, content_hash, content) VALUES (?, ?, ?, ?, ?)",
                (abs_path, st.st_mtime, st.st_size, content_hash, content),
            )
            conn.commit()
        return content, content_hash

    def _return_cached(self, row: dict, query: Optional[str]) -> str:
        # Small-file fast path. Large-file logic added in later tasks.
        return row["content"]
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_small_file_miss_then_hit -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add file_cache.py tests/test_file_cache.py
git commit -m "feat(file_cache): add FileCache with small-file caching"
```

---

## Task 4: Invalidation

**Files:**
- Modify: `file_cache.py` (already has `invalidate` from Task 3; this task only adds the test)
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_invalidate_clears_cache(tmp_db, small_file, small_text):
    cache = FileCache(tmp_db)
    cache.get(small_file, query=None)

    cache.invalidate(small_file)

    # After invalidation, a stat that doesn't match a stored row forces a fresh read.
    # We assert by patching `open` to detect reopen:
    import file_cache as fc_mod
    real_open = builtins.open
    opens = []

    def tracking_open(*a, **k):
        opens.append(a[0])
        return real_open(*a, **k)

    with patch("builtins.open", side_effect=tracking_open):
        cache.get(small_file, query=None)
    assert small_file in opens
```

- [ ] **Step 2: Run it to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_invalidate_clears_cache -v`
Expected: PASS (the implementation from Task 3 already handles invalidation correctly).

- [ ] **Step 3: Commit**

```bash
git add tests/test_file_cache.py
git commit -m "test(file_cache): cover invalidate clears the row"
```

---

## Task 5: mtime change forces re-read

**Files:**
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
import time


def test_mtime_change_invalidates(tmp_db, small_file, small_text):
    cache = FileCache(tmp_db)
    cache.get(small_file, query=None)

    # Modify content (changes mtime AND size — both should miss).
    new_text = small_text + "added line\n"
    with open(small_file, "w") as f:
        f.write(new_text)
    # Some filesystems coalesce mtimes if writes are within the same second.
    # Bump it explicitly to be safe.
    future = time.time() + 5
    os.utime(small_file, (future, future))

    result = cache.get(small_file, query=None)
    assert result == new_text
```

- [ ] **Step 2: Run it to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_mtime_change_invalidates -v`
Expected: PASS (Task 3's stat comparison already handles this).

- [ ] **Step 3: Commit**

```bash
git add tests/test_file_cache.py
git commit -m "test(file_cache): cover mtime-change invalidation"
```

---

## Task 6: Chunk and embed on large-file miss

**Files:**
- Modify: `file_cache.py`
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_large_file_miss_writes_chunks(tmp_db, large_file, monkeypatch):
    import file_cache as fc_mod

    # Stub embed() to return a deterministic vector so we don't hit Ollama in tests.
    def fake_embed(text):
        # Cheap deterministic embedding: ord sums spread across 4 dims.
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h[:4]]

    monkeypatch.setattr(fc_mod, "embed", fake_embed, raising=False)

    cache = FileCache(tmp_db)
    cache.get(large_file, query=None)

    with sqlite3.connect(tmp_db.db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM file_chunks WHERE path = ?", (large_file,)
        ).fetchone()[0]
    assert n >= 2  # large_text fixture should produce multiple chunks
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_large_file_miss_writes_chunks -v`
Expected: FAIL — `assert 0 >= 2` (no chunks written yet).

- [ ] **Step 3: Implement chunking and embedding**

Add to `file_cache.py` near the top of the file, with the other imports:

```python
import pickle
from embed import embed, cosine_similarity
```

Modify `_read_and_store` in `file_cache.py` to also chunk-and-embed large files. Replace the existing `_read_and_store` with:

```python
def _read_and_store(self, abs_path: str, st: os.stat_result) -> tuple[str, str]:
    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with sqlite3.connect(self.db.db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO file_cache "
            "(path, mtime, size, content_hash, content) VALUES (?, ?, ?, ?, ?)",
            (abs_path, st.st_mtime, st.st_size, content_hash, content),
        )
        conn.commit()
    if st.st_size >= LARGE_FILE_THRESHOLD:
        self._chunk_and_embed(abs_path, content, content_hash)
    return content, content_hash
```

Then add `_chunk_and_embed` as a new method on `FileCache`:

```python
def _chunk_and_embed(self, abs_path: str, content: str, content_hash: str) -> None:
    # Drop any chunks for this path with a different hash (stale versions).
    with sqlite3.connect(self.db.db_path) as conn:
        conn.execute(
            "DELETE FROM file_chunks WHERE path = ? AND content_hash != ?",
            (abs_path, content_hash),
        )
        # If chunks already exist for this exact hash, no-op.
        n = conn.execute(
            "SELECT COUNT(*) FROM file_chunks WHERE path = ? AND content_hash = ?",
            (abs_path, content_hash),
        ).fetchone()[0]
        if n > 0:
            conn.commit()
            return

        lines = content.splitlines(keepends=True)
        if not lines:
            conn.commit()
            return

        step = CHUNK_LINES - CHUNK_OVERLAP
        chunk_index = 0
        for start in range(0, len(lines), step):
            end = min(start + CHUNK_LINES, len(lines))
            chunk_text = "".join(lines[start:end])
            try:
                vec = embed(chunk_text)
                blob = pickle.dumps(vec)
            except Exception:
                # Embedding service unavailable — skip chunk storage entirely so
                # the lazy retry path in get() kicks in next time.
                conn.rollback()
                return
            conn.execute(
                "INSERT INTO file_chunks "
                "(path, content_hash, chunk_index, start_line, end_line, content, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (abs_path, content_hash, chunk_index, start + 1, end, chunk_text, blob),
            )
            chunk_index += 1
            if end == len(lines):
                break
        conn.commit()
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_large_file_miss_writes_chunks -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add file_cache.py tests/test_file_cache.py
git commit -m "feat(file_cache): chunk and embed large files on miss"
```

---

## Task 7: Large-file + query returns top-K chunks

**Files:**
- Modify: `file_cache.py`
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_large_file_with_query_returns_chunks(tmp_db, large_file, monkeypatch):
    import file_cache as fc_mod

    # Deterministic embedding: vectors based on whether "topic_3" is present.
    def fake_embed(text):
        return [1.0 if "topic_3" in text else 0.1, 0.5, 0.0]

    monkeypatch.setattr(fc_mod, "embed", fake_embed, raising=False)

    cache = FileCache(tmp_db)
    result = cache.get(large_file, query="topic_3")

    assert "File too large for full read" in result
    assert "--- lines " in result
    # Most relevant excerpts should contain "topic_3"
    assert "topic_3" in result
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_large_file_with_query_returns_chunks -v`
Expected: FAIL — current `_return_cached` returns full content, not chunk excerpts.

- [ ] **Step 3: Implement retrieval and the dispatch in `_return_cached`**

Replace `_return_cached` in `file_cache.py` with:

```python
def _return_cached(self, row: dict, query: Optional[str]) -> str:
    if row["size"] < LARGE_FILE_THRESHOLD:
        return row["content"]
    chunk_count = self._chunk_count(row["path"], row["content_hash"])
    if chunk_count == 0:
        # Lazy retry: chunks were never written (embedding outage or first-time miss
        # raced with this code path). Fall back to head/tail.
        return self._head_tail(row["content"], total_chunks=0)
    if query:
        return self._retrieve_chunks(row["path"], row["content_hash"], query, row["size"], chunk_count)
    return self._head_tail(row["content"], total_chunks=chunk_count)
```

Add the supporting methods to `FileCache`:

```python
def _chunk_count(self, abs_path: str, content_hash: str) -> int:
    with sqlite3.connect(self.db.db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM file_chunks WHERE path = ? AND content_hash = ?",
            (abs_path, content_hash),
        ).fetchone()[0]

def _retrieve_chunks(self, abs_path: str, content_hash: str, query: str,
                    size: int, chunk_count: int) -> str:
    try:
        q_vec = embed(query)
    except Exception:
        # If we can't embed the query, fall back to head/tail.
        return self._load_full_then_head_tail(abs_path, content_hash, chunk_count, size)
    with sqlite3.connect(self.db.db_path) as conn:
        rows = conn.execute(
            "SELECT chunk_index, start_line, end_line, content, embedding "
            "FROM file_chunks WHERE path = ? AND content_hash = ?",
            (abs_path, content_hash),
        ).fetchall()
    scored = []
    for idx, start, end, text, blob in rows:
        c_vec = pickle.loads(blob)
        scored.append((cosine_similarity(q_vec, c_vec), start, end, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]
    # Re-sort selected by start_line so output reads in file order.
    top.sort(key=lambda x: x[1])
    header = (
        f"File too large for full read ({size} bytes, {chunk_count} chunks). "
        f"Showing {len(top)} most relevant excerpts.\n\n"
    )
    body = "\n".join(f"--- lines {s}-{e} ---\n{t}" for _, s, e, t in top)
    return header + body

def _load_full_then_head_tail(self, abs_path: str, content_hash: str,
                              chunk_count: int, size: int) -> str:
    with sqlite3.connect(self.db.db_path) as conn:
        row = conn.execute(
            "SELECT content FROM file_cache WHERE path = ?", (abs_path,)
        ).fetchone()
    content = row[0] if row else ""
    return self._head_tail(content, total_chunks=chunk_count)
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_large_file_with_query_returns_chunks -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add file_cache.py tests/test_file_cache.py
git commit -m "feat(file_cache): return top-K relevant chunks for large file + query"
```

---

## Task 8: Large-file head/tail fallback when no query

**Files:**
- Modify: `file_cache.py`
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_large_file_no_query_returns_head_tail(tmp_db, large_file, monkeypatch):
    import file_cache as fc_mod
    monkeypatch.setattr(fc_mod, "embed", lambda t: [0.0, 0.0, 0.0], raising=False)

    cache = FileCache(tmp_db)
    result = cache.get(large_file, query=None)

    assert "File too large for full read" in result
    assert "Showing head and tail" in result
    assert "... (truncated," in result
    # First and last lines of the fixture must appear
    assert "line 0000" in result
    assert "line 0399" in result
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_large_file_no_query_returns_head_tail -v`
Expected: FAIL — `_head_tail` does not yet exist on `FileCache`.

- [ ] **Step 3: Implement `_head_tail`**

Add to `FileCache`:

```python
def _head_tail(self, content: str, total_chunks: int) -> str:
    lines = content.splitlines(keepends=True)
    if len(lines) <= HEAD_LINES + TAIL_LINES:
        # File fits within head+tail span — just return whole content with header.
        header = (
            f"File too large for full read ({len(content)} bytes). "
            "Showing full content.\n\n"
        )
        return header + content
    head = "".join(lines[:HEAD_LINES])
    tail = "".join(lines[-TAIL_LINES:])
    middle_chunks = max(total_chunks - 2, 0)  # rough; chunks overlap so this is approximate
    header = (
        f"File too large for full read ({len(content)} bytes, {total_chunks} chunks). "
        "Showing head and tail.\n\n"
    )
    sep = f"\n... (truncated, ~{middle_chunks} middle chunks) ...\n\n"
    return header + head + sep + tail
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_large_file_no_query_returns_head_tail -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add file_cache.py tests/test_file_cache.py
git commit -m "feat(file_cache): head/tail fallback for large files with no query"
```

---

## Task 9: Embedding outage — raw content still cached, lazy retry

**Files:**
- Modify: `file_cache.py` (already handles via try/except + rollback)
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_embedding_failure_still_caches_raw_then_retries(tmp_db, large_file, large_text, monkeypatch):
    import file_cache as fc_mod

    calls = {"n": 0}

    def flaky_embed(text):
        calls["n"] += 1
        if calls["n"] <= 1:
            raise RuntimeError("ollama down")
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(fc_mod, "embed", flaky_embed, raising=False)

    cache = FileCache(tmp_db)
    # First read: chunking embedding raises -> chunks empty, but raw cached.
    result1 = cache.get(large_file, query=None)
    assert "File too large" in result1

    with sqlite3.connect(tmp_db.db_path) as conn:
        chunks = conn.execute(
            "SELECT COUNT(*) FROM file_chunks WHERE path = ?", (large_file,)
        ).fetchone()[0]
        cached = conn.execute(
            "SELECT COUNT(*) FROM file_cache WHERE path = ?", (large_file,)
        ).fetchone()[0]
    assert cached == 1
    assert chunks == 0

    # Second read: detects empty chunks for current hash, retries chunking,
    # this time embed() succeeds.
    cache.get(large_file, query=None)
    with sqlite3.connect(tmp_db.db_path) as conn:
        chunks_after = conn.execute(
            "SELECT COUNT(*) FROM file_chunks WHERE path = ?", (large_file,)
        ).fetchone()[0]
    assert chunks_after >= 2
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_embedding_failure_still_caches_raw_then_retries -v`
Expected: FAIL — current code does not retry chunking on cache hit when chunks are empty.

- [ ] **Step 3: Add lazy chunking retry inside `get`**

In `file_cache.py`, modify `get` so that on a cache hit for a large file with zero chunks, we attempt to chunk again before returning:

```python
def get(self, abs_path: str, query: Optional[str]) -> str:
    row = self._lookup(abs_path)
    st = os.stat(abs_path)

    if row and row["mtime"] == st.st_mtime and row["size"] == st.st_size:
        # Lazy retry: large file with no chunks yet -> try to embed now.
        if st.st_size >= LARGE_FILE_THRESHOLD and self._chunk_count(abs_path, row["content_hash"]) == 0:
            self._chunk_and_embed(abs_path, row["content"], row["content_hash"])
        return self._return_cached(row, query)

    content, content_hash = self._read_and_store(abs_path, st)
    # Refetch row so `_return_cached` has the fresh fields.
    row = self._lookup(abs_path)
    return self._return_cached(row, query)
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_embedding_failure_still_caches_raw_then_retries -v`
Expected: PASS.

- [ ] **Step 5: Run the full file_cache test file**

Run: `pytest tests/test_file_cache.py -v`
Expected: all tests so far PASS.

- [ ] **Step 6: Commit**

```bash
git add file_cache.py tests/test_file_cache.py
git commit -m "feat(file_cache): lazy retry chunking after embedding outage"
```

---

## Task 10: Binary / non-UTF-8 files are not cached

**Files:**
- Modify: `file_cache.py`
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_binary_file_not_cached(tmp_db, tmp_path):
    bin_path = tmp_path / "data.bin"
    bin_path.write_bytes(b"\x00\xff\xfe\x80not valid utf-8\x80")

    cache = FileCache(tmp_db)
    with pytest.raises(UnicodeDecodeError):
        cache.get(str(bin_path), query=None)

    with sqlite3.connect(tmp_db.db_path) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM file_cache WHERE path = ?", (str(bin_path),)
        ).fetchone()[0]
    assert rows == 0
```

Also at the top of the file, add `import pytest` if it isn't already there.

- [ ] **Step 2: Run the test to confirm it fails or passes**

Run: `pytest tests/test_file_cache.py::test_binary_file_not_cached -v`
Expected: it likely already PASSES (because `_read_and_store` raises on decode, which bubbles up before the INSERT). Verify behaviour. If it FAILS, see Step 3.

- [ ] **Step 3: If failing, ensure decode happens before INSERT**

The current `_read_and_store` reads, then inserts. The `f.read()` call raises `UnicodeDecodeError` before the SQL INSERT, so no row should be created. If the test is failing, inspect for an INSERT-then-decode ordering bug and fix by moving the SQL INSERT below the `content` assignment. (The reference implementation in Task 6 already has the correct order.)

- [ ] **Step 4: Re-run**

Run: `pytest tests/test_file_cache.py::test_binary_file_not_cached -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_file_cache.py
git commit -m "test(file_cache): binary file does not pollute cache"
```

---

## Task 11: Add ContextVar and init_file_cache to tools.py

**Files:**
- Modify: `tools.py`
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_contextvar_and_init(tmp_db):
    import tools
    assert hasattr(tools, "current_query")
    assert tools.current_query.get() is None

    tools.init_file_cache(tmp_db)
    assert tools._file_cache is not None
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_contextvar_and_init -v`
Expected: FAIL — `AttributeError: module 'tools' has no attribute 'current_query'`.

- [ ] **Step 3: Add the contextvar and init**

At the top of `tools.py`, add to the imports:

```python
import contextvars
```

Then after the imports block and before `class ToolRegistry`, add:

```python
current_query: contextvars.ContextVar = contextvars.ContextVar("current_query", default=None)

_file_cache = None


def init_file_cache(db) -> None:
    """
    Initialize the module-level FileCache. Called once by the agent during startup.
    """
    global _file_cache
    from file_cache import FileCache
    _file_cache = FileCache(db)
```

The `from file_cache import FileCache` is deliberately inside the function to avoid a circular import at module load (since `file_cache.py` imports from `memory.py` only, not from `tools.py`, this is just defensive — but it costs nothing).

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_contextvar_and_init -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_file_cache.py
git commit -m "feat(tools): add current_query ContextVar and init_file_cache"
```

---

## Task 12: Wire read_file to use FileCache

**Files:**
- Modify: `tools.py:197-209` (the `read_file` body)
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_read_file_uses_cache(tmp_db, tmp_path, monkeypatch):
    import tools

    # Workspace is wherever tools.get_workspace_path resolves — for the test we
    # point WORKSPACE_DIR (if respected) or write a file at the exact resolved path.
    test_file = tmp_path / "hello.txt"
    test_file.write_text("hello world\n")

    monkeypatch.setattr(tools, "get_workspace_path", lambda p: str(test_file))
    tools.init_file_cache(tmp_db)

    # First call: cache miss + read.
    out1 = tools.read_file("hello.txt")
    assert out1 == "hello world\n"

    # Now delete the file on disk. If cache is working, second call still
    # returns cached content because mtime+size still match what we cached.
    # (We do NOT actually delete because mtime would mismatch; instead we
    # patch open to detect it isn't reopened.)
    import builtins
    real_open = builtins.open

    def fail_open(*a, **k):
        raise AssertionError(f"read_file should not re-open: {a}")

    # Allow sqlite's open through (which uses C-level open, not builtins.open).
    monkeypatch.setattr("builtins.open", fail_open)
    out2 = tools.read_file("hello.txt")
    assert out2 == "hello world\n"
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_read_file_uses_cache -v`
Expected: FAIL — current `read_file` always opens the file.

- [ ] **Step 3: Replace `read_file` body**

In `tools.py`, replace the entire `read_file` function (lines 197-209 in the original) with:

```python
@registry.register(auth_required=True)
def read_file(path: str) -> str:
    """
    Read the content of a file within the workspace.
    """
    try:
        full_path = get_workspace_path(path)
        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist."
        if _file_cache is None:
            # Cache not initialized (e.g. running outside the agent) — fall back to plain read.
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        return _file_cache.get(full_path, current_query.get())
    except UnicodeDecodeError:
        return f"Error: File '{path}' is not valid UTF-8."
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_read_file_uses_cache -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_file_cache.py
git commit -m "feat(tools): route read_file through FileCache"
```

---

## Task 13: Wire write_file to invalidate

**Files:**
- Modify: `tools.py:211-239` (the `write_file` body)
- Test: `tests/test_file_cache.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_file_cache.py`:

```python
def test_write_file_invalidates_cache(tmp_db, tmp_path, monkeypatch):
    import tools

    test_file = tmp_path / "note.txt"
    test_file.write_text("v1\n")

    monkeypatch.setattr(tools, "get_workspace_path", lambda p: str(test_file))
    tools.init_file_cache(tmp_db)

    assert tools.read_file("note.txt") == "v1\n"
    tools.write_file("note.txt", "v2\n")
    assert tools.read_file("note.txt") == "v2\n"
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_file_cache.py::test_write_file_invalidates_cache -v`
Expected: FAIL — second `read_file` returns `"v1\n"` from the stale cache row because mtime may not have changed within the same second on some filesystems, and current `write_file` doesn't invalidate.

- [ ] **Step 3: Add invalidation to `write_file`**

In `tools.py`, modify `write_file` so that just before each of the two `return` statements in the success path, it invalidates. Cleanest is to invalidate once after the disk write and before returning. Replace the body:

```python
@registry.register(auth_required=True)
def write_file(path: str, content: str) -> str:
    """
    Write or update a file. Shows a diff if the file already exists.
    """
    try:
        full_path = get_workspace_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        old_content = ""
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        if _file_cache is not None:
            _file_cache.invalidate(full_path)
        
        if old_content:
            diff = difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}"
            )
            diff_text = "".join(diff)
            return f"File updated: {path}\n\nDiff:\n{diff_text}" if diff_text else "No changes detected."
        return f"File created: {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `pytest tests/test_file_cache.py::test_write_file_invalidates_cache -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_file_cache.py
git commit -m "feat(tools): invalidate FileCache on write_file"
```

---

## Task 14: Hook agent.py — init and contextvar set/reset

**Files:**
- Modify: `agent.py` (`ChatAgent.__init__` and `chat_stream`)

- [ ] **Step 1: Add init_file_cache call in `__init__`**

In `agent.py`, inside `ChatAgent.__init__`, immediately after the line:

```python
self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
```

(currently `agent.py:69`), add:

```python
from tools import init_file_cache
init_file_cache(self.db)
```

- [ ] **Step 2: Wrap `chat_stream` body with contextvar set/reset**

Also in `agent.py`, modify `chat_stream` (currently starting at line 261). The current shape is:

```python
def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
    # Truncate long user input to prevent context overflow
    if len(user_input) > 4000:
        user_input = user_input[:4000] + "\n... (truncated)"

    # 1. Consolidated intent analysis with full agent context
    intent = self._process_intent(user_input)
    ...
```

Change it to wrap the entire body in a `try/finally` that holds the ContextVar token. Since `chat_stream` is a generator (`yield` statements), the `finally` will still run when the generator is closed or exhausted:

```python
def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
    # Truncate long user input to prevent context overflow
    if len(user_input) > 4000:
        user_input = user_input[:4000] + "\n... (truncated)"

    from tools import current_query
    _query_token = current_query.set(user_input)
    try:
        # 1. Consolidated intent analysis with full agent context
        intent = self._process_intent(user_input)
        ...
        # rest of the existing body, indented one extra level
    finally:
        current_query.reset(_query_token)
```

Concretely: indent every line from `intent = self._process_intent(...)` through the end of the function (down to and including the existing `if iteration_count >= max_iterations:` block at `agent.py:438-439`) by 4 spaces, and add the `finally:` reset at the same indentation as `try:`.

- [ ] **Step 3: Smoke-test the import path**

Run: `python -c "from agent import ChatAgent; print('ok')"`
Expected: `ok` (no import errors). The Database init may print or open a connection — that's fine.

- [ ] **Step 4: Run all tests to confirm nothing regressed**

Run: `pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add agent.py
git commit -m "feat(agent): init FileCache and propagate current_query to tools"
```

---

## Task 15: Integration test — two reads, one disk open

**Files:**
- Create: `tests/test_read_write_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_read_write_integration.py`:

```python
import builtins
import os
from unittest.mock import patch

import tools


def test_two_reads_in_same_turn_share_disk_and_embed(tmp_db, tmp_path, monkeypatch):
    """Two read_file calls within one tool-dispatch turn must perform exactly
    one disk read of the target file. (Query embedding is irrelevant for
    small files, which is what we use here.)"""

    target = tmp_path / "shared.txt"
    target.write_text("shared content\n")

    monkeypatch.setattr(tools, "get_workspace_path", lambda p: str(target))
    tools.init_file_cache(tmp_db)

    # Count opens of the target file specifically (not all opens, since SQLite
    # may go through builtins.open in some configurations).
    real_open = builtins.open
    target_opens = []

    def counting_open(path, *a, **k):
        if str(path) == str(target):
            target_opens.append(path)
        return real_open(path, *a, **k)

    with patch("builtins.open", side_effect=counting_open):
        out1 = tools.read_file("shared.txt")
        out2 = tools.read_file("shared.txt")

    assert out1 == "shared content\n"
    assert out2 == "shared content\n"
    assert len(target_opens) == 1, f"expected 1 disk read of {target}, got {len(target_opens)}"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_read_write_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_read_write_integration.py
git commit -m "test: integration test for shared-cache reads within one turn"
```

---

## Task 16: Manual smoke test against the real agent

**Files:** none

- [ ] **Step 1: Start the CLI**

Run: `python cli.py`

- [ ] **Step 2: Prompt a read**

In the CLI, send: `Read README.md and summarize it.`

Expected: the agent calls `read_file("README.md")` and returns content. No errors in the trace.

- [ ] **Step 3: Prompt a second read of the same file in the same session**

Send: `Read README.md again.`

Expected: still works, returns the same content. (You can confirm a cache hit by checking `ezclaw.db`:)

Run: `sqlite3 ezclaw.db "SELECT path, size FROM file_cache LIMIT 5;"`
Expected: a row for the absolute path of `README.md`.

- [ ] **Step 4: Prompt a write, then a read of the same file**

Send: `Use write_file to add a new line to a new file scratch.txt with content "hello".`
Then send: `Read scratch.txt.`

Expected: write succeeds, read returns `hello`. Then:

Run: `sqlite3 ezclaw.db "SELECT path, size FROM file_cache WHERE path LIKE '%scratch.txt';"`
Expected: one row with size > 0.

- [ ] **Step 5: Test a large file**

Create `/tmp/big.py` with 400 lines of distinct content, copy it to the workspace, then in the CLI: `Read big.py — what's in line 200?`

Expected: response includes relevant chunks (not full file or blind truncation).

- [ ] **Step 6: No commit needed (manual test).** If anything fails, file a follow-up.

---

## Final Verification

- [ ] **Run the complete test suite**

Run: `pytest tests/ -v`
Expected: every test from Tasks 2–15 PASSES.

- [ ] **Confirm no regressions in import paths**

Run: `python -c "from agent import ChatAgent; from tools import read_file, write_file, current_query, init_file_cache; from file_cache import FileCache; print('all imports ok')"`
Expected: `all imports ok`.

- [ ] **Inspect commit history is clean**

Run: `git log --oneline -16`
Expected: roughly 14 commits in feature order (one per task with code/tests).
