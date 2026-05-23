# Embedding-Aware Filesystem Cache

**Status:** Draft
**Date:** 2026-05-23
**Owner:** Mohamed Kotp

## Problem

Agents in ezclaw frequently re-read the same workspace files within a single
session, and across sessions. Every read goes to disk through `read_file`
(`tools.py:198`), and the full content is then handed back to the LLM. This
wastes I/O, wastes tokens, and for files larger than the model can comfortably
hold, the existing `agent.py:429` middle-truncation drops content the agent
might actually need.

## Goal

Eliminate redundant disk reads and reduce token waste by introducing a
content-addressed cache for `read_file`. For files at or above an 8 KB
threshold, also store per-chunk embeddings so that — when the agent has a
current user query in context — `read_file` returns the most relevant excerpts
instead of the full file or a blind head/tail truncation.

The LLM-visible tool signature does not change. The behavior is entirely
server-side.

## Non-goals

- Semantic deduplication of *different* files. Caching is keyed by exact path
  and exact (mtime, size, content hash). Embeddings drive retrieval within a
  large file, not similarity matching across files.
- A general-purpose RAG index over the whole workspace. Chunks are stored
  per-file; there is no cross-file vector search.
- Cache eviction policy. The tables are bounded (one row per path in
  `file_cache`; chunks pruned when content hash changes) and are not expected
  to grow unbounded in v1. Eviction can be added later if needed.
- Changes to `write_file`'s diff behavior or any other tool besides
  `read_file` / `write_file`.

## Architecture

Three layers:

1. **Storage** — two new SQLite tables added to the existing `ezclaw.db`
   managed by `memory.Database`.
2. **Cache module** — a new `file_cache.py` exposing a single `FileCache`
   class that owns the read/invalidate API.
3. **Tool plumbing** — a `contextvars.ContextVar` declared in `tools.py`
   carries the current user query into tool calls. `agent.py` sets and resets
   it around each tool-call batch. `read_file` and `write_file` consult the
   cache.

## Components

### 1. Schema (memory.py)

Added to `Database._init_db` and `Database._migrate`:

```sql
CREATE TABLE IF NOT EXISTS file_cache (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS file_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE(path, content_hash, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_file_chunks_path_hash
    ON file_chunks(path, content_hash);
```

Migrations follow the existing `_migrate()` try/except-OperationalError
pattern so older `ezclaw.db` files upgrade in place.

### 2. file_cache.py (new module)

```python
LARGE_FILE_THRESHOLD = 8 * 1024  # bytes
CHUNK_LINES = 40
CHUNK_OVERLAP = 10
TOP_K = 5
HEAD_LINES = 150
TAIL_LINES = 50

class FileCache:
    def __init__(self, db: Database) -> None: ...
    def get(self, abs_path: str, query: str | None) -> str: ...
    def invalidate(self, abs_path: str) -> None: ...

    # internals
    def _stat_matches(self, abs_path, row) -> bool: ...
    def _read_and_store(self, abs_path) -> tuple[str, str]: ...   # (content, hash)
    def _chunk_and_embed(self, abs_path, content, content_hash) -> None: ...
    def _retrieve_chunks(self, abs_path, content_hash, query) -> str: ...
    def _head_tail(self, content, total_chunks: int) -> str: ...
```

Chunking strategy: split by lines into windows of `CHUNK_LINES` with
`CHUNK_OVERLAP` lines of overlap, so a boundary-straddling concept appears in
two chunks and is not missed by single-chunk retrieval. Chunk text is stored
verbatim alongside its `mxbai-embed-large` embedding via `embed.embed`. Top-K
retrieval scores cached chunk embeddings against the query embedding using
`embed.cosine_similarity`.

Return format for large-file chunked retrieval:

```
File too large for full read (12,400 bytes, 18 chunks). Showing 5 most relevant excerpts.

--- lines 42-81 ---
<chunk text>

--- lines 200-239 ---
<chunk text>
...
```

Return format for large-file fallback (no query in context):

```
File too large for full read (12,400 bytes, 18 chunks). Showing head and tail.

<first 150 lines>

... (truncated, 13 middle chunks) ...

<last 50 lines>
```

### 3. tools.py wiring

```python
import contextvars
from file_cache import FileCache

current_query: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_query", default=None
)

_file_cache: FileCache | None = None

def init_file_cache(db) -> None:
    global _file_cache
    _file_cache = FileCache(db)
```

`read_file(path)` (replaces existing body):

```python
full_path = get_workspace_path(path)
if not os.path.exists(full_path):
    return f"Error: File '{path}' does not exist."
try:
    return _file_cache.get(full_path, current_query.get())
except UnicodeDecodeError:
    return f"Error: File '{path}' is not valid UTF-8."
except Exception as e:
    return f"Error reading file: {str(e)}"
```

`write_file(path, content)` keeps its existing body verbatim, plus one line at
the end of the success path:

```python
_file_cache.invalidate(full_path)
```

### 4. agent.py wiring

In `Agent.__init__`, after `self.db` is assigned, add:

```python
from tools import init_file_cache
init_file_cache(self.db)
```

Inside `chat_stream(self, user_input)`, wrap the entire body once with
set/reset so the query is available across all iterations and tool calls in
the turn:

```python
from tools import current_query

def chat_stream(self, user_input: str):
    token = current_query.set(user_input)
    try:
        # existing chat_stream body unchanged
        ...
    finally:
        current_query.reset(token)
```

Because `user_input` does not change within a single `chat_stream` call,
setting it once at the top of the function is simpler than re-setting it
inside the agent's iteration loop, and correctly covers every tool dispatch.

## Data flow

### Read

1. Agent enters tool-dispatch loop; sets `current_query` to the user's
   current message via `ContextVar.set`.
2. LLM emits a `read_file("foo.py")` call.
3. `read_file` resolves the absolute workspace path and calls
   `FileCache.get(abs_path, current_query.get())`.
4. `FileCache.get`:
   - `os.stat()` the file. If it doesn't exist, raise — the caller's existence
     check already handled that path, so a missing-file at this stage is an
     external race; return an error string.
   - Look up `file_cache` row for the path.
   - If a row exists and `(mtime, size)` match:
     - Small file (`size < LARGE_FILE_THRESHOLD`): return `row.content`.
     - Large file with non-None query: call `_retrieve_chunks`.
     - Large file with None query: call `_head_tail`.
   - If no row or stat mismatch: call `_read_and_store`, then `_chunk_and_embed`
     if large, then return as above.
5. Result flows back through `read_file` to the agent and into the LLM
   context.

### Write

1. `write_file` runs its existing read-old / write-new / diff sequence.
2. On the success path, calls `_file_cache.invalidate(full_path)` which
   deletes from both `file_cache` and `file_chunks` for that path.
3. Next read for that path repopulates.

### Chunk content-hash semantics

`file_chunks.content_hash` ties a chunk row to a specific file version. On
repopulate, `_chunk_and_embed` first deletes `file_chunks` rows with the same
`path` but a different `content_hash`, then inserts the new chunks. This is
self-healing — if an invalidation is ever missed (crash mid-write, external
edit not yet observed), the next miss-driven repopulate cleans up.

## Error handling

| Scenario | Behavior |
|---|---|
| File not found | Existing error message; cache row pruned if present. |
| Binary / non-UTF-8 file | `UnicodeDecodeError` caught at the `read_file` boundary; existing error message returned; nothing cached. |
| Ollama embedding service down during chunking | `_chunk_and_embed` catches the exception, logs a warning, leaves `file_chunks` empty for this hash. Raw content is still cached. Subsequent reads fall back to `_head_tail`. Lazy retry: when a large file has a `file_cache` row but zero `file_chunks` rows for its current hash, the next read attempts chunking again. |
| External edit between reads | Detected by mtime+size mismatch; cache invalidates and repopulates on next read. |
| File deleted between cache write and next read | `os.stat` raises `FileNotFoundError`; return the existing "File does not exist" error and delete the cache row. |
| Concurrent writes within one process | SQLite handles row-level locking; cache module uses one short transaction per call. |
| Path normalization | Always store the absolute path from `get_workspace_path` so different relative paths to the same file share a single cache row. |
| Schema migration on existing `ezclaw.db` | New `CREATE TABLE IF NOT EXISTS` statements run idempotently; no destructive migration. |

## Testing

### Unit tests (`test_file_cache.py`)

- **First read populates, second read does not reopen.** Patch `builtins.open`
  and assert the second `read_file` call does not open the file.
- **mtime change invalidates.** Touch the file's mtime; next read repopulates.
- **`write_file` invalidates.** Write new content; next read returns new
  content.
- **Large file + query returns chunks.** Create a >8 KB fixture; set the
  contextvar; assert response begins with the chunked-format header and
  contains the expected number of `--- lines N-M ---` blocks.
- **Large file + no query returns head/tail.** Same fixture, contextvar unset;
  assert head+tail format.
- **Binary file is not cached.** Try a non-UTF-8 fixture; assert no row in
  `file_cache`.
- **Chunk pruning on hash change.** Read a large file, modify content,
  re-read; assert no chunks with the old `content_hash` remain.

### Integration test

- In a stubbed `Agent.chat_stream` flow, dispatch two consecutive
  `read_file("foo.py")` tool calls in the same turn; assert one disk open and
  one query-embedding call total.

### Failure-mode test

- Mock `embed.embed` to raise. Read a large file. Assert raw content is
  cached, `file_chunks` is empty for this hash, and the returned string uses
  the head/tail format.

## Open questions

None.
