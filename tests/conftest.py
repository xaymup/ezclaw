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
