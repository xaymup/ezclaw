import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class Database:
    def __init__(self, db_path: str = "ezclaw.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    role TEXT,
                    content TEXT,
                    tool_calls TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            # Long-term memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def create_session(self, name: Optional[str] = None) -> int:
        if not name:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
            return cursor.lastrowid

    def add_message(self, session_id: int, role: str, content: str, tool_calls: Optional[List[Dict]] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
                (session_id, role, content, tool_calls_json)
            )
            conn.commit()

    def get_messages(self, session_id: int) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, tool_calls FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,)
            )
            messages = []
            for role, content, tool_calls_json in cursor.fetchall():
                msg = {"role": role, "content": content}
                if tool_calls_json:
                    msg["tool_calls"] = json.loads(tool_calls_json)
                messages.append(msg)
            return messages

    def add_memory(self, fact: str, tags: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO memories (fact, tags) VALUES (?, ?)", (fact, tags))
            conn.commit()

    def search_memories(self, query: str) -> List[str]:
        # Simple keyword-based search for now
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Simple LIKE search on fact and tags
            cursor.execute(
                "SELECT fact FROM memories WHERE fact LIKE ? OR tags LIKE ?",
                (f"%{query}%", f"%{query}%")
            )
            return [row[0] for row in cursor.fetchall()]

    def get_last_session_id(self) -> Optional[int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None
