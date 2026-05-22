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
            cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, role TEXT, content TEXT, tool_calls TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (session_id) REFERENCES sessions (id))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT, tags TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()

    def create_session(self, name: Optional[str] = None) -> int:
        if not name: name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
            return cursor.lastrowid

    def add_message(self, session_id: int, role: str, content: str, tool_calls: Optional[List[Dict]] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            cursor.execute("INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)", (session_id, role, content, tool_calls_json))
            conn.commit()

    def get_messages(self, session_id: int) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, content, tool_calls FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
            messages = []
            for role, content, tool_calls_json in cursor.fetchall():
                msg = {"role": role, "content": content}
                if tool_calls_json: msg["tool_calls"] = json.loads(tool_calls_json)
                messages.append(msg)
            return messages

    def add_memory(self, fact: str, tags: Optional[str] = None):
        if not tags:
            # Basic auto-tagging
            keywords = {
                'location': ['location', 'lives in', 'city', 'country', 'home'],
                'birthday': ['birthday', 'born', 'birth'],
                'preference': ['prefer', 'like', 'dislike', 'favorite'],
                'project': ['project', 'work', 'repo', 'path']
            }
            found_tags = []
            fact_lower = fact.lower()
            for tag, keys in keywords.items():
                if any(k in fact_lower for k in keys):
                    found_tags.append(tag)
            if found_tags:
                tags = ",".join(found_tags)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO memories (fact, tags) VALUES (?, ?)", (fact, tags))
            conn.commit()

    def search_memories(self, query: str) -> List[str]:
        """Semantic search via embeddings with keyword fallback."""
        from embed import rank_by_similarity

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fact FROM memories")
            all_facts = [row[0] for row in cursor.fetchall()]

        if not all_facts:
            return []

        ranked = rank_by_similarity(query, all_facts, top_n=5)
        if ranked:
            return ranked

        # Embedding fallback — keyword search
        stop_words = {'what', 'when', 'is', 'the', 'how', 'many', 'of', 'a', 'an', 'and', 'for', 'with', 'about', 'tell', 'me', 'you', 'your', 'my'}
        words = [w.strip().lower() for w in query.replace('?', '').replace('.', '').replace(',', '').split()
                 if w.lower() not in stop_words and len(w) >= 3]
        if not words:
            return []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            fact_scores = {}
            for word in words:
                cursor.execute("SELECT fact FROM memories WHERE fact LIKE ? OR tags LIKE ?", (f"%{word}%", f"%{word}%"))
                for (fact,) in cursor.fetchall():
                    fact_scores[fact] = fact_scores.get(fact, 0) + 1
            if not fact_scores:
                return []
            min_score = 2 if len(words) >= 3 else 1
            filtered = [(f, s) for f, s in fact_scores.items() if s >= min_score]
            if not filtered:
                filtered = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)[:1]
            sorted_facts = sorted(filtered, key=lambda x: (-x[1], len(x[0])))
            return [f[0] for f in sorted_facts[:5]]

    def get_key_facts(self) -> List[str]:
        """Retrieve a focused baseline of critical context."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Restore location as a high-value fact
            cursor.execute("""
                SELECT fact FROM memories 
                WHERE tags LIKE '%key%' OR tags LIKE '%important%' 
                OR tags LIKE '%user_name%' OR tags LIKE '%birthday%'
                OR tags LIKE '%location%' OR tags LIKE '%city%'
                UNION
                SELECT fact FROM (SELECT fact FROM memories ORDER BY created_at DESC LIMIT 5)
            """)
            results = [row[0] for row in cursor.fetchall()]
            return list(set(results))[:10] # Slightly increase to 10 for better baseline

    def delete_memory(self, query: str) -> List[str]:
        """Delete memories matching the query keywords. Returns list of deleted facts."""
        stop_words = {'what', 'when', 'is', 'the', 'how', 'many', 'of', 'a', 'an', 'my', 'your', 'forget', 'remove', 'delete'}
        words = [w.strip().lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2]
        if not words: words = [query.lower()]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Find facts first so we can report what was deleted
            conditions = []
            params = []
            for word in words:
                conditions.append("(fact LIKE ? OR tags LIKE ?)")
                params.extend([f"%{word}%", f"%{word}%"])

            search_sql = f"SELECT id, fact FROM memories WHERE {' OR '.join(conditions)}"
            cursor.execute(search_sql, params)
            to_delete = cursor.fetchall()

            if not to_delete:
                return []

            ids = [row[0] for row in to_delete]
            facts = [row[1] for row in to_delete]

            # Perform deletion
            placeholders = ','.join(['?'] * len(ids))
            cursor.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
            conn.commit()
            return facts

    def get_last_session_id(self) -> Optional[int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None
