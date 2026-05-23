import sqlite3
import json
import pickle
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
            cursor.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, role TEXT, content TEXT, tool_calls TEXT, embedding BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (session_id) REFERENCES sessions (id))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT, tags TEXT, embedding BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, trace TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS routing_history (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, selected_agent TEXT, success BOOLEAN, query_embedding BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
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
            cursor.execute('''CREATE TABLE IF NOT EXISTS actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL,
    tool          TEXT NOT NULL,
    args_json     TEXT NOT NULL,
    summary       TEXT NOT NULL,
    why           TEXT,
    outcome       TEXT NOT NULL,
    error_excerpt TEXT,
    embedding     BLOB,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_actions_session
    ON actions(session_id, created_at)''')
            conn.commit()
            self._migrate()

    def _migrate(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for migration in [
                ("ALTER TABLE memories ADD COLUMN embedding BLOB", "memories"),
                ("ALTER TABLE messages ADD COLUMN embedding BLOB", "messages"),
                ("ALTER TABLE experiences ADD COLUMN embedding BLOB", "experiences"),
            ]:
                try:
                    cursor.execute(migration[0])
                except sqlite3.OperationalError:
                    pass

    def create_session(self, name: Optional[str] = None) -> int:
        if not name: name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
            return cursor.lastrowid

    def add_message(self, session_id: int, role: str, content: str, tool_calls: Optional[List[Dict]] = None, embedding: Optional[bytes] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            cursor.execute("INSERT INTO messages (session_id, role, content, tool_calls, embedding) VALUES (?, ?, ?, ?, ?)",
                           (session_id, role, content, tool_calls_json, embedding))
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

        from embed import embed
        try:
            vector = embed(fact)
            blob = pickle.dumps(vector)
        except Exception:
            blob = None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO memories (fact, tags, embedding) VALUES (?, ?, ?)", (fact, tags, blob))
            conn.commit()

    def _ensure_embedding(self, row_id: int, fact: str, emb_blob: Optional[bytes]):
        if emb_blob:
            return pickle.loads(emb_blob)
        from embed import embed as _embed
        f_vec = _embed(fact)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE memories SET embedding=? WHERE id=?", (pickle.dumps(f_vec), row_id))
            conn.commit()
        return f_vec

    def search_memories(self, query: str, threshold: float = 0.25) -> List[str]:
        from embed import embed, cosine_similarity

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, fact, embedding FROM memories")
            rows = cursor.fetchall()

        if not rows:
            return []

        q_vec = embed(query)
        scored = []
        for row_id, fact, emb_blob in rows:
            f_vec = self._ensure_embedding(row_id, fact, emb_blob)
            sim = cosine_similarity(q_vec, f_vec)
            if sim >= threshold:
                scored.append((sim, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:5]]

    def search_memories_hybrid(self, query: str, alpha: float = 0.7, threshold: float = 0.2) -> List[str]:
        from embed import embed, cosine_similarity

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, fact, embedding FROM memories")
            rows = cursor.fetchall()

        if not rows:
            return []

        stop_words = {'what', 'when', 'is', 'the', 'how', 'many', 'of', 'a', 'an', 'and', 'for', 'with', 'about', 'tell', 'me', 'you', 'your', 'my'}
        query_words = [w.lower() for w in query.replace('?', '').replace('.', '').replace(',', '').split()
                       if w.lower() not in stop_words and len(w) >= 3]

        q_vec = embed(query)
        scored = []
        for row_id, fact, emb_blob in rows:
            f_vec = self._ensure_embedding(row_id, fact, emb_blob)
            emb_score = cosine_similarity(q_vec, f_vec)

            bm25_score = sum(1 for w in query_words if w in fact.lower()) / max(len(query_words), 1) if query_words else 0.0

            combined = alpha * emb_score + (1 - alpha) * bm25_score
            if combined >= threshold:
                scored.append((combined, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:5]]

    def get_key_facts(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT fact FROM memories 
                WHERE tags LIKE '%key%' OR tags LIKE '%important%' 
                OR tags LIKE '%user_name%' OR tags LIKE '%birthday%'
                OR tags LIKE '%location%' OR tags LIKE '%city%'
                UNION
                SELECT fact FROM (SELECT fact FROM memories ORDER BY created_at DESC LIMIT 5)
            """)
            results = [row[0] for row in cursor.fetchall()]
            return list(set(results))[:10]

    def delete_memory(self, query: str) -> List[str]:
        stop_words = {'what', 'when', 'is', 'the', 'how', 'many', 'of', 'a', 'an', 'my', 'your', 'forget', 'remove', 'delete'}
        words = [w.strip().lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2]
        if not words: words = [query.lower()]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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

    def add_experience(self, task: str, trace: str):
        from embed import embed
        try:
            vector = embed(task)
            blob = pickle.dumps(vector)
        except Exception:
            blob = None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO experiences (task, trace, embedding) VALUES (?, ?, ?)", (task, trace, blob))
            conn.commit()

    def search_experiences(self, query: str, limit: int = 3, threshold: float = 0.2) -> List[Dict[str, str]]:
        from embed import embed, cosine_similarity
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT task, trace, embedding FROM experiences")
            rows = cursor.fetchall()

        if not rows:
            return []

        q_vec = embed(query)
        scored = []
        for task, trace, emb_blob in rows:
            if emb_blob:
                e_vec = pickle.loads(emb_blob)
                sim = cosine_similarity(q_vec, e_vec)
            else:
                # Fallback to simple keyword match if no embedding
                sim = 0.5 if any(w in task.lower() for w in query.lower().split()) else 0.0
            
            if sim >= threshold:
                scored.append((sim, {"task": task, "trace": trace}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    # ── Routing History ──────────────────────────────────────────────

    def store_routing_decision(self, query: str, agent: str, success: bool):
        from embed import embed
        try:
            vec = embed(query)
            blob = pickle.dumps(vec)
        except Exception:
            blob = None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO routing_history (query, selected_agent, success, query_embedding) VALUES (?, ?, ?, ?)",
                           (query, agent, success, blob))
            conn.commit()

    def search_similar_routing(self, query: str, limit: int = 3) -> List[Dict]:
        from embed import embed, cosine_similarity

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT query, selected_agent, success, query_embedding FROM routing_history ORDER BY created_at DESC LIMIT 100")
            rows = cursor.fetchall()

        if not rows:
            return []

        q_vec = embed(query)
        scored = []
        for past_query, agent, success, emb_blob in rows:
            if emb_blob:
                p_vec = pickle.loads(emb_blob)
                sim = cosine_similarity(q_vec, p_vec)
                scored.append((sim, {"query": past_query, "agent": agent, "success": success}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def get_message_embeddings(self, session_id: int) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, role, content, embedding FROM messages WHERE session_id = ? AND embedding IS NOT NULL ORDER BY created_at ASC", (session_id,))
            return [{"id": r[0], "role": r[1], "content": r[2], "embedding": pickle.loads(r[3]) if r[3] else None} for r in cursor.fetchall()]

    def set_message_embedding(self, msg_id: int, embedding: bytes):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE messages SET embedding=? WHERE id=?", (embedding, msg_id))
            conn.commit()
