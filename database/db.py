import sqlite3
import os
import numpy as np
from datetime import datetime

class Database:
    def __init__(self, db_path="data.db"):
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT,
                    embedding BLOB,
                    FOREIGN KEY (face_id) REFERENCES faces (id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT,
                    event_type TEXT,
                    timestamp TIMESTAMP,
                    image_path TEXT,
                    FOREIGN KEY (face_id) REFERENCES faces (id)
                )
            ''')
            conn.commit()

    def add_face(self, face_id, embedding_bytes):
        """Adds a new face and its first embedding."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO faces (id, first_seen, last_seen)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen
            ''', (face_id, now, now))
            
            # Check if this precise embedding already exists (deduplication)
            cursor.execute("SELECT 1 FROM face_embeddings WHERE face_id = ? AND embedding = ?", (face_id, embedding_bytes))
            if not cursor.fetchone():
                cursor.execute('INSERT INTO face_embeddings (face_id, embedding) VALUES (?, ?)', (face_id, embedding_bytes))
            conn.commit()

    def add_embedding(self, face_id, embedding_bytes):
        """Adds an additional embedding to an existing face's gallery if it's new."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM face_embeddings WHERE face_id = ? AND embedding = ?", (face_id, embedding_bytes))
            if cursor.fetchone():
                return
            cursor.execute('INSERT INTO face_embeddings (face_id, embedding) VALUES (?, ?)', (face_id, embedding_bytes))
            conn.commit()

    def update_face_last_seen(self, face_id):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE faces SET last_seen = ? WHERE id = ?', (now, face_id))
            conn.commit()

    def get_all_faces(self):
        """Returns a dict of face_id -> list of embeddings (numpy arrays)."""
        if not os.path.exists(self.db_path): return {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT face_id, embedding FROM face_embeddings')
            rows = cursor.fetchall()
            gallery = {}
            for face_id, emb_blob in rows:
                if face_id not in gallery:
                    gallery[face_id] = []
                gallery[face_id].append(np.frombuffer(emb_blob, dtype=np.float32))
            return gallery

    def log_event(self, face_id, event_type, image_path):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (face_id, event_type, timestamp, image_path)
                VALUES (?, ?, ?, ?)
            ''', (face_id, event_type, now, image_path))
            conn.commit()

    def get_unique_visitor_count(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM faces')
            return cursor.fetchone()[0]

    def clear_data(self):
        """Truncates all tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM events')
            cursor.execute('DELETE FROM face_embeddings')
            cursor.execute('DELETE FROM faces')
            conn.commit()
