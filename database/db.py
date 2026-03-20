import sqlite3
import os
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
            # faces table: id, embedding (blob), first_seen, last_seen
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    embedding BLOB,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP
                )
            ''')
            # events table: id, face_id, event_type, timestamp, image_path
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

    def add_face(self, face_id, embedding):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO faces (id, embedding, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            ''', (face_id, embedding, now, now))
            conn.commit()

    def update_face_last_seen(self, face_id):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE faces SET last_seen = ? WHERE id = ?
            ''', (now, face_id))
            conn.commit()

    def get_all_faces(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, embedding FROM faces')
            return cursor.fetchall()

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
        """Truncates all tables in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM events')
            cursor.execute('DELETE FROM faces')
            conn.commit()
