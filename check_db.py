import sqlite3
import os

db_path = "data.db"
if not os.path.exists(db_path):
    print("Database not found!")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM faces")
    print(f"Faces count: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM events")
    print(f"Events count: {cursor.fetchone()[0]}")
    conn.close()
