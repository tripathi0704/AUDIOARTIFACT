"""
db.py
-----
Purpose : Minimal SQLite helper to store every analysis result as history.
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "history.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_result(filename: str, result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analysis_history (filename, result_json, created_at) VALUES (?, ?, ?)",
        (filename, json.dumps(result), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, filename, result_json, created_at FROM analysis_history ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
