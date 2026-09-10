"""
db.py
-----
Purpose : SQLite persistence helper storing forensic audio analysis runs.
Stores filename, duration_sec, fake_ratio, verdict, created_at, and full result JSON.
Includes auto-migration for legacy schema compatibility.
"""

import os
import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "history.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            duration_sec REAL DEFAULT 0.0,
            fake_ratio REAL DEFAULT 0.0,
            verdict TEXT DEFAULT '',
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    # Automatic migration: Add new columns if missing from older table versions
    cursor.execute("PRAGMA table_info(analysis_history)")
    existing_cols = [col[1] for col in cursor.fetchall()]

    if "duration_sec" not in existing_cols:
        cursor.execute("ALTER TABLE analysis_history ADD COLUMN duration_sec REAL DEFAULT 0.0")
    if "fake_ratio" not in existing_cols:
        cursor.execute("ALTER TABLE analysis_history ADD COLUMN fake_ratio REAL DEFAULT 0.0")
    if "verdict" not in existing_cols:
        cursor.execute("ALTER TABLE analysis_history ADD COLUMN verdict TEXT DEFAULT ''")

    conn.commit()
    conn.close()


def save_result(filename: str, result: dict):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    duration_sec = float(result.get("total_duration", 0.0))
    fake_ratio = float(result.get("fake_ratio", 0.0))
    verdict = str(result.get("verdict", ""))
    result_str = json.dumps(result)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    cursor.execute(
        """
        INSERT INTO analysis_history (filename, duration_sec, fake_ratio, verdict, result_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (filename, duration_sec, fake_ratio, verdict, result_str, created_at)
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 20):
    init_db()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    rows = cursor.execute(
        """
        SELECT id, filename, duration_sec, fake_ratio, verdict, result_json, created_at
        FROM analysis_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    conn.close()

    history = []
    for r in rows:
        item = dict(r)
        try:
            item["result"] = json.loads(item["result_json"])
        except Exception:
            item["result"] = {}
        history.append(item)
    return history
