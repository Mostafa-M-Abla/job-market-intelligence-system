"""
Phase 1: simple synchronous SQLite connection.
Phase 2 will replace this with SQLAlchemy async + aiosqlite / asyncpg.
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "job_market.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    sql_path = Path(__file__).parent / "migrations" / "001_initial_schema.sql"
    with get_connection() as conn:
        conn.executescript(sql_path.read_text())
