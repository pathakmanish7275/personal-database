"""SQLite-backed chat sessions with optional running summary per session."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .config import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    summary_up_to_msg_id INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

# Migrations for already-created DBs that predate the summary columns.
_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN summary TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE sessions ADD COLUMN summary_up_to_msg_id INTEGER NOT NULL DEFAULT 0",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    return config.data_dir / "sessions.sqlite3"


def _apply_migrations(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "summary" not in cols:
        conn.execute(_MIGRATIONS[0])
    if "summary_up_to_msg_id" not in cols:
        conn.execute(_MIGRATIONS[1])


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class Message:
    role: str
    content: str
    id: int | None = None
    citations: str | None = None
    created_at: str = field(default_factory=_now)


@dataclass
class Session:
    id: str
    title: str
    created_at: str
    updated_at: str
    summary: str = ""
    summary_up_to_msg_id: int = 0


def create_session(title: str = "New chat") -> Session:
    sid = uuid4().hex[:12]
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (sid, title, now, now),
        )
    return Session(id=sid, title=title, created_at=now, updated_at=now)


def list_sessions() -> list[Session]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, created_at, updated_at, summary, summary_up_to_msg_id "
            "FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [Session(**dict(r)) for r in rows]


def get_session(sid: str) -> Session | None:
    with _conn() as c:
        row = c.execute(
            "SELECT id, title, created_at, updated_at, summary, summary_up_to_msg_id "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
    return Session(**dict(row)) if row else None


def delete_session(sid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        c.execute("DELETE FROM sessions WHERE id = ?", (sid,))


def rename_session(sid: str, title: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), sid),
        )


def update_summary(sid: str, summary: str, up_to_msg_id: int) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE sessions SET summary = ?, summary_up_to_msg_id = ?, updated_at = ? "
            "WHERE id = ?",
            (summary, up_to_msg_id, _now(), sid),
        )


def add_message(sid: str, role: str, content: str, citations: str | None = None) -> int:
    now = _now()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO messages (session_id, role, content, citations, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, role, content, citations, now),
        )
        c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
        return int(cur.lastrowid)


def get_messages(sid: str, since_id: int = 0) -> list[Message]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, role, content, citations, created_at FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id",
            (sid, since_id),
        ).fetchall()
    return [Message(**dict(r)) for r in rows]
