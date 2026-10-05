"""SQLite storage for support tickets and assistant answer feedback."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from core.config import SUPPORT_DB_PATH

CATEGORIES = ["General question", "Estimate feedback", "Bug report", "Home loan help", "Privacy", "Other"]
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL,
    transcript TEXT,
    status TEXT NOT NULL DEFAULT 'Open'
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    assistant TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    helpful INTEGER NOT NULL
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_ticket_id(n: int) -> str:
    return f"PL-{n:06d}"


def parse_ticket_id(ticket_id: str) -> int | None:
    m = re.fullmatch(r"\s*(?:PL-?)?0*(\d+)\s*", ticket_id.upper())
    return int(m.group(1)) if m else None


def validate(name: str, email: str, message: str) -> list[str]:
    errors = []
    if not name.strip():
        errors.append("Please enter your name.")
    if not _EMAIL.match(email.strip()):
        errors.append("Please enter a valid email address.")
    if len(message.strip()) < 10:
        errors.append("Please describe the issue in at least 10 characters.")
    return errors


def create_ticket(name: str, email: str, category: str, message: str, transcript: str = "", db_path: Path = SUPPORT_DB_PATH) -> str:
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO tickets (created_at, name, email, category, message, transcript) VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), name.strip(), email.strip(), category, message.strip(), transcript),
        )
        return format_ticket_id(cur.lastrowid)


def get_ticket(ticket_id: str, db_path: Path = SUPPORT_DB_PATH) -> dict | None:
    n = parse_ticket_id(ticket_id)
    if n is None:
        return None
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT id, created_at, category, status, message FROM tickets WHERE id = ?", (n,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["ticket_id"] = format_ticket_id(data.pop("id"))
    return data


def record_feedback(assistant: str, question: str, answer: str, helpful: bool, db_path: Path = SUPPORT_DB_PATH) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO feedback (created_at, assistant, question, answer, helpful) VALUES (?, ?, ?, ?, ?)",
            (_now(), assistant, question, answer, int(helpful)),
        )


def feedback_summary(db_path: Path = SUPPORT_DB_PATH) -> dict:
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(helpful), 0) AS up FROM feedback").fetchone()
    return {"total": row["n"], "helpful": row["up"]}
