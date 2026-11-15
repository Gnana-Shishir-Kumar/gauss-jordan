"""SQLite persistence (stdlib sqlite3). Every write commits immediately so a crash leaves a trace."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, question TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'planning', final_answer TEXT, error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, finished_at TEXT);
-- status: planning|researching|writing|verifying|done|failed|insufficient_evidence
CREATE TABLE IF NOT EXISTS sub_queries (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), text TEXT NOT NULL, round INTEGER DEFAULT 1, status TEXT DEFAULT 'pending', result_count INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sources (id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, domain TEXT, title TEXT, published_date TEXT, content_hash TEXT, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), sub_query_id INTEGER REFERENCES sub_queries(id), source_id INTEGER REFERENCES sources(id), excerpt TEXT NOT NULL, relevance_score REAL, kept BOOLEAN DEFAULT 1, drop_reason TEXT);
CREATE TABLE IF NOT EXISTS claims (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), text TEXT NOT NULL, position INTEGER, claim_type TEXT DEFAULT 'fact');
CREATE TABLE IF NOT EXISTS claim_citations (claim_id INTEGER REFERENCES claims(id), evidence_id INTEGER REFERENCES evidence(id), verdict TEXT, verifier_note TEXT, checked_at TEXT, PRIMARY KEY (claim_id, evidence_id));
CREATE TABLE IF NOT EXISTS trace_events (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), step TEXT NOT NULL, detail TEXT, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""

TASK_STATUSES = {
    "planning", "researching", "writing", "verifying", "done", "failed", "insufficient_evidence",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def connect(path: str = "research.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _insert(conn: sqlite3.Connection, sql: str, params: tuple) -> int:
    cur = conn.execute(sql, params)
    conn.commit()
    return int(cur.lastrowid)


# --- tasks -----------------------------------------------------------------

def create_task(conn: sqlite3.Connection, question: str) -> int:
    return _insert(conn, "INSERT INTO tasks (question) VALUES (?)", (question,))


def set_task_status(conn: sqlite3.Connection, task_id: int, status: str) -> None:
    if status not in TASK_STATUSES:
        raise ValueError(f"unknown status {status!r}")
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()


def finish_task(
    conn: sqlite3.Connection,
    task_id: int,
    status: str,
    final_answer: str | None = None,
    error: str | None = None,
) -> None:
    if status not in TASK_STATUSES:
        raise ValueError(f"unknown status {status!r}")
    conn.execute(
        "UPDATE tasks SET status = ?, final_answer = ?, error = ?, finished_at = ? WHERE id = ?",
        (status, final_answer, error, _now(), task_id),
    )
    conn.commit()


def get_task(conn: sqlite3.Connection, task_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


# --- sub-queries -------------------------------------------------------------

def add_sub_query(conn: sqlite3.Connection, task_id: int, text: str, round_: int = 1) -> int:
    return _insert(
        conn, "INSERT INTO sub_queries (task_id, text, round) VALUES (?, ?, ?)", (task_id, text, round_)
    )


def update_sub_query(
    conn: sqlite3.Connection, sub_query_id: int, status: str, result_count: int | None = None
) -> None:
    if result_count is None:
        conn.execute("UPDATE sub_queries SET status = ? WHERE id = ?", (status, sub_query_id))
    else:
        conn.execute(
            "UPDATE sub_queries SET status = ?, result_count = ? WHERE id = ?",
            (status, result_count, sub_query_id),
        )
    conn.commit()


def get_sub_queries(conn: sqlite3.Connection, task_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM sub_queries WHERE task_id = ? ORDER BY round, id", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# --- sources -----------------------------------------------------------------

def upsert_source(
    conn: sqlite3.Connection,
    url: str,
    domain: str | None,
    title: str | None,
    published_date: str | None,
    content_hash: str | None,
) -> int:
    """Insert a source by normalized URL, or return the existing id (filling blank fields)."""
    row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
    if row:
        conn.execute(
            """UPDATE sources SET
                 title = COALESCE(title, ?), published_date = COALESCE(published_date, ?),
                 content_hash = COALESCE(content_hash, ?), domain = COALESCE(domain, ?)
               WHERE id = ?""",
            (title, published_date, content_hash, domain, row["id"]),
        )
        conn.commit()
        return int(row["id"])
    return _insert(
        conn,
        "INSERT INTO sources (url, domain, title, published_date, content_hash) VALUES (?, ?, ?, ?, ?)",
        (url, domain, title, published_date, content_hash),
    )


# --- evidence ----------------------------------------------------------------

def add_evidence(
    conn: sqlite3.Connection,
    task_id: int,
    sub_query_id: int,
    source_id: int,
    excerpt: str,
    relevance_score: float,
    kept: bool = True,
    drop_reason: str | None = None,
) -> int:
    return _insert(
        conn,
        """INSERT INTO evidence (task_id, sub_query_id, source_id, excerpt, relevance_score, kept, drop_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (task_id, sub_query_id, source_id, excerpt, relevance_score, int(kept), drop_reason),
    )


def get_evidence(conn: sqlite3.Connection, task_id: int, kept_only: bool = False) -> list[dict[str, Any]]:
    sql = """SELECT e.*, s.url, s.domain, s.title, s.published_date, s.content_hash
             FROM evidence e JOIN sources s ON s.id = e.source_id
             WHERE e.task_id = ?"""
    if kept_only:
        sql += " AND e.kept = 1"
    sql += " ORDER BY e.id"
    return [dict(r) for r in conn.execute(sql, (task_id,)).fetchall()]


# --- claims ------------------------------------------------------------------

def add_claim(conn: sqlite3.Connection, task_id: int, text: str, position: int, claim_type: str) -> int:
    return _insert(
        conn,
        "INSERT INTO claims (task_id, text, position, claim_type) VALUES (?, ?, ?, ?)",
        (task_id, text, position, claim_type),
    )


def add_citation(conn: sqlite3.Connection, claim_id: int, evidence_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO claim_citations (claim_id, evidence_id) VALUES (?, ?)",
        (claim_id, evidence_id),
    )
    conn.commit()


def set_verdict(
    conn: sqlite3.Connection, claim_id: int, evidence_id: int, verdict: str, note: str | None
) -> None:
    conn.execute(
        """UPDATE claim_citations SET verdict = ?, verifier_note = ?, checked_at = ?
           WHERE claim_id = ? AND evidence_id = ?""",
        (verdict, note, _now(), claim_id, evidence_id),
    )
    conn.commit()


def get_claims(conn: sqlite3.Connection, task_id: int) -> list[dict[str, Any]]:
    claims = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM claims WHERE task_id = ? ORDER BY position, id", (task_id,)
        ).fetchall()
    ]
    for c in claims:
        rows = conn.execute(
            """SELECT cc.evidence_id, cc.verdict, cc.verifier_note, cc.checked_at,
                      e.excerpt, s.url, s.title, s.domain
               FROM claim_citations cc
               LEFT JOIN evidence e ON e.id = cc.evidence_id
               LEFT JOIN sources s ON s.id = e.source_id
               WHERE cc.claim_id = ? ORDER BY cc.evidence_id""",
            (c["id"],),
        ).fetchall()
        c["citations"] = [dict(r) for r in rows]
    return claims


# --- trace -------------------------------------------------------------------

def add_trace(
    conn: sqlite3.Connection,
    task_id: int,
    step: str,
    detail: str | dict | None = None,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> int:
    if isinstance(detail, (dict, list)):
        detail = json.dumps(detail, ensure_ascii=False, default=str)
    return _insert(
        conn,
        """INSERT INTO trace_events (task_id, step, detail, latency_ms, input_tokens, output_tokens)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, step, detail, latency_ms, input_tokens, output_tokens),
    )


def get_trace(conn: sqlite3.Connection, task_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM trace_events WHERE task_id = ? ORDER BY id", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]
