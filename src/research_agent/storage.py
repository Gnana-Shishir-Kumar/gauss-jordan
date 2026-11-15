"""SQLite persistence (stdlib sqlite3). Every write commits immediately so a crash leaves a trace
and the UI can poll progress from another connection."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 3

# The specified schema, plus: oncology columns (tasks.cancer_type/setting/anchor_date, sources.trust_tier/source_type,
# evidence.evidence_date, claims.evidence_level/peer_reviewed), governance columns (model_ids, prompt_version),
# answer structure (tasks.answer_json, claims.section/population/item_date, notes.details,
# claim_citations.source_sentence) and two cache tables.
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, question TEXT NOT NULL, brief TEXT, status TEXT NOT NULL DEFAULT 'briefing', final_answer TEXT, stop_reason TEXT, error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, finished_at TEXT,
  cancer_type TEXT, setting TEXT, anchor_date TEXT, answer_json TEXT, model_ids TEXT, prompt_version TEXT);
-- status: briefing|needs_clarification|awaiting_approval|researching|writing|verifying|done|failed|insufficient_evidence
CREATE TABLE IF NOT EXISTS sub_queries (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), text TEXT NOT NULL, round INTEGER DEFAULT 1, depth TEXT DEFAULT 'basic', status TEXT DEFAULT 'pending', result_count INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sources (id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, domain TEXT, title TEXT, published_date TEXT, content_hash TEXT, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
  trust_tier INTEGER, source_type TEXT);
CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), sub_query_id INTEGER REFERENCES sub_queries(id), source_id INTEGER REFERENCES sources(id), excerpt TEXT NOT NULL, relevance_score REAL, kept BOOLEAN DEFAULT 1, drop_reason TEXT,
  evidence_date TEXT);
CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), sub_query_id INTEGER REFERENCES sub_queries(id), text TEXT NOT NULL, evidence_ids TEXT NOT NULL,
  details TEXT);
CREATE TABLE IF NOT EXISTS claims (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), text TEXT NOT NULL, position INTEGER, claim_type TEXT DEFAULT 'fact', revised BOOLEAN DEFAULT 0,
  evidence_level TEXT, peer_reviewed BOOLEAN DEFAULT 0, section TEXT DEFAULT 'new', population TEXT, item_date TEXT);
CREATE TABLE IF NOT EXISTS claim_citations (claim_id INTEGER REFERENCES claims(id), evidence_id INTEGER REFERENCES evidence(id), verdict TEXT, verifier_note TEXT, checked_at TEXT,
  source_sentence TEXT, PRIMARY KEY (claim_id, evidence_id));
CREATE TABLE IF NOT EXISTS trace_events (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), step TEXT NOT NULL, detail TEXT, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER, tavily_credits REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  model_id TEXT, prompt_version TEXT);
CREATE TABLE IF NOT EXISTS search_cache (key TEXT PRIMARY KEY, query TEXT, params TEXT, results TEXT NOT NULL, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS extract_cache (url TEXT NOT NULL, query_key TEXT NOT NULL DEFAULT '', content TEXT NOT NULL, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (url, query_key));
"""

TASK_STATUSES = {
    "briefing", "needs_clarification", "awaiting_approval", "researching", "writing", "verifying",
    "done", "failed", "insufficient_evidence",
}
TERMINAL_STATUSES = {"done", "failed", "insufficient_evidence", "needs_clarification"}
_TASK_FIELDS = {
    "brief", "status", "final_answer", "stop_reason", "error", "finished_at", "cancer_type", "setting",
    "anchor_date", "answer_json", "model_ids", "prompt_version",
}
_CLAIM_FIELDS = {"text", "claim_type", "revised", "evidence_level", "peer_reviewed", "section", "population", "item_date"}


class SchemaMismatch(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def connect(path: str = "research.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    has_tables = conn.execute("SELECT count(*) FROM sqlite_master WHERE name='tasks'").fetchone()[0]
    if has_tables and version == 2:
        # v2 -> v3 only changes the extract cache key (URL -> URL + query). It is a cache, so drop and rebuild it.
        conn.execute("DROP TABLE IF EXISTS extract_cache")
        version = SCHEMA_VERSION
    if has_tables and version != SCHEMA_VERSION:
        conn.close()
        raise SchemaMismatch(f"{path} uses an older schema (v{version}); move it aside or delete it.")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def _insert(conn: sqlite3.Connection, sql: str, params: tuple) -> int:
    cur = conn.execute(sql, params)
    conn.commit()
    return int(cur.lastrowid)


def _dumps(v: Any) -> str | None:
    if v is None or isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, default=str)


# --- tasks ---------------------------------------------------------------------

def create_task(conn, question: str, model_ids: dict | None = None, prompt_version: str | None = None) -> int:
    return _insert(conn, "INSERT INTO tasks (question, model_ids, prompt_version) VALUES (?, ?, ?)",
                   (question, _dumps(model_ids), prompt_version))


def update_task(conn, task_id: int, **fields: Any) -> None:
    bad = set(fields) - _TASK_FIELDS
    if bad:
        raise ValueError(f"unknown task fields {bad}")
    if "status" in fields and fields["status"] not in TASK_STATUSES:
        raise ValueError(f"unknown status {fields['status']!r}")
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE tasks SET {cols} WHERE id = ?", (*(_dumps(v) for v in fields.values()), task_id))
    conn.commit()


def finish_task(conn, task_id: int, status: str, **fields: Any) -> None:
    update_task(conn, task_id, status=status, finished_at=now(), **fields)


def get_task(conn, task_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if not row:
        return None
    t = dict(row)
    for k in ("brief", "answer_json", "model_ids"):
        if t.get(k):
            try:
                t[k] = json.loads(t[k])
            except ValueError:
                pass
    return t


def recent_tasks(conn, limit: int = 20) -> list[dict]:
    rows = conn.execute("SELECT id, question, status, created_at FROM tasks ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows.fetchall()]


# --- sub-queries ---------------------------------------------------------------

def add_sub_query(conn, task_id: int, text: str, round_: int = 1, depth: str = "basic", status: str = "pending") -> int:
    return _insert(conn, "INSERT INTO sub_queries (task_id, text, round, depth, status) VALUES (?, ?, ?, ?, ?)",
                   (task_id, text, round_, depth, status))


def update_sub_query(conn, sub_query_id: int, status: str, result_count: int | None = None) -> None:
    if result_count is None:
        conn.execute("UPDATE sub_queries SET status = ? WHERE id = ?", (status, sub_query_id))
    else:
        conn.execute("UPDATE sub_queries SET status = ?, result_count = ? WHERE id = ?",
                     (status, result_count, sub_query_id))
    conn.commit()


def get_sub_queries(conn, task_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM sub_queries WHERE task_id = ? ORDER BY round, id", (task_id,))
    return [dict(r) for r in rows.fetchall()]


# --- sources / evidence --------------------------------------------------------

def upsert_source(conn, url: str, domain: str | None, title: str | None, published_date: str | None,
                  content_hash: str | None, trust_tier: int | None, source_type: str | None) -> int:
    row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
    if row:
        conn.execute(
            """UPDATE sources SET title = COALESCE(title, ?), published_date = COALESCE(published_date, ?),
                 content_hash = COALESCE(?, content_hash), domain = COALESCE(domain, ?),
                 trust_tier = COALESCE(trust_tier, ?), source_type = COALESCE(source_type, ?)
               WHERE id = ?""",
            (title, published_date, content_hash, domain, trust_tier, source_type, row["id"]),
        )
        conn.commit()
        return int(row["id"])
    return _insert(conn, """INSERT INTO sources (url, domain, title, published_date, content_hash, trust_tier, source_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                   (url, domain, title, published_date, content_hash, trust_tier, source_type))


def set_source_type(conn, source_id: int, source_type: str) -> None:
    conn.execute("UPDATE sources SET source_type = ? WHERE id = ? AND source_type IS NULL", (source_type, source_id))
    conn.commit()


def add_evidence(conn, task_id: int, sub_query_id: int, source_id: int, excerpt: str, relevance_score: float,
                 kept: bool = True, drop_reason: str | None = None, evidence_date: str | None = None) -> int:
    return _insert(conn, """INSERT INTO evidence (task_id, sub_query_id, source_id, excerpt, relevance_score, kept,
                              drop_reason, evidence_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                   (task_id, sub_query_id, source_id, excerpt, relevance_score, int(kept), drop_reason, evidence_date))


def set_evidence_date(conn, evidence_id: int, evidence_date: str) -> None:
    conn.execute("UPDATE evidence SET evidence_date = ? WHERE id = ? AND evidence_date IS NULL",
                 (evidence_date, evidence_id))
    conn.commit()


def get_evidence(conn, task_id: int, kept_only: bool = False, ids: list[int] | None = None) -> list[dict]:
    sql = """SELECT e.*, s.url, s.domain, s.title, s.published_date, s.content_hash, s.trust_tier, s.source_type
             FROM evidence e JOIN sources s ON s.id = e.source_id WHERE e.task_id = ?"""
    params: list[Any] = [task_id]
    if kept_only:
        sql += " AND e.kept = 1"
    if ids is not None:
        if not ids:
            return []
        sql += f" AND e.id IN ({','.join('?' * len(ids))})"
        params += list(ids)
    return [dict(r) for r in conn.execute(sql + " ORDER BY e.id", params).fetchall()]


def task_urls(conn, task_id: int) -> set[str]:
    rows = conn.execute("""SELECT DISTINCT s.url FROM evidence e JOIN sources s ON s.id = e.source_id
                           WHERE e.task_id = ? AND (e.kept = 1 OR e.drop_reason LIKE 'source_unavailable%')""",
                        (task_id,))
    return {r[0] for r in rows.fetchall()}


# --- notes ---------------------------------------------------------------------

def add_note(conn, task_id: int, sub_query_id: int | None, text: str, evidence_ids: list[int], details: dict) -> int:
    return _insert(conn, "INSERT INTO notes (task_id, sub_query_id, text, evidence_ids, details) VALUES (?, ?, ?, ?, ?)",
                   (task_id, sub_query_id, text, json.dumps(evidence_ids), _dumps(details)))


def get_notes(conn, task_id: int) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM notes WHERE task_id = ? ORDER BY id", (task_id,)).fetchall():
        d = dict(r)
        d["evidence_ids"] = json.loads(d["evidence_ids"])
        d["details"] = json.loads(d["details"]) if d.get("details") else {}
        out.append(d)
    return out


# --- claims --------------------------------------------------------------------

def add_claim(conn, task_id: int, text: str, position: int, claim_type: str, *, section: str = "new",
              evidence_level: str | None = None, population: str | None = None, item_date: str | None = None) -> int:
    return _insert(conn, """INSERT INTO claims (task_id, text, position, claim_type, section, evidence_level,
                              population, item_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                   (task_id, text, position, claim_type, section, evidence_level, population, item_date))


def update_claim(conn, claim_id: int, **fields: Any) -> None:
    bad = set(fields) - _CLAIM_FIELDS
    if bad:
        raise ValueError(f"unknown claim fields {bad}")
    if fields:
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(f"UPDATE claims SET {cols} WHERE id = ?", (*fields.values(), claim_id))
        conn.commit()


def add_citation(conn, claim_id: int, evidence_id: int, source_sentence: str | None = None) -> None:
    conn.execute("INSERT OR IGNORE INTO claim_citations (claim_id, evidence_id, source_sentence) VALUES (?, ?, ?)",
                 (claim_id, evidence_id, source_sentence))
    conn.commit()


def delete_citations(conn, claim_id: int) -> None:
    conn.execute("DELETE FROM claim_citations WHERE claim_id = ?", (claim_id,))
    conn.commit()


def set_verdict(conn, claim_id: int, evidence_id: int, verdict: str, note: str | None) -> None:
    conn.execute("""UPDATE claim_citations SET verdict = ?, verifier_note = ?, checked_at = ?
                    WHERE claim_id = ? AND evidence_id = ?""", (verdict, note, now(), claim_id, evidence_id))
    conn.commit()


def get_claims(conn, task_id: int, ids: list[int] | None = None) -> list[dict]:
    sql, params = "SELECT * FROM claims WHERE task_id = ?", [task_id]
    if ids is not None:
        if not ids:
            return []
        sql += f" AND id IN ({','.join('?' * len(ids))})"
        params += list(ids)
    claims = [dict(r) for r in conn.execute(sql + " ORDER BY position, id", params).fetchall()]
    for c in claims:
        rows = conn.execute(
            """SELECT cc.*, e.excerpt, e.kept, e.drop_reason, e.evidence_date, s.url, s.title, s.domain,
                      s.trust_tier, s.source_type
               FROM claim_citations cc LEFT JOIN evidence e ON e.id = cc.evidence_id
               LEFT JOIN sources s ON s.id = e.source_id WHERE cc.claim_id = ? ORDER BY cc.evidence_id""",
            (c["id"],)).fetchall()
        c["citations"] = [dict(r) for r in rows]
    return claims


# --- trace ---------------------------------------------------------------------

def add_trace(conn, task_id: int, step: str, detail: Any = None, *, latency_ms: int | None = None,
              input_tokens: int | None = None, output_tokens: int | None = None, tavily_credits: float | None = None,
              model_id: str | None = None, prompt_version: str | None = None) -> int:
    return _insert(conn, """INSERT INTO trace_events (task_id, step, detail, latency_ms, input_tokens, output_tokens,
                              tavily_credits, model_id, prompt_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (task_id, step, _dumps(detail), latency_ms, input_tokens, output_tokens, tavily_credits,
                    model_id, prompt_version))


def get_trace(conn, task_id: int) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM trace_events WHERE task_id = ? ORDER BY id", (task_id,)).fetchall():
        t = dict(r)
        try:
            d = json.loads(t["detail"]) if t["detail"] else {}
        except ValueError:
            d = {"message": t["detail"]}
        if not isinstance(d, dict):
            d = {"message": str(d)}
        t["detail"] = d
        t["message"] = d.get("message", "")
        t["level"] = d.get("level", "info")
        out.append(t)
    return out


def task_totals(conn, task_id: int) -> dict[str, float]:
    r = conn.execute("""SELECT COALESCE(SUM(tavily_credits), 0), COALESCE(SUM(input_tokens), 0),
                               COALESCE(SUM(output_tokens), 0) FROM trace_events WHERE task_id = ?""",
                     (task_id,)).fetchone()
    return {"credits": float(r[0]), "input_tokens": int(r[1]), "output_tokens": int(r[2])}


# --- cache ---------------------------------------------------------------------

def get_search_cache(conn, key: str, ttl_days: int) -> list[dict] | None:
    row = conn.execute("SELECT results FROM search_cache WHERE key = ? AND fetched_at >= datetime('now', ?)",
                       (key, f"-{int(ttl_days)} days")).fetchone()
    return json.loads(row[0]) if row else None


def put_search_cache(conn, key: str, query: str, params: dict, results: list[dict]) -> None:
    conn.execute("INSERT OR REPLACE INTO search_cache (key, query, params, results, fetched_at) VALUES (?, ?, ?, ?, ?)",
                 (key, query, _dumps(params), _dumps(results), now()))
    conn.commit()


def get_extract_cache(conn, urls: list[str], query_key: str, ttl_days: int) -> dict[str, str]:
    if not urls:
        return {}
    rows = conn.execute(
        f"""SELECT url, content FROM extract_cache WHERE url IN ({','.join('?' * len(urls))}) AND query_key = ?
            AND fetched_at >= datetime('now', ?)""", (*urls, query_key, f"-{int(ttl_days)} days")).fetchall()
    return {r[0]: r[1] for r in rows}


def put_extract_cache(conn, pages: dict[str, str], query_key: str) -> None:
    ts = now()
    conn.executemany("INSERT OR REPLACE INTO extract_cache (url, query_key, content, fetched_at) VALUES (?, ?, ?, ?)",
                     [(u, query_key, c, ts) for u, c in pages.items()])
    conn.commit()
