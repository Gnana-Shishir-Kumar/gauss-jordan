"""SQLite persistence (stdlib sqlite3). Every write commits immediately so a crash leaves a trace
and the UI can poll progress from another connection."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 4

# Sources hold the fetched raw text once (sources.raw_text) and its numbered sentences (sentences table). Within a task a
# source is one evidence row E<id>; its sentences are cited as E<id>-<idx>. Claim citations store the cited sentence
# indices and each verifier gate's result separately (number_gate, id_gate, entail_gate). id_cache keeps resolved
# PMIDs, NCT numbers and DOIs. trace_events.cached_tokens records cached input tokens where the API reports them.
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, question TEXT NOT NULL, brief TEXT, status TEXT NOT NULL DEFAULT 'briefing', final_answer TEXT, stop_reason TEXT, error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, finished_at TEXT,
  cancer_type TEXT, setting TEXT, anchor_date TEXT, answer_json TEXT, model_ids TEXT, prompt_version TEXT, usage TEXT);
-- status: briefing|needs_clarification|awaiting_approval|researching|writing|verifying|done|failed|insufficient_evidence
CREATE TABLE IF NOT EXISTS sub_queries (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), text TEXT NOT NULL, round INTEGER DEFAULT 1, depth TEXT DEFAULT 'basic', status TEXT DEFAULT 'pending', result_count INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sources (id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, domain TEXT, title TEXT, published_date TEXT, content_hash TEXT, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
  trust_tier INTEGER, source_type TEXT, raw_text TEXT, retrieved_at TEXT, source_date TEXT, date_source TEXT, fetched_via TEXT);
CREATE TABLE IF NOT EXISTS sentences (source_id INTEGER REFERENCES sources(id), idx INTEGER NOT NULL, text TEXT NOT NULL, PRIMARY KEY (source_id, idx));
CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), sub_query_id INTEGER REFERENCES sub_queries(id), source_id INTEGER REFERENCES sources(id), excerpt TEXT NOT NULL, relevance_score REAL, kept BOOLEAN DEFAULT 1, drop_reason TEXT,
  evidence_date TEXT, shown TEXT);
CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), sub_query_id INTEGER REFERENCES sub_queries(id), text TEXT NOT NULL, evidence_ids TEXT NOT NULL,
  details TEXT, sentence_ids TEXT);
CREATE TABLE IF NOT EXISTS claims (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), text TEXT NOT NULL, position INTEGER, claim_type TEXT DEFAULT 'fact', revised BOOLEAN DEFAULT 0,
  evidence_level TEXT, peer_reviewed BOOLEAN DEFAULT 0, section TEXT DEFAULT 'new', population TEXT, item_date TEXT,
  verdict TEXT, status TEXT DEFAULT 'active', rejected TEXT, labels TEXT, original_text TEXT, repair_log TEXT);
-- claims.status: active | unverified (kept, flagged) | deleted (removed by repair; kept for the trace)
CREATE TABLE IF NOT EXISTS claim_citations (claim_id INTEGER REFERENCES claims(id), evidence_id INTEGER REFERENCES evidence(id), verdict TEXT, verifier_note TEXT, checked_at TEXT,
  sentence_ids TEXT, number_gate TEXT, id_gate TEXT, entail_gate TEXT, PRIMARY KEY (claim_id, evidence_id));
CREATE TABLE IF NOT EXISTS trace_events (id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), step TEXT NOT NULL, detail TEXT, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER, tavily_credits REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  model_id TEXT, prompt_version TEXT, cached_tokens INTEGER);
CREATE TABLE IF NOT EXISTS search_cache (key TEXT PRIMARY KEY, query TEXT, params TEXT, results TEXT NOT NULL, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS id_cache (kind TEXT NOT NULL, ident TEXT NOT NULL, status TEXT NOT NULL, title TEXT, data TEXT, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (kind, ident));
"""

# Columns added after v4 shipped; connect() adds them to existing v4 databases.
_ADDITIVE_COLUMNS = [
    ("sources", "id_status", "TEXT"),  # confirmed | title_mismatch | not_found | error | skipped | no_identifier
    ("sources", "identifier", "TEXT"),  # e.g. "PMID 39389055", resolved title in id_title
    ("sources", "id_title", "TEXT"),
    ("claims", "change_type", "TEXT"),
    ("claims", "impact", "TEXT"),
    ("claims", "conflict_with", "TEXT"),
]

TASK_STATUSES = {
    "briefing", "needs_clarification", "awaiting_approval", "researching", "writing", "verifying",
    "done", "failed", "insufficient_evidence",
}
TERMINAL_STATUSES = {"done", "failed", "insufficient_evidence", "needs_clarification"}
_TASK_FIELDS = {
    "brief", "status", "final_answer", "stop_reason", "error", "finished_at", "cancer_type", "setting",
    "anchor_date", "answer_json", "model_ids", "prompt_version", "usage",
}
_CLAIM_FIELDS = {"text", "claim_type", "revised", "evidence_level", "peer_reviewed", "section", "population", "item_date",
                 "verdict", "status", "rejected", "labels", "original_text", "repair_log", "change_type", "impact",
                 "conflict_with"}
_JSON_CLAIM_FIELDS = ("rejected", "labels", "repair_log", "conflict_with")
_GATES = ("number_gate", "id_gate", "entail_gate")


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
    if has_tables and version != SCHEMA_VERSION:
        conn.close()
        raise SchemaMismatch(f"{path} uses an older schema (v{version}; this version needs v{SCHEMA_VERSION}, which "
                             "stores sentence-level evidence). Move it aside or delete it; a new one is created.")
    conn.executescript(SCHEMA)
    for table, col, typ in _ADDITIVE_COLUMNS:  # columns added within v4: add if missing, never remove
        if col not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
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
    for k in ("brief", "answer_json", "model_ids", "usage"):
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


# --- sources / sentences / evidence ---------------------------------------------

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


def get_source(conn, source_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    return dict(row) if row else None


def set_source_text(conn, source_id: int, raw_text: str, sentences: list[str], fetched_via: str) -> None:
    """Store the fetched raw text and its numbered sentences (1-based). Replaces any earlier fetch of the source."""
    conn.execute("UPDATE sources SET raw_text = ?, retrieved_at = ?, fetched_via = ? WHERE id = ?",
                 (raw_text, now(), fetched_via, source_id))
    conn.execute("DELETE FROM sentences WHERE source_id = ?", (source_id,))
    conn.executemany("INSERT INTO sentences (source_id, idx, text) VALUES (?, ?, ?)",
                     [(source_id, i, t) for i, t in enumerate(sentences, 1)])
    conn.commit()


def cached_texts(conn, urls: list[str], ttl_days: int) -> dict[str, str]:
    """Raw text already fetched for these URLs within the TTL (this is the extract cache)."""
    if not urls:
        return {}
    rows = conn.execute(
        f"""SELECT url, raw_text FROM sources WHERE url IN ({','.join('?' * len(urls))}) AND raw_text IS NOT NULL
            AND raw_text != '' AND retrieved_at >= datetime('now', ?)""", (*urls, f"-{int(ttl_days)} days")).fetchall()
    return {r[0]: r[1] for r in rows}


def set_source_type(conn, source_id: int, source_type: str) -> None:
    conn.execute("UPDATE sources SET source_type = ? WHERE id = ? AND source_type IS NULL", (source_type, source_id))
    conn.commit()


def set_source_trust(conn, source_id: int, *, trust_tier: int, source_type: str | None = None,
                     id_status: str | None = None, identifier: str | None = None, id_title: str | None = None) -> None:
    conn.execute("""UPDATE sources SET trust_tier = ?, source_type = COALESCE(?, source_type), id_status = ?,
                    identifier = ?, id_title = ? WHERE id = ?""",
                 (trust_tier, source_type, id_status, identifier, id_title, source_id))
    conn.commit()


def set_source_date(conn, source_id: int, source_date: str, date_source: str) -> None:
    conn.execute("UPDATE sources SET source_date = ?, date_source = ? WHERE id = ?", (source_date, date_source, source_id))
    conn.commit()


def get_sentences(conn, source_id: int, idxs: list[int] | None = None) -> dict[int, str]:
    if idxs is not None:
        if not idxs:
            return {}
        rows = conn.execute(
            f"SELECT idx, text FROM sentences WHERE source_id = ? AND idx IN ({','.join('?' * len(idxs))})",
            (source_id, *idxs))
    else:
        rows = conn.execute("SELECT idx, text FROM sentences WHERE source_id = ? ORDER BY idx", (source_id,))
    return {int(r[0]): r[1] for r in rows.fetchall()}


def add_evidence(conn, task_id: int, sub_query_id: int, source_id: int, excerpt: str, relevance_score: float,
                 kept: bool = True, drop_reason: str | None = None, evidence_date: str | None = None,
                 shown: list[int] | None = None) -> int:
    return _insert(conn, """INSERT INTO evidence (task_id, sub_query_id, source_id, excerpt, relevance_score, kept,
                              drop_reason, evidence_date, shown) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (task_id, sub_query_id, source_id, excerpt, relevance_score, int(kept), drop_reason, evidence_date,
                    _dumps(shown) if shown is not None else None))


def set_evidence_date(conn, evidence_id: int, evidence_date: str) -> None:
    conn.execute("UPDATE evidence SET evidence_date = ? WHERE id = ? AND evidence_date IS NULL",
                 (evidence_date, evidence_id))
    conn.commit()


def get_evidence(conn, task_id: int, kept_only: bool = False, ids: list[int] | None = None) -> list[dict]:
    sql = """SELECT e.*, s.url, s.domain, s.title, s.published_date, s.content_hash, s.trust_tier, s.source_type,
                    s.source_date, s.date_source, s.retrieved_at, s.fetched_via, s.id_status, s.identifier, s.id_title
             FROM evidence e JOIN sources s ON s.id = e.source_id WHERE e.task_id = ?"""
    params: list[Any] = [task_id]
    if kept_only:
        sql += " AND e.kept = 1"
    if ids is not None:
        if not ids:
            return []
        sql += f" AND e.id IN ({','.join('?' * len(ids))})"
        params += list(ids)
    out = []
    for r in conn.execute(sql + " ORDER BY e.id", params).fetchall():
        d = dict(r)
        d["shown"] = json.loads(d["shown"]) if d.get("shown") else []
        out.append(d)
    return out


def evidence_sentences(conn, task_id: int, evidence_id: int, idxs: list[int] | None = None) -> dict[int, str]:
    """Sentences of one kept evidence row of this task (empty if the row is not kept evidence of the task)."""
    row = conn.execute("SELECT source_id FROM evidence WHERE id = ? AND task_id = ? AND kept = 1",
                       (evidence_id, task_id)).fetchone()
    return get_sentences(conn, int(row[0]), idxs) if row else {}


def kept_sentence_total(conn, task_id: int) -> int:
    """Distinct sentence IDs available in the task's kept evidence (for the no_new_evidence stop rule)."""
    r = conn.execute("""SELECT COUNT(*) FROM evidence e JOIN sentences s ON s.source_id = e.source_id
                        WHERE e.task_id = ? AND e.kept = 1""", (task_id,)).fetchone()
    return int(r[0])


def task_urls(conn, task_id: int) -> set[str]:
    rows = conn.execute("""SELECT DISTINCT s.url FROM evidence e JOIN sources s ON s.id = e.source_id
                           WHERE e.task_id = ? AND (e.kept = 1 OR e.drop_reason LIKE 'source_unavailable%')""",
                        (task_id,))
    return {r[0] for r in rows.fetchall()}


# --- notes ---------------------------------------------------------------------

def add_note(conn, task_id: int, sub_query_id: int | None, text: str, evidence_ids: list[int], details: dict,
             sentence_ids: list[str] | None = None) -> int:
    return _insert(conn, """INSERT INTO notes (task_id, sub_query_id, text, evidence_ids, details, sentence_ids)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                   (task_id, sub_query_id, text, json.dumps(evidence_ids), _dumps(details),
                    json.dumps(sentence_ids or [])))


def get_notes(conn, task_id: int) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM notes WHERE task_id = ? ORDER BY id", (task_id,)).fetchall():
        d = dict(r)
        d["evidence_ids"] = json.loads(d["evidence_ids"])
        d["details"] = json.loads(d["details"]) if d.get("details") else {}
        d["sentence_ids"] = json.loads(d["sentence_ids"]) if d.get("sentence_ids") else []
        out.append(d)
    return out


# --- claims --------------------------------------------------------------------

def add_claim(conn, task_id: int, text: str, position: int, claim_type: str, *, section: str = "new",
              evidence_level: str | None = None, population: str | None = None, item_date: str | None = None,
              rejected: list | None = None, labels: list | None = None, change_type: str | None = None,
              impact: str | None = None) -> int:
    return _insert(conn, """INSERT INTO claims (task_id, text, position, claim_type, section, evidence_level,
                              population, item_date, rejected, labels, original_text, change_type, impact)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (task_id, text, position, claim_type, section, evidence_level, population, item_date,
                    _dumps(rejected or []), _dumps(labels or []), text, change_type, impact))


def update_claim(conn, claim_id: int, **fields: Any) -> None:
    bad = set(fields) - _CLAIM_FIELDS
    if bad:
        raise ValueError(f"unknown claim fields {bad}")
    if fields:
        vals = [_dumps(v) if k in _JSON_CLAIM_FIELDS else v for k, v in fields.items()]
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(f"UPDATE claims SET {cols} WHERE id = ?", (*vals, claim_id))
        conn.commit()


def append_repair_log(conn, claim_id: int, entry: dict) -> None:
    row = conn.execute("SELECT repair_log FROM claims WHERE id = ?", (claim_id,)).fetchone()
    log = json.loads(row[0]) if row and row[0] else []
    log.append(entry)
    update_claim(conn, claim_id, repair_log=log)


def add_citation(conn, claim_id: int, evidence_id: int, sentence_ids: list[int]) -> None:
    conn.execute("INSERT OR IGNORE INTO claim_citations (claim_id, evidence_id, sentence_ids) VALUES (?, ?, ?)",
                 (claim_id, evidence_id, json.dumps(sorted(set(sentence_ids)))))
    conn.commit()


def delete_claims(conn, task_id: int) -> None:
    """Remove a task's claims (used when a failed run is resumed from its stored research)."""
    conn.execute("DELETE FROM claim_citations WHERE claim_id IN (SELECT id FROM claims WHERE task_id = ?)", (task_id,))
    conn.execute("DELETE FROM claims WHERE task_id = ?", (task_id,))
    conn.commit()


def delete_citations(conn, claim_id: int) -> None:
    conn.execute("DELETE FROM claim_citations WHERE claim_id = ?", (claim_id,))
    conn.commit()


def set_gates(conn, claim_id: int, evidence_id: int, *, verdict: str | None = None, note: str | None = None,
              **gates: dict | None) -> None:
    """Store one citation's gate results (number_gate, id_gate, entail_gate) and, optionally, its verdict."""
    bad = set(gates) - set(_GATES)
    if bad:
        raise ValueError(f"unknown gates {bad}")
    fields: dict[str, Any] = {k: _dumps(v) for k, v in gates.items()}
    if verdict is not None:
        fields.update(verdict=verdict, verifier_note=note, checked_at=now())
    if fields:
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(f"UPDATE claim_citations SET {cols} WHERE claim_id = ? AND evidence_id = ?",
                     (*fields.values(), claim_id, evidence_id))
        conn.commit()


def get_claims(conn, task_id: int, ids: list[int] | None = None, include_deleted: bool = True) -> list[dict]:
    sql, params = "SELECT * FROM claims WHERE task_id = ?", [task_id]
    if not include_deleted:
        sql += " AND COALESCE(status, 'active') != 'deleted'"
    if ids is not None:
        if not ids:
            return []
        sql += f" AND id IN ({','.join('?' * len(ids))})"
        params += list(ids)
    claims = [dict(r) for r in conn.execute(sql + " ORDER BY position, id", params).fetchall()]
    for c in claims:
        for k in _JSON_CLAIM_FIELDS:
            c[k] = json.loads(c[k]) if c.get(k) else []
        c["status"] = c.get("status") or "active"
        rows = conn.execute(
            """SELECT cc.*, e.excerpt, e.kept, e.drop_reason, e.evidence_date, e.source_id, s.url, s.title, s.domain,
                      s.trust_tier, s.source_type, s.source_date, s.date_source, s.id_status, s.identifier
               FROM claim_citations cc LEFT JOIN evidence e ON e.id = cc.evidence_id
               LEFT JOIN sources s ON s.id = e.source_id WHERE cc.claim_id = ? ORDER BY cc.evidence_id""",
            (c["id"],)).fetchall()
        cits = []
        for r in rows:
            x = dict(r)
            x["sentence_ids"] = json.loads(x["sentence_ids"]) if x.get("sentence_ids") else []
            for g in _GATES:
                x[g] = json.loads(x[g]) if x.get(g) else None
            sents = get_sentences(conn, x["source_id"], x["sentence_ids"]) if x.get("source_id") else {}
            x["sentences"] = [{"id": f"E{x['evidence_id']}-{i}", "idx": i, "text": sents[i]}
                              for i in x["sentence_ids"] if i in sents]
            cits.append(x)
        c["citations"] = cits
    return claims


# --- trace ---------------------------------------------------------------------

def add_trace(conn, task_id: int, step: str, detail: Any = None, *, latency_ms: int | None = None,
              input_tokens: int | None = None, output_tokens: int | None = None, tavily_credits: float | None = None,
              model_id: str | None = None, prompt_version: str | None = None,
              cached_tokens: int | None = None) -> int:
    return _insert(conn, """INSERT INTO trace_events (task_id, step, detail, latency_ms, input_tokens, output_tokens,
                              tavily_credits, model_id, prompt_version, cached_tokens)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (task_id, step, _dumps(detail), latency_ms, input_tokens, output_tokens, tavily_credits,
                    model_id, prompt_version, cached_tokens))


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
                               COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cached_tokens), 0)
                        FROM trace_events WHERE task_id = ?""", (task_id,)).fetchone()
    return {"credits": float(r[0]), "input_tokens": int(r[1]), "output_tokens": int(r[2]), "cached_tokens": int(r[3])}


def step_summary(conn, task_id: int) -> list[dict]:
    """Per-step totals: LLM calls, input/output tokens, cached share (where reported), Tavily credits, seconds."""
    rows = conn.execute("""SELECT step, COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
                                  COALESCE(SUM(cached_tokens), 0), SUM(cached_tokens IS NOT NULL),
                                  COALESCE(SUM(tavily_credits), 0), COALESCE(SUM(latency_ms), 0),
                                  SUM(COALESCE(input_tokens, 0) > 0)
                           FROM trace_events WHERE task_id = ? GROUP BY step ORDER BY MIN(id)""", (task_id,))
    out = []
    for step, tin, tout, cached, cached_reported, credits, ms, llm_calls in rows.fetchall():
        if not (tin or tout or credits or ms):
            continue
        out.append({"step": step, "llm_calls": int(llm_calls or 0), "input_tokens": int(tin),
                    "output_tokens": int(tout),
                    "cached_share": round(cached / tin, 3) if cached_reported and tin else None,
                    "credits": float(credits), "seconds": round(ms / 1000, 1)})
    return out


# --- cache ---------------------------------------------------------------------

def get_search_cache(conn, key: str, ttl_days: int) -> list[dict] | None:
    row = conn.execute("SELECT results FROM search_cache WHERE key = ? AND fetched_at >= datetime('now', ?)",
                       (key, f"-{int(ttl_days)} days")).fetchone()
    return json.loads(row[0]) if row else None


def put_search_cache(conn, key: str, query: str, params: dict, results: list[dict]) -> None:
    conn.execute("INSERT OR REPLACE INTO search_cache (key, query, params, results, fetched_at) VALUES (?, ?, ?, ?, ?)",
                 (key, query, _dumps(params), _dumps(results), now()))
    conn.commit()


def get_id_cache(conn, kind: str, ident: str, ttl_days: int) -> dict | None:
    row = conn.execute("""SELECT status, title, data FROM id_cache WHERE kind = ? AND ident = ?
                          AND fetched_at >= datetime('now', ?)""", (kind, ident, f"-{int(ttl_days)} days")).fetchone()
    if not row:
        return None
    return {"status": row[0], "title": row[1], "data": json.loads(row[2]) if row[2] else {}}


def put_id_cache(conn, kind: str, ident: str, status: str, title: str | None, data: dict | None) -> None:
    conn.execute("""INSERT OR REPLACE INTO id_cache (kind, ident, status, title, data, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?)""", (kind, ident, status, title, _dumps(data or {}), now()))
    conn.commit()
