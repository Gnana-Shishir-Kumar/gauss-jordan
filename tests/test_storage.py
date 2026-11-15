import pytest

from research_agent import storage


@pytest.fixture
def conn():
    c = storage.connect(":memory:")
    yield c
    c.close()


def test_schema_tables(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "sub_queries", "sources", "evidence", "claims", "claim_citations", "trace_events"} <= names


def test_task_lifecycle(conn):
    tid = storage.create_task(conn, "q?")
    assert storage.get_task(conn, tid)["status"] == "planning"
    storage.set_task_status(conn, tid, "researching")
    storage.finish_task(conn, tid, "done", final_answer="a")
    t = storage.get_task(conn, tid)
    assert t["status"] == "done" and t["final_answer"] == "a" and t["finished_at"]
    with pytest.raises(ValueError):
        storage.set_task_status(conn, tid, "bogus")


def test_source_upsert_is_unique_by_url(conn):
    a = storage.upsert_source(conn, "https://x.com/a", "x.com", None, None, None)
    b = storage.upsert_source(conn, "https://x.com/a", "x.com", "Title", "2025-01-01", "h")
    assert a == b
    row = conn.execute("SELECT title, content_hash FROM sources WHERE id=?", (a,)).fetchone()
    assert row["title"] == "Title" and row["content_hash"] == "h"


def test_claims_citations_verdicts(conn):
    tid = storage.create_task(conn, "q")
    sq = storage.add_sub_query(conn, tid, "sq")
    src = storage.upsert_source(conn, "https://x.com", "x.com", "T", None, "h")
    ev = storage.add_evidence(conn, tid, sq, src, "excerpt", 0.9)
    cid = storage.add_claim(conn, tid, "claim", 0, "fact")
    storage.add_citation(conn, cid, ev)
    storage.add_citation(conn, cid, ev)  # idempotent
    storage.set_verdict(conn, cid, ev, "supported", "ok")
    claims = storage.get_claims(conn, tid)
    assert len(claims) == 1 and len(claims[0]["citations"]) == 1
    assert claims[0]["citations"][0]["verdict"] == "supported"
    assert claims[0]["citations"][0]["url"] == "https://x.com"


def test_trace_dict_detail(conn):
    tid = storage.create_task(conn, "q")
    storage.add_trace(conn, tid, "plan", {"n": 3}, latency_ms=10, input_tokens=5, output_tokens=7)
    tr = storage.get_trace(conn, tid)
    assert tr[0]["detail"] == '{"n": 3}' and tr[0]["output_tokens"] == 7
