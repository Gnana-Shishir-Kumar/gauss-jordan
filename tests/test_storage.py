import sqlite3

import pytest

from research_agent import storage


@pytest.fixture
def conn():
    c = storage.connect(":memory:")
    yield c
    c.close()


def test_schema_tables_and_columns(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "sub_queries", "sources", "evidence", "notes", "claims", "claim_citations", "trace_events",
            "search_cache", "sentences", "id_cache"} <= names

    def cols(t):
        return {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}

    assert {"brief", "stop_reason", "cancer_type", "setting", "anchor_date", "model_ids", "prompt_version"} <= cols("tasks")
    assert {"trust_tier", "source_type", "raw_text", "retrieved_at", "source_date"} <= cols("sources")
    assert {"evidence_date", "shown"} <= cols("evidence")
    assert {"evidence_level", "peer_reviewed", "revised", "population", "verdict", "status", "rejected"} <= cols("claims")
    assert {"tavily_credits", "model_id", "prompt_version", "cached_tokens"} <= cols("trace_events")
    assert {"sentence_ids", "number_gate", "id_gate", "entail_gate"} <= cols("claim_citations")


def test_old_schema_is_refused(tmp_path):
    p = str(tmp_path / "old.db")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, question TEXT)")
    c.commit()
    c.close()
    with pytest.raises(storage.SchemaMismatch):
        storage.connect(p)


def test_task_lifecycle_and_json_fields(conn):
    tid = storage.create_task(conn, "q?", {"strong": "s", "fast": "f"}, "v1")
    storage.update_task(conn, tid, brief={"aspects": ["a"]}, status="awaiting_approval")
    storage.finish_task(conn, tid, "done", final_answer="a", stop_reason="max_rounds")
    t = storage.get_task(conn, tid)
    assert t["brief"] == {"aspects": ["a"]} and t["model_ids"]["fast"] == "f" and t["prompt_version"] == "v1"
    assert t["status"] == "done" and t["finished_at"] and t["stop_reason"] == "max_rounds"
    with pytest.raises(ValueError):
        storage.update_task(conn, tid, status="bogus")


def test_cache_ttl(conn):
    storage.put_search_cache(conn, "k", "q", {}, [{"url": "u"}])
    assert storage.get_search_cache(conn, "k", 7) == [{"url": "u"}]
    conn.execute("UPDATE search_cache SET fetched_at = datetime('now', '-8 days')")
    assert storage.get_search_cache(conn, "k", 7) is None
    src = storage.upsert_source(conn, "https://a", "a", "A", None, None, 1, None)
    storage.set_source_text(conn, src, "Full page body here. Second sentence here.", ["Full page body here.",
                                                                                    "Second sentence here."], "test")
    assert storage.cached_texts(conn, ["https://a", "https://b"], 7) == {"https://a": "Full page body here. Second sentence here."}
    conn.execute("UPDATE sources SET retrieved_at = datetime('now', '-8 days')")
    assert storage.cached_texts(conn, ["https://a"], 7) == {}
    storage.put_id_cache(conn, "pmid", "1", "found", "T", {"x": 1})
    assert storage.get_id_cache(conn, "pmid", "1", 30) == {"status": "found", "title": "T", "data": {"x": 1}}


def test_v3_database_is_refused_with_instructions(tmp_path):
    path = str(tmp_path / "old.db")
    c = storage.connect(path)
    c.execute("PRAGMA user_version = 3")
    c.commit()
    c.close()
    with pytest.raises(storage.SchemaMismatch, match="sentence-level evidence"):
        storage.connect(path)


def test_citations_sentences_and_totals(conn):
    tid = storage.create_task(conn, "q")
    sq = storage.add_sub_query(conn, tid, "sq")
    src = storage.upsert_source(conn, "https://fda.gov/x", "fda.gov", "T", None, "h", 1, "regulatory_action")
    ev = storage.add_evidence(conn, tid, sq, src, "excerpt", 0.9, evidence_date="2025-06-01")
    cid = storage.add_claim(conn, tid, "claim", 0, "fact", evidence_level="regulatory decision")
    storage.set_source_text(conn, src, "One two three. Four five six.", ["One two three.", "Four five six."], "test")
    storage.add_citation(conn, cid, ev, [2])
    storage.set_gates(conn, cid, ev, verdict="supported", note="ok", number_gate={"status": "n/a"},
                      id_gate={"status": "n/a"}, entail_gate={"status": "supported"})
    c = storage.get_claims(conn, tid)[0]
    x = c["citations"][0]
    assert x["sentences"] == [{"id": f"E{ev}-2", "idx": 2, "text": "Four five six."}] and x["trust_tier"] == 1
    assert x["entail_gate"] == {"status": "supported"} and x["verdict"] == "supported"
    storage.add_trace(conn, tid, "search", {"message": "m"}, tavily_credits=2, prompt_version="v")
    storage.add_trace(conn, tid, "write", {"message": "m"}, input_tokens=10, output_tokens=3, model_id="x")
    assert storage.task_totals(conn, tid) == {"credits": 2.0, "input_tokens": 10, "output_tokens": 3,
                                              "cached_tokens": 0}
    steps = {s["step"]: s for s in storage.step_summary(conn, tid)}
    assert steps["write"]["input_tokens"] == 10 and steps["write"]["cached_share"] is None
    assert steps["search"]["credits"] == 2.0
