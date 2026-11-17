"""Straight pipeline: approval, normal run, failures, insufficient evidence, dedupe, batching, cache."""

import asyncio

import pytest
from conftest import (GOOD_TEXT, NUM_SENTENCE, PLAN, POP_SENTENCE, POPULATION, QUESTION, FakeLLM, FakeSearch, make_settings,
                      trace_steps)

from research_agent import storage
from research_agent.backends.llm import LLMOutputError
from research_agent.backends.search_base import SearchHit
from research_agent.config import TIER_DOMAINS
from research_agent.engine import run
from research_agent.prompts import PROMPT_VERSION
from research_agent.render import DISCLAIMER


def test_prepare_stops_for_approval(agent, conn):
    llm, search = FakeLLM(conn), FakeSearch()
    r = agent.prepare(llm, search)
    t = r["task"]
    assert t["status"] == "awaiting_approval"
    assert t["cancer_type"] == "non-small cell lung cancer" and t["anchor_date"] == "2025-01-01"
    assert [s["text"] for s in r["sub_queries"]] == PLAN and {s["status"] for s in r["sub_queries"]} == {"planned"}
    assert search.search_calls == [] and llm.count("WorkerNotes") == 0
    with pytest.raises(ValueError):  # cannot run twice / unapproved states
        asyncio.run(run(999, llm=llm, search=search, conn=conn, settings=agent.settings))


def test_normal_run(agent, conn):
    llm, search, events = FakeLLM(conn), FakeSearch(), []
    r = agent.full(llm, search, events=events)
    t = r["task"]
    assert t["status"] == "done", t
    assert t["stop_reason"] == "coverage_complete"
    # round 1: tiers 1-2 only, advanced depth (evidence searches), restricted to on/after the anchor date
    for call in search.search_calls:
        assert call["depth"] == "advanced" and call["start_date"] == "2025-01-01"
        assert "onclive.com" not in call["include_domains"] and "fda.gov" in call["include_domains"]
    # one batched extract call for the whole round (3 queries x 2 URLs)
    assert len(search.extract_calls) == 1 and len(search.extract_calls[0]) == 6
    assert search.extract_depths == ["basic"]
    claims = [c for c in r["claims"] if c["section"] == "new"]
    assert len(claims) == 2
    for c in claims:
        assert c["text"] == GOOD_TEXT and c["claim_type"] == "fact"
        assert c["evidence_level"] == "phase 3 randomized trial" and c["population"] == POPULATION
        assert c["verdict"] == "supported" and c["citations"][0]["verdict"] == "supported"
        texts = [x["text"] for x in c["citations"][0]["sentences"]]
        assert NUM_SENTENCE in texts  # read back from the evidence store, verbatim
        assert all(x["id"].startswith(f"E{c['citations'][0]['evidence_id']}-") for x in c["citations"][0]["sentences"])
        g = c["citations"][0]
        assert g["number_gate"]["status"] == "pass" and g["id_gate"]["status"] == "n/a"
        assert g["entail_gate"]["status"] == "supported"
    assert {c["citations"][0]["source_type"] for c in claims} <= {"regulatory_action", "peer_reviewed_article"}
    ans = t["final_answer"]
    for section in ("1. BOTTOM LINE", "2. WHAT'S NEW SINCE 2025-01-01", "3. SOURCES DISAGREE",
                    "4. NOT YET ESTABLISHED", "5. CHECK BEFORE ACTING", DISCLAIMER, "Cited sentences",
                    "Checks: numbers pass"):
        assert section in ans
    # credits: 3 advanced searches (2 each) + one extract batch of 6 pages (ceil(6/5) = 2)
    assert r["totals"]["credits"] == 8
    assert r["task"]["usage"]["used"]["tavily"] == 4  # 3 searches + 1 extract call
    assert t["answer_json"]["bottom_line"]["confidence"] == "high"
    assert events[0]["step"] == "start" and events[-1]["step"] == "done"


def test_governance_versions_on_task_and_every_trace_event(agent, conn, settings):
    r = agent.full(FakeLLM(conn), FakeSearch())
    assert r["task"]["prompt_version"] == PROMPT_VERSION
    assert r["task"]["model_ids"] == settings.model_ids()
    assert all(t["prompt_version"] == PROMPT_VERSION for t in r["trace"])
    llm_steps = [t for t in r["trace"] if t["input_tokens"]]
    assert llm_steps and all(t["model_id"] in settings.model_ids().values() for t in llm_steps)
    assert {t["model_id"] for t in trace_steps(r, "notes")} == {settings.models.fast}
    assert {t["model_id"] for t in trace_steps(r, "write")} == {settings.models.strong}


def test_writer_sees_notes_not_raw_page_text(agent, conn):
    llm = FakeLLM(conn)
    agent.full(llm, FakeSearch())
    writer_prompt = llm.users("Draft")[0]
    assert "<untrusted_note" in writer_prompt and "<untrusted_source" not in writer_prompt
    assert "Source page https://" not in writer_prompt  # raw page text never reaches the writer
    assert NUM_SENTENCE not in writer_prompt and POP_SENTENCE not in writer_prompt
    for name in ("Coverage", "Plan"):
        for u in llm.users(name):
            assert NUM_SENTENCE not in u and "<untrusted_source" not in u


def test_empty_search_results_end_insufficient(agent, conn):
    def thin(call):
        return {"aspects": [{"aspect": "regulatory approvals", "status": "thin"}], "follow_up_queries": ["x y z"]}

    llm = FakeLLM(conn, Coverage=thin)
    r = agent.full(llm, FakeSearch(empty=True))
    assert r["task"]["status"] == "insufficient_evidence"
    assert r["task"]["stop_reason"] == "no_new_evidence"
    assert llm.count("Draft") == 0 and r["claims"] == []
    assert "Insufficient evidence" in r["task"]["final_answer"]
    assert {s["status"] for s in r["sub_queries"]} == {"empty"}


def test_search_api_failure_recorded_and_run_continues(agent, conn, settings):
    search = FakeSearch(fail={PLAN[0]: ConnectionError("network down")})
    r = agent.full(FakeLLM(conn), search)
    assert [c["query"] for c in search.search_calls].count(PLAN[0]) == settings.network.max_retries
    failed = next(s for s in r["sub_queries"] if s["text"] == PLAN[0])
    assert failed["status"] == "failed"
    assert any("network down" in t["message"] for t in trace_steps(r, "search") if t["level"] == "error")
    assert r["task"]["status"] == "done"


def test_extract_api_failure_recorded(agent, conn):
    r = agent.full(FakeLLM(conn), FakeSearch(extract_error=ConnectionError("extract down")))
    assert any(t["level"] == "error" for t in trace_steps(r, "extract"))
    assert all(not e["kept"] and e["drop_reason"].startswith("source_unavailable") for e in r["evidence"])
    assert r["task"]["status"] == "insufficient_evidence"


def test_writer_reports_insufficient_evidence(agent, conn):
    def refuse(call):
        return {"sufficient": False, "insufficiency_note": "No 2025 label change found.", "items": []}

    r = agent.full(FakeLLM(conn, Draft=refuse), FakeSearch())
    assert r["task"]["status"] == "insufficient_evidence"
    assert "No 2025 label change found." in r["task"]["final_answer"]


def test_dedupe_by_url_and_content(agent, conn):
    same = "https://www.fda.gov/shared"
    mirror = "https://www.nejm.org/mirror"
    body = "Identical page body about drugx EGFR NSCLC approval and trial results."
    hits = {PLAN[0]: [SearchHit(url=same, title="a", content="drugx egfr nsclc fda approval", score=0.9)],
            PLAN[1]: [SearchHit(url=same + "/?utm_source=x", title="b", content="drugx egfr nsclc", score=0.9),
                      SearchHit(url=mirror, title="c", content="drugx egfr nsclc phase 3", score=0.9)],
            PLAN[2]: []}
    search = FakeSearch(results=hits, pages={same: body, mirror: body})
    r = agent.full(FakeLLM(conn), search)
    reasons = {e["url"]: e["drop_reason"] for e in r["evidence"] if not e["kept"]}
    assert reasons["https://nejm.org/mirror"] == "duplicate_content"
    assert "duplicate_url" in reasons.values()
    assert sum(len(c) for c in search.extract_calls) == 2  # shared URL extracted once


def test_batched_extraction_uses_at_most_20_urls_per_call(agent, conn):
    domains = tuple(f"sub{i}.{d}" for i in range(3) for d in TIER_DOMAINS[2])[:9]
    settings = make_settings(extract_per_query=10, extract_per_round=25)
    search = FakeSearch(domains=domains)
    r = agent.full(FakeLLM(conn), search, settings=settings)
    assert [len(c) for c in search.extract_calls] == [20, 5]
    extract_credits = [t["tavily_credits"] for t in trace_steps(r, "extract")]
    assert extract_credits == [4.0, 1.0]


def test_cache_hit_reuses_search_and_extract(agent, conn):
    search = FakeSearch()
    first = agent.full(FakeLLM(conn), search)
    n_search, n_extract = len(search.search_calls), len(search.extract_calls)
    second = agent.full(FakeLLM(conn), search)
    assert second["task"]["status"] == "done"
    assert len(search.search_calls) == n_search and len(search.extract_calls) == n_extract
    assert second["totals"]["credits"] == 0 and first["totals"]["credits"] > 0
    assert any(t["message"].startswith("Cache hit") for t in trace_steps(second, "search"))
    assert any(t["message"].startswith("Cache hit") for t in trace_steps(second, "extract"))


def test_invalid_llm_json_falls_back(agent, conn):
    llm = FakeLLM(conn, Plan=lambda c: LLMOutputError("Plan: invalid output after 2 attempts"))
    r = agent.prepare(llm, FakeSearch())
    assert r["task"]["status"] == "awaiting_approval"
    assert [s["text"] for s in r["sub_queries"]] == ["What changed for first-line therapy?"]
    assert any("Planning failed" in t["message"] for t in trace_steps(r, "plan"))

    llm = FakeLLM(conn, Brief=lambda c: LLMOutputError("Brief: invalid"))
    r = agent.prepare(llm, FakeSearch())
    assert r["task"]["brief"]["question"] == QUESTION and r["task"]["status"] == "awaiting_approval"


def test_failed_writer_marks_task_failed(agent, conn):
    r = agent.full(FakeLLM(conn, Draft=lambda c: RuntimeError("writer down")), FakeSearch())
    assert r["task"]["status"] == "failed" and "writer down" in r["task"]["error"]


def test_task_state_persisted_before_finish(agent, conn):
    agent.prepare(FakeLLM(conn), FakeSearch())
    assert storage.get_task(conn, 1)["brief"]["aspects"]
    assert storage.get_trace(conn, 1)[-1]["step"] == "approval"


def test_writer_connection_drop_is_retried_patiently(agent, conn):
    n = {"i": 0}

    def flaky(call):
        n["i"] += 1
        if n["i"] == 1:
            return ConnectionError("connection reset")
        from conftest import h_draft
        return h_draft(call)

    r = agent.full(FakeLLM(conn, Draft=flaky), FakeSearch())
    assert r["task"]["status"] == "done" and n["i"] == 2
    assert any("trying again" in t["message"] for t in trace_steps(r, "write"))


def test_failed_writer_can_be_resumed_without_new_research(agent, conn):
    search = FakeSearch()
    r = agent.full(FakeLLM(conn, Draft=lambda c: ConnectionError("connection reset")), search)
    t = r["task"]
    assert t["status"] == "failed" and "Resume with: research --resume" in trace_steps(r, "error")[-1]["message"]
    n_search, n_extract = len(search.search_calls), len(search.extract_calls)

    from research_agent.engine import resume
    r2 = asyncio.run(resume(t["id"], **agent.kw(FakeLLM(conn), search)))
    assert r2["task"]["status"] == "done" and r2["task"]["error"] is None
    assert len(search.search_calls) == n_search and len(search.extract_calls) == n_extract
    assert [c for c in r2["claims"] if c["verdict"] == "supported"]
    with pytest.raises(ValueError):  # a finished task cannot be resumed
        asyncio.run(resume(t["id"], **agent.kw(FakeLLM(conn), search)))
