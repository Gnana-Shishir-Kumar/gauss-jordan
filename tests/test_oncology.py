"""Oncology specialization: tiers, login walls, population, brief handling, identifiers, exports."""

import csv
import io

import pytest
from conftest import (POPULATION, TODAY, FakeLLM, FakeSearch, active, good_item, h_coverage_complete,
                      latest_task, num_notes, page_for)

from research_agent import storage


from research_agent.engine.checks import POPULATION_NOT_STATED
from research_agent.exports import to_csv, to_ris
from research_agent.privacy import IdentifierError


def test_tier_4_only_efficacy_claim_tagged_company_reported_and_uncertain(agent, conn):
    calls = []

    def coverage(call):
        calls.append(1)
        if len(calls) == 1:
            return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}],
                    "follow_up_queries": ["drugx response rate news"]}
        return h_coverage_complete(call)

    def draft(call):
        notes = [n for n in storage.get_notes(call.llm.conn, latest_task(call.llm.conn)) if "61%" in n["text"]]
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "high", "reason": "r"},
                "items": [{"text": "Drugx had an objective response rate of 61%.", "claim_type": "fact",
                           "sentence_ids": notes[0]["sentence_ids"]}]}

    from research_agent.backends.search_base import SearchHit
    news = [SearchHit(url="https://www.onclive.com/drugx-orr", title="Drugx ORR", content="drugx response rate",
                      score=0.9)]
    r = agent.full(FakeLLM(conn, Coverage=coverage, Draft=draft), FakeSearch(results={"drugx response rate news": news}))
    c = r["claims"][0]
    assert c["citations"][0]["trust_tier"] == 4 and c["citations"][0]["verdict"] == "supported"
    assert c["claim_type"] == "uncertain" and "company-reported, not peer-reviewed" in c["labels"]
    assert c["citations"][0]["source_type"] == "news"
    assert "Tier 4" in r["task"]["final_answer"] and "COMPANY-REPORTED" in r["task"]["final_answer"]
    assert r["task"]["answer_json"]["bottom_line"]["confidence"] == "low"  # no tier 1-2 support


def test_nccn_is_never_searched_or_fetched(agent, conn):
    """NCCN's terms prohibit automated access: it is not on the allowlist, and a leaked NCCN result is dropped."""
    from research_agent.backends.search_base import SearchHit

    class Leaky(FakeSearch):
        async def search(self, query, **kw):
            res = await super().search(query, **kw)
            res.hits.append(SearchHit(url="https://www.nccn.org/guidelines/x", title="NCCN", content=query, score=0.99))
            return res

    search = Leaky()
    r = agent.full(FakeLLM(conn), search)
    assert all("nccn.org" not in d for c in search.search_calls for d in c["include_domains"])
    assert all("nccn.org" not in u for batch in search.extract_calls for u in batch)
    nccn = [e for e in r["evidence"] if "nccn.org" in e["url"]]
    assert nccn and all(not e["kept"] and e["drop_reason"] == "off_allowlist" for e in nccn)


def test_tier_3_claim_is_always_labelled_preliminary(agent, conn):
    """Conference abstracts are searched only in follow-up rounds and are always shown as PRELIMINARY."""
    from research_agent.backends.search_base import SearchHit
    abstract = "https://meetings.asco.org/abstracts-presentations/drugx-egfr-nsclc"
    q = "drugx asco abstract update"
    hits = {q: [SearchHit(url=abstract, title="Drugx abstract", content="drugx egfr nsclc abstract", score=0.95)]}
    calls = []

    def coverage(call):
        calls.append(1)
        if len(calls) == 1:
            return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}], "follow_up_queries": [q]}
        return h_coverage_complete(call)

    def draft(call):
        conn_ = call.llm.conn
        note = next(n for n in num_notes(conn_) if any(
            e["url"] == abstract.replace("https://", "https://") for e in storage.get_evidence(
                conn_, latest_task(conn_), ids=n["evidence_ids"])))
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [good_item(conn_, note)]}

    r = agent.full(FakeLLM(conn, Coverage=coverage, Draft=draft), FakeSearch(results=hits))
    claim = active(r)[0]
    assert claim["citations"][0]["trust_tier"] == 3 and "preliminary" in claim["labels"]
    assert claim["claim_type"] == "fact"  # preliminary, but not demoted: tier 3 is not tier 4
    assert "PRELIMINARY" in r["task"]["final_answer"]

def test_population_not_stated_when_not_in_sources(agent, conn):
    def draft(call):
        n = num_notes(call.llm.conn)[0]
        a = good_item(call.llm.conn, n) | {"population": "adults aged 65 or older in Japan"}  # not stated anywhere
        b = good_item(call.llm.conn, n) | {"population": None}
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [a, b]}

    r = agent.full(FakeLLM(conn, Draft=draft), FakeSearch())
    assert [c["population"] for c in r["claims"]] == [POPULATION_NOT_STATED, POPULATION_NOT_STATED]
    assert "Population: population not stated" in r["task"]["final_answer"]


def test_population_never_inferred_when_page_omits_it(agent, conn):
    pages_without_population = {}

    class NoPop(FakeSearch):
        async def extract(self, urls, *, depth="basic"):
            res = await super().extract(urls, depth=depth)
            res.pages = {u: t.replace("Approval was based on a phase 3 randomized trial in adults aged 18 years or "
                                      "older in North America and Asia. ", "") for u, t in res.pages.items()}
            return res

    # the writer claims a population that no cited sentence states
    r = agent.full(FakeLLM(conn), NoPop())
    assert active(r) and all(c["population"] == POPULATION_NOT_STATED for c in r["claims"])
    assert POPULATION not in str(r["notes"])
    assert pages_without_population == {}


def test_evidence_level_not_inferred(agent, conn):
    def draft(call):
        n = num_notes(call.llm.conn)[0]
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [good_item(call.llm.conn, n) | {"evidence_level": "phase 2 single-arm trial"}]}

    r = agent.full(FakeLLM(conn, Draft=draft), FakeSearch())
    assert r["claims"][0]["evidence_level"] == "not stated"


def test_brief_asks_clarifying_question(agent, conn):
    llm = FakeLLM(conn, Brief=lambda c: {"needs_clarification": True,
                                         "clarifying_question": "Which cancer type and line of therapy?"})
    search = FakeSearch()
    r = agent.prepare(llm, search, question="What's new in cancer treatment?")
    assert r["task"]["status"] == "needs_clarification"
    assert llm.count("Plan") == 0 and search.search_calls == []
    assert r["task"]["brief"]["clarifying_question"] == "Which cancer type and line of therapy?"


def test_missing_anchor_defaults_to_12_months(agent, conn):
    from conftest import h_brief
    llm = FakeLLM(conn, Brief=lambda c: h_brief(c) | {"anchor_date": None})
    search = FakeSearch()
    r = agent.full(llm, search)
    assert r["task"]["anchor_date"] == "2025-11-15" and r["task"]["brief"]["anchor_defaulted"]
    assert search.search_calls[0]["start_date"] == "2025-11-15"
    assert "12 months since 2025-11-15" in r["task"]["final_answer"]


@pytest.mark.parametrize("question", [
    "MRN 4482913: what changed for EGFR NSCLC since 2025?",
    "My patient John Smith, DOB 03/14/1961, has EGFR NSCLC. What changed?",
    "Mrs. Alvarez has HER2+ breast cancer; any updates since June 2025?",
    "Record 12345678 shows stage IV CRC; new options?",
])
def test_identifier_check_blocks_and_stores_nothing(agent, conn, question):
    with pytest.raises(IdentifierError):
        agent.prepare(FakeLLM(conn), FakeSearch(), question=question)
    assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_ris_and_csv_export(agent, conn):
    r = agent.full(FakeLLM(conn), FakeSearch())
    ris = to_ris(r)
    records = [rec for rec in ris.split("ER  - ") if rec.strip()]
    assert len(records) == 2
    assert "TY  - GOVDOC" in ris and "TY  - JOUR" in ris
    assert "UR  - https://fda.gov/drugx-egfr-nsclc-fda-approval" in ris
    assert "KW  - Tier 1 · regulatory / registry / guideline" in ris
    assert "N1  - Source type: regulatory_action; trust tier: 1" in ris
    assert all(line == "" or line[2:6] == "  - " for line in ris.splitlines())
    rows = list(csv.DictReader(io.StringIO(to_csv(r))))
    assert {row["source_type"] for row in rows} == {"regulatory_action", "peer_reviewed_article"}
    assert {"title", "url", "source_type", "date", "trust_tier"} <= set(rows[0])
    assert page_for("x")  # fixture sanity
    assert TODAY.year == 2026



# --- guideline baseline and the "What changed" table --------------------------------

def test_guideline_baseline_kept_only_when_user_states_it(agent, conn):
    from conftest import h_brief
    q = "What changed for first-line EGFR NSCLC since NCCN NSCLC v3.2025?"
    llm = FakeLLM(conn, Brief=lambda c: h_brief(c) | {"anchor_date": None, "guideline_name": "NCCN NSCLC",
                                                      "guideline_version": "v3.2025", "guideline_date": "2025-01-14"})
    r = agent.prepare(llm, FakeSearch(), question=q)
    b = r["task"]["brief"]
    assert b["guideline_name"] == "NCCN NSCLC" and b["guideline_version"] == "v3.2025"
    assert b["guideline_date"] is None  # the model guessed a date the user never wrote
    llm = FakeLLM(conn, Brief=lambda c: h_brief(c) | {"guideline_name": "ESMO 2024", "guideline_version": "v9"})
    b = agent.prepare(llm, FakeSearch())["task"]["brief"]
    assert b["guideline_name"] is None and b["guideline_version"] is None
    assert any("not stated in the question" in t["message"] for t in storage.get_trace(conn, 2))


def test_user_supplied_guideline_sets_anchor(agent, conn):
    import asyncio

    from conftest import TODAY, h_brief
    from research_agent.engine import prepare
    llm = FakeLLM(conn, Brief=lambda c: h_brief(c) | {"anchor_date": None})
    kw = agent.kw(llm, FakeSearch())
    kw.pop("ids"), kw.pop("structured")
    r = asyncio.run(prepare("What changed for first-line EGFR NSCLC?", today=TODAY,
                            guideline={"name": "NCCN NSCLC", "version": "v3.2025", "date": "2025-01-14"}, **kw))
    t = r["task"]
    assert t["anchor_date"] == "2025-01-14" and not t["brief"]["anchor_defaulted"]
    assert t["brief"]["guideline_version"] == "v3.2025"
    from research_agent.prompts import brief_text
    assert "NCCN NSCLC version v3.2025 dated 2025-01-14" in brief_text(t["brief"])


def test_what_changed_table_and_impact_interpretation(agent, conn):
    from conftest import h_draft

    def draft(call):
        d = h_draft(call)
        d["items"][0].update(change_type="trial readout", impact="May support drugx as a first-line option.")
        d["items"][1].update(change_type="new approval", impact="Could change care for 45% of patients.")
        return d

    r = agent.full(FakeLLM(conn, Draft=draft), FakeSearch())
    claims = active(r)
    assert sorted(c["change_type"] for c in claims) == ["new approval", "new trial data"]
    assert sorted(bool(c["impact"]) for c in claims) == [False, True]  # the invented 45% note was dropped
    ans = r["task"]["final_answer"]
    assert "| Change | Date | Evidence level | Tier | Review status | Verified | Practice impact (interpretation) |" in ans
    assert "| new trial data | 2025-06-01 | phase 3 randomized trial |" in ans
    assert "Practice impact (interpretation, not a finding): May support drugx" in ans
    assert any("practice-impact note states numbers" in t["message"] for t in r["trace"])
