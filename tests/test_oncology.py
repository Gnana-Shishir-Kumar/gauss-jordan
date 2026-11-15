"""Oncology specialization: tiers, login walls, population, brief handling, identifiers, exports."""

import csv
import io

import pytest
from conftest import (NUM_SENTENCE, POPULATION, TIER3_SENTENCE, TODAY, FakeLLM, FakeSearch, h_coverage_complete,
                      notes_with_numbers, page_for)

from research_agent.engine.checks import POPULATION_NOT_STATED
from research_agent.exports import to_csv, to_ris
from research_agent.privacy import IdentifierError


def test_tier_3_only_claim_is_tagged_uncertain(agent, conn):
    calls = []

    def coverage(call):
        calls.append(1)
        if len(calls) == 1:
            return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}],
                    "follow_up_queries": ["drugx response rate news"]}
        return h_coverage_complete(call)

    def draft(call):
        notes = [n for n in notes_with_numbers(call.llm.conn) if TIER3_SENTENCE in str(n["details"])]
        eid = notes[0]["evidence_ids"][0]
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "high", "reason": "r"},
                "items": [{"text": "Drugx had an objective response rate of 61%.", "claim_type": "fact",
                           "evidence_ids": [eid], "numbers": [{"value": "61", "sentence": TIER3_SENTENCE,
                                                               "evidence_id": eid}]}]}

    from research_agent.backends.search_base import SearchHit
    news = [SearchHit(url="https://www.onclive.com/drugx-orr", title="Drugx ORR", content="drugx response rate",
                      score=0.9)]
    r = agent.full(FakeLLM(conn, Coverage=coverage, Draft=draft), FakeSearch(results={"drugx response rate news": news}))
    c = r["claims"][0]
    assert c["citations"][0]["trust_tier"] == 3 and c["citations"][0]["verdict"] == "supported"
    assert c["claim_type"] == "uncertain"
    assert c["citations"][0]["source_type"] == "news"
    assert "Tier 3" in r["task"]["final_answer"]
    assert r["task"]["answer_json"]["bottom_line"]["confidence"] == "low"  # no tier 1-2 support


def test_login_blocked_guideline_page(agent, conn):
    blocked = "https://www.nccn.org/drugx-egfr-nsclc-fda-approval"
    search = FakeSearch(domains=("nccn.org", "fda.gov", "nejm.org"), login_blocked=(blocked,))

    def draft(call):
        eid = conn.execute("""SELECT e.id FROM evidence e JOIN sources s ON s.id = e.source_id
                              WHERE s.domain = 'nccn.org' AND e.kept = 0""").fetchone()[0]
        good = notes_with_numbers(call.llm.conn)[0]
        from conftest import good_item
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [good_item(good),
                          {"text": "NCCN updated its first-line recommendation.", "claim_type": "fact",
                           "evidence_ids": [eid]}]}

    # The writer cannot fix it in repair, so the citation stays source_unavailable.
    r = agent.full(FakeLLM(conn, Draft=draft, Repair=lambda c: {"claims": []}), search)
    ev = next(e for e in r["evidence"] if e["url"] == "https://nccn.org/drugx-egfr-nsclc-fda-approval")
    assert not ev["kept"] and ev["drop_reason"].startswith("source_unavailable") and "login" in ev["drop_reason"]
    assert ev["source_type"] == "guideline" and ev["trust_tier"] == 1
    nccn_claim = next(c for c in r["claims"] if c["text"].startswith("NCCN"))
    assert nccn_claim["citations"][0]["verdict"] == "source_unavailable"
    assert r["task"]["status"] == "done"  # the run continued with the other sources


def test_population_not_stated_when_not_in_sources(agent, conn):
    def draft(call):
        n = notes_with_numbers(call.llm.conn)[0]
        from conftest import good_item
        a = good_item(n) | {"population": "adults aged 65 or older in Japan"}  # not stated anywhere
        b = good_item(n) | {"population": None}
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [a, b]}

    r = agent.full(FakeLLM(conn, Draft=draft), FakeSearch())
    assert [c["population"] for c in r["claims"]] == [POPULATION_NOT_STATED, POPULATION_NOT_STATED]
    assert "Population: population not stated" in r["task"]["final_answer"]


def test_population_never_inferred_when_page_omits_it(agent, conn):
    pages_without_population = {}

    class NoPop(FakeSearch):
        async def extract(self, urls, *, query=None):
            res = await super().extract(urls, query=query)
            res.pages = {u: t.replace("Approval was based on a phase 3 randomized trial in adults aged 18 years or "
                                      "older in North America and Asia. ", "") for u, t in res.pages.items()}
            return res

    def notes(call):  # a careless worker claims a population the excerpt does not state
        from conftest import blocks
        return {"notes": [{"text": "f", "evidence_ids": [e], "population": POPULATION,
                           "population_sentence": "Patients were adults aged 18 years or older.",
                           "numbers": [{"value": "0.70", "sentence": NUM_SENTENCE}]} for e, _ in blocks(call.user)]}

    r = agent.full(FakeLLM(conn, WorkerNotes=notes), NoPop())
    assert all(n["details"]["population"] is None for n in r["notes"])
    assert all(c["population"] == POPULATION_NOT_STATED for c in r["claims"])
    assert pages_without_population == {}


def test_evidence_level_not_inferred(agent, conn):
    def draft(call):
        n = notes_with_numbers(call.llm.conn)[0]
        from conftest import good_item
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [good_item(n) | {"evidence_level": "phase 2 single-arm trial"}]}

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
