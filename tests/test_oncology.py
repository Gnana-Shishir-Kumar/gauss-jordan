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


def test_tier_3_only_claim_is_tagged_uncertain(agent, conn):
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
        good = num_notes(call.llm.conn)[0]
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [good_item(call.llm.conn, good),
                          {"text": "NCCN updated its first-line recommendation.", "claim_type": "fact",
                           "sentence_ids": [f"E{eid}-1"]}]}

    # The blocked page has no stored sentences, so the citation is rejected; repair deletes the claim.
    r = agent.full(FakeLLM(conn, Draft=draft, RepairAction=lambda c: {"action": "delete", "reason": "no source"}),
                   search)
    ev = next(e for e in r["evidence"] if e["url"] == "https://nccn.org/drugx-egfr-nsclc-fda-approval")
    assert not ev["kept"] and ev["drop_reason"].startswith("source_unavailable") and "login" in ev["drop_reason"]
    assert ev["source_type"] == "guideline" and ev["trust_tier"] == 1
    # The blocked page never counts as support for any claim.
    assert all(x["evidence_id"] != ev["id"] for c in r["claims"] for x in c["citations"])
    nccn = next(c for c in r["claims"] if c["text"].startswith("NCCN"))
    assert nccn["status"] == "deleted" and nccn["rejected"][0]["reason"] == "sentence ID not in the evidence store"
    assert r["task"]["status"] == "done"  # the run continued with the other sources



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
