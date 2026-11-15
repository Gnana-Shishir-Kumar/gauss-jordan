"""Coverage loop (follow-ups, tiers, stop reasons) and the repair loop."""

import itertools

from conftest import GOOD_TEXT, NUM_SENTENCE, PLAN, FakeLLM, FakeSearch, make_settings, notes_with_numbers, trace_steps

from research_agent.backends.search_base import SearchHit


def thin_then_complete(follow_ups):
    calls = itertools.count()

    def handler(call):
        if next(calls) == 0:
            return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin", "evidence_ids": []}],
                    "follow_up_queries": follow_ups}
        from conftest import h_coverage_complete
        return h_coverage_complete(call)
    return handler


def test_follow_up_round_uses_advanced_depth_and_tier_3(agent, conn):
    search = FakeSearch()
    r = agent.full(FakeLLM(conn, Coverage=thin_then_complete(["drugx overall survival update 2025"])), search)
    round2 = [s for s in r["sub_queries"] if s["round"] == 2]
    assert [s["text"] for s in round2] == ["drugx overall survival update 2025"] and round2[0]["depth"] == "advanced"
    call = next(c for c in search.search_calls if c["query"] == round2[0]["text"])
    assert call["depth"] == "advanced" and "onclive.com" in call["include_domains"]
    assert any(e["domain"] == "onclive.com" and e["trust_tier"] == 3 for e in r["evidence"])
    assert r["task"]["stop_reason"] == "coverage_complete"


def test_round_one_never_searches_tier_3(agent, conn):
    search = FakeSearch(domains=("onclive.com", "targetedonc.com"))
    r = agent.full(FakeLLM(conn), search)
    assert all("onclive.com" not in c["include_domains"] for c in search.search_calls)
    assert r["evidence"] == [] and r["task"]["status"] == "insufficient_evidence"


def test_off_allowlist_results_are_dropped(agent, conn):
    results = {PLAN[0]: [SearchHit(url="https://www.fda.gov/a", content="drugx egfr nsclc", score=0.9),
                         SearchHit(url="https://randomblog.example/x", content="drugx egfr nsclc", score=0.9)]}

    class Leaky(FakeSearch):  # a backend that ignores include_domains
        async def search(self, query, **kw):
            res = await super().search(query, **{**kw, "include_domains": ["fda.gov", "randomblog.example"]})
            return res

    r = agent.full(FakeLLM(conn), Leaky(results=results))
    dropped = {e["url"]: e["drop_reason"] for e in r["evidence"] if not e["kept"]}
    assert dropped.get("https://randomblog.example/x") == "off_allowlist"


def test_stops_at_max_rounds(agent, conn):
    n = itertools.count()

    def always_thin(call):
        i = next(n)
        return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}],
                "follow_up_queries": [f"unique angle number{i} alpha{i} beta{i}"]}

    r = agent.full(FakeLLM(conn, Coverage=always_thin), FakeSearch())
    assert max(s["round"] for s in r["sub_queries"]) == 3
    assert r["task"]["stop_reason"] == "max_rounds"


def test_stops_when_round_adds_too_little(agent, conn):
    # The follow-up query returns only URLs already seen in round 1 -> 0 new kept evidence.
    results = {"drugx followup": [SearchHit(url=f"https://www.fda.gov/{q.replace(' ', '-')}", content=q, score=0.8)
                                  for q in PLAN]}
    n = itertools.count()

    def thin(call):
        next(n)
        return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}],
                "follow_up_queries": ["drugx followup"]}

    r = agent.full(FakeLLM(conn, Coverage=thin), FakeSearch(results=results))
    assert r["task"]["stop_reason"] == "few_new_evidence"
    assert max(s["round"] for s in r["sub_queries"]) == 2


def test_near_duplicate_follow_ups_are_skipped(agent, conn):
    def thin(call):
        return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}],
                "follow_up_queries": ["FDA approval drugx EGFR NSCLC"]}  # same terms as PLAN[0]

    r = agent.full(FakeLLM(conn, Coverage=thin), FakeSearch())
    assert r["task"]["stop_reason"] == "no_new_queries"
    cov = trace_steps(r, "coverage")[0]
    assert cov["detail"]["skipped_near_duplicates"] == ["FDA approval drugx EGFR NSCLC"]


def test_credit_budget_stops_research(agent, conn):
    def thin(call):
        return {"aspects": [{"aspect": "x", "status": "thin"}], "follow_up_queries": ["another angle entirely"]}

    settings = make_settings()
    from dataclasses import replace
    from research_agent.config import Budget
    settings = replace(settings, budget=Budget(time_s=60, credits=4))
    r = agent.full(FakeLLM(conn, Coverage=thin), FakeSearch(), settings=settings)
    assert r["task"]["stop_reason"] == "credit_budget"


def test_coverage_check_failure_stops_with_reason(agent, conn):
    r = agent.full(FakeLLM(conn, Coverage=lambda c: RuntimeError("bad json")), FakeSearch())
    assert r["task"]["stop_reason"] == "coverage_check_failed"
    assert r["task"]["status"] == "done"  # still writes from the notes it has


# --- repair loop -----------------------------------------------------------------

def verify_unless_revised(call):
    ok = "(revised)" in call.user
    from conftest import blocks
    return {"verdicts": [{"evidence_id": eid, "verdict": "supported" if ok else "unsupported",
                          "note": "matches" if ok else "excerpt does not say this"} for eid, _ in blocks(call.user)]}


def test_unsupported_citation_is_repaired(agent, conn):
    llm = FakeLLM(conn, ClaimVerification=verify_unless_revised)
    r = agent.full(llm, FakeSearch())
    claims = [c for c in r["claims"] if c["section"] == "new"]
    assert claims and all(c["revised"] for c in claims)
    assert all(x["verdict"] == "supported" for c in claims for x in c["citations"])
    assert llm.count("Repair") == 1 and r["task"]["status"] == "done"
    assert "FLAGGED" not in r["task"]["final_answer"]


def test_unsupported_citation_stays_flagged(agent, conn):
    def never(call):
        from conftest import blocks
        return {"verdicts": [{"evidence_id": e, "verdict": "unsupported", "note": "not stated"} for e, _ in blocks(call.user)]}

    llm = FakeLLM(conn, ClaimVerification=never)
    r = agent.full(llm, FakeSearch())
    assert llm.count("Repair") == 1  # max one repair pass
    assert all(x["verdict"] == "unsupported" for c in r["claims"] for x in c["citations"])
    assert "[FLAGGED: not verified]" in r["task"]["final_answer"]
    assert r["task"]["status"] == "insufficient_evidence"
    assert r["task"]["answer_json"]["bottom_line"]["confidence"] == "low"


def test_number_missing_from_source_sentence_is_repaired(agent, conn):
    def wrong_number(call):
        n = notes_with_numbers(call.llm.conn)[0]
        eid = n["evidence_ids"][0]
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [{"text": "PFS hazard ratio was 0.68 with drugx.", "claim_type": "fact",
                           "evidence_ids": [eid], "evidence_level": "phase 3 randomized trial",
                           "numbers": [{"value": "0.68", "sentence": NUM_SENTENCE, "evidence_id": eid}]}]}

    llm = FakeLLM(conn, Draft=wrong_number)
    r = agent.full(llm, FakeSearch())
    checks = [t["detail"] for t in trace_steps(r, "check") if t["detail"].get("numeric")]
    assert [(c["phase"], c["passed"]) for c in checks] == [("initial", False), ("repair", True)]
    assert "0.68" in checks[0]["problems"][0]
    # the LLM verifier was not asked about the failing claim; only after repair
    assert all("0.68" not in u for u in llm.users("ClaimVerification"))
    claim = r["claims"][0]
    assert claim["revised"] and claim["text"].startswith(GOOD_TEXT)
    assert claim["citations"][0]["verdict"] == "supported" and r["task"]["status"] == "done"
