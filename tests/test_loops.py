"""Coverage loop (follow-ups, tiers), the four stop reasons and the Tavily quota with its repair reserve."""

import itertools

from conftest import PLAN, FakeLLM, FakeSearch, make_settings, trace_steps

from research_agent.backends.search_base import SearchHit
from research_agent.engine.orchestrator import STOP_REASONS


def thin_then_complete(follow_ups):
    calls = itertools.count()

    def handler(call):
        if next(calls) == 0:
            return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin", "evidence_ids": []}],
                    "follow_up_queries": follow_ups}
        from conftest import h_coverage_complete
        return h_coverage_complete(call)
    return handler


def always_thin_unique():
    n = itertools.count()

    def handler(call):
        i = next(n)
        return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}],
                "follow_up_queries": [f"unique angle number{i} alpha{i} beta{i}"]}
    return handler


def stop_event(r):
    return trace_steps(r, "stop")[-1]


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
    assert all("onclive.com" not in c["include_domains"] for c in search.search_calls[:len(PLAN)])
    assert r["task"]["status"] == "insufficient_evidence"


def test_off_allowlist_results_are_dropped(agent, conn):
    results = {PLAN[0]: [SearchHit(url="https://www.fda.gov/a", content="drugx egfr nsclc", score=0.9),
                         SearchHit(url="https://randomblog.example/x", content="drugx egfr nsclc", score=0.9)]}

    class Leaky(FakeSearch):  # a backend that ignores include_domains
        async def search(self, query, **kw):
            return await super().search(query, **{**kw, "include_domains": ["fda.gov", "randomblog.example"]})

    r = agent.full(FakeLLM(conn), Leaky(results=results))
    dropped = {e["url"]: e["drop_reason"] for e in r["evidence"] if not e["kept"]}
    assert dropped.get("https://randomblog.example/x") == "off_allowlist"


def test_follow_ups_capped_per_aspect(agent, conn):
    n = itertools.count()

    def many(call):
        if next(n):
            from conftest import h_coverage_complete
            return h_coverage_complete(call)
        return {"aspects": [{"aspect": "safety communications", "status": "thin"},
                            {"aspect": "trial results", "status": "thin"}],
                "follow_up_queries": [
                    {"aspect": "safety communications", "query": "drugx FDA safety communication 2025"},
                    {"aspect": "safety communications", "query": "drugx boxed warning label update"},
                    {"aspect": "safety communications", "query": "drugx interstitial lung disease alert"},
                    {"aspect": "trial results", "query": "drugx overall survival readout"},
                ]}

    r = agent.full(FakeLLM(conn, Coverage=many), FakeSearch())
    round2 = [s["text"] for s in r["sub_queries"] if s["round"] == 2]
    assert round2 == ["drugx FDA safety communication 2025", "drugx overall survival readout"]
    capped = trace_steps(r, "coverage")[0]["detail"]["skipped_by_aspect_cap"]
    assert [c["reason"] for c in capped] == ["per-aspect cap", "per-aspect cap"]


def test_no_follow_up_for_an_aspect_already_covered(agent, conn):
    from conftest import h_coverage_complete

    def covered_but_asks(call):
        out = h_coverage_complete(call)
        out["aspects"].append({"aspect": "safety communications", "status": "thin"})
        out["follow_up_queries"] = [{"aspect": "regulatory approvals", "query": "drugx approval news"}]
        return out

    r = agent.full(FakeLLM(conn, Coverage=covered_but_asks), FakeSearch())
    assert r["task"]["stop_reason"] == "no_new_evidence"
    assert "no new follow-up queries" in stop_event(r)["detail"]["why"]
    capped = trace_steps(r, "coverage")[0]["detail"]["skipped_by_aspect_cap"]
    assert capped[0]["reason"] == "aspect already covered"


def test_near_duplicate_follow_ups_are_skipped(agent, conn):
    def thin(call):
        return {"aspects": [{"aspect": "phase 3 trial results", "status": "thin"}],
                "follow_up_queries": ["FDA approval drugx EGFR NSCLC"]}  # same terms as PLAN[0]

    r = agent.full(FakeLLM(conn, Coverage=thin), FakeSearch())
    assert r["task"]["stop_reason"] == "no_new_evidence"
    assert trace_steps(r, "coverage")[0]["detail"]["skipped_near_duplicates"] == ["FDA approval drugx EGFR NSCLC"]


# --- the four stop reasons ---------------------------------------------------------

def test_every_stop_reason_is_one_of_four():
    assert STOP_REASONS == ("coverage_complete", "no_new_evidence", "quota_reached", "time_budget")


def test_stop_coverage_complete(agent, conn):
    r = agent.full(FakeLLM(conn), FakeSearch())
    assert r["task"]["stop_reason"] == "coverage_complete"
    assert "coverage_complete" in stop_event(r)["message"]


def test_stop_no_new_evidence_after_two_empty_rounds(agent, conn):
    """Follow-up searches only return URLs already read in round 1, so rounds 2 and 3 add no sentence IDs."""
    class SeenOnly(FakeSearch):
        async def search(self, query, **kw):
            if query in PLAN:
                return await super().search(query, **kw)
            self.search_calls.append({"query": query, **kw})
            from research_agent.backends.search_base import SearchResult
            return SearchResult(hits=[SearchHit(url=f"https://www.fda.gov/{q.replace(' ', '-')}", content=q, score=0.8)
                                      for q in PLAN], credits=2.0)

    r = agent.full(FakeLLM(conn, Coverage=always_thin_unique()), SeenOnly())
    assert r["task"]["stop_reason"] == "no_new_evidence"
    assert stop_event(r)["detail"]["why"] == "two consecutive rounds added no new sentence IDs"
    added = [t["detail"]["new_sentences"] for t in trace_steps(r, "round") if "new_sentences" in t["detail"]]
    assert added[0] > 0 and added[1:] == [0, 0]


def test_stop_quota_reached_and_repair_reserve_kept(agent, conn):
    # 12 Tavily calls, 8 reserved for repair: research may use 4 (3 searches + 1 extract in round 1).
    settings = make_settings(budget={"tavily_calls": 12, "tavily_repair_reserve": 8})
    r = agent.full(FakeLLM(conn, Coverage=always_thin_unique()), FakeSearch(), settings=settings)
    assert r["task"]["stop_reason"] == "quota_reached"
    usage = r["task"]["usage"]
    assert usage["used"]["tavily"] == 4 and usage["tavily_repair_used"] == 0
    assert "8 reserved for repair" in stop_event(r)["detail"]["why"]


def test_stop_round_cap_is_reported_as_quota_reached(agent, conn):
    r = agent.full(FakeLLM(conn, Coverage=always_thin_unique()), FakeSearch())
    assert max(s["round"] for s in r["sub_queries"]) == agent.settings.limits.max_research_rounds
    assert r["task"]["stop_reason"] == "quota_reached"
    assert "round cap" in stop_event(r)["detail"]["why"]


def test_stop_time_budget(agent, conn):
    settings = make_settings()
    from dataclasses import replace
    settings = replace(settings, budget=replace(settings.budget, time_s=0))
    r = agent.full(FakeLLM(conn), FakeSearch(), settings=settings)
    assert r["task"]["stop_reason"] == "time_budget"
    assert "Research stopped" in stop_event(r)["message"] and "time_budget" in stop_event(r)["message"]


def test_coverage_check_failure_stops_as_no_new_evidence(agent, conn):
    r = agent.full(FakeLLM(conn, Coverage=lambda c: RuntimeError("bad json")), FakeSearch())
    assert r["task"]["stop_reason"] == "no_new_evidence"
    assert "coverage check failed" in stop_event(r)["detail"]["why"]
    assert r["task"]["status"] == "done"  # still writes from the notes it has


def test_stop_reason_printed_by_report(agent, conn):
    from research_agent.render import format_report
    r = agent.full(FakeLLM(conn), FakeSearch())
    out = format_report(r)
    assert "Stop reason: coverage_complete" in out and "COST BY STEP" in out
