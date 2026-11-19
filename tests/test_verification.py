"""Sentence-level evidence, the three verifier gates, the bounded repair loop, Extract fallbacks and 429 handling."""

import asyncio

import httpx
from conftest import (GOOD_TEXT, NUM_SENTENCE, PLAN, POP_SENTENCE, FakeIds, FakeLLM, FakeSearch, active, make_settings,
                      num_notes, sid, trace_steps)

from research_agent import storage
from research_agent.backends.http import HTTP
from research_agent.backends.identifiers import Identifier, ids_in_text, ids_in_url, title_matches
from research_agent.backends.search_base import SearchHit
from research_agent.backends.tavily import TavilyBackend, TavilyHTTP
from research_agent.engine import numbers
from research_agent.retry import RateLimitedError, with_retry
from research_agent.sentences import parse_ref, split_sentences


def gate_events(r, gate):
    return [t["detail"] for t in trace_steps(r, "check") + trace_steps(r, "verify") if t["detail"].get("gate") == gate]


def draft_with(text, refs_fn=None):
    """A writer that returns one item with `text`, citing the first drugx note's sentences (or refs_fn(conn))."""
    def draft(call):
        conn = call.llm.conn
        refs = refs_fn(conn) if refs_fn else num_notes(conn)[0]["sentence_ids"]
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "moderate", "reason": "r"},
                "items": [{"text": text, "claim_type": "fact", "sentence_ids": refs, "date": "2025-06-01",
                           "evidence_level": "phase 3 randomized trial"}]}
    return draft


# --- sentence store ------------------------------------------------------------------

def test_sentence_ids_and_splitting():
    s = split_sentences("Median PFS was 23.7 vs. 16.6 months (HR 0.70; p<.001). Smith J. et al. reported it.\n"
                        "The U.S. FDA approved drugx on June 1, 2025.")
    assert s == ["Median PFS was 23.7 vs. 16.6 months (HR 0.70; p<.001).", "Smith J. et al. reported it.",
                 "The U.S. FDA approved drugx on June 1, 2025."]
    assert str(parse_ref("[e12-4]")) == "E12-4" and parse_ref("E12") is None and parse_ref("N7") is None


def test_raw_text_stored_and_sentences_numbered(agent, conn):
    r = agent.full(FakeLLM(conn), FakeSearch())
    e = next(e for e in r["evidence"] if e["kept"])
    src = storage.get_source(conn, e["source_id"])
    assert NUM_SENTENCE in src["raw_text"] and src["retrieved_at"] and src["fetched_via"] == "tavily_extract_basic"
    sents = storage.get_sentences(conn, e["source_id"])
    assert sents[3] == NUM_SENTENCE and sents[2] == POP_SENTENCE
    for n in storage.get_notes(conn, r["task"]["id"]):
        assert n["sentence_ids"] and all(parse_ref(x) for x in n["sentence_ids"])


def test_worker_sees_sentences_with_ids_and_bad_notes_are_dropped(agent, conn):
    def notes(call):
        from conftest import sentence_lines
        lines = dict(sentence_lines(call.user))
        ref = next(k for k, v in lines.items() if v == NUM_SENTENCE)
        return {"notes": [{"text": GOOD_TEXT, "sentence_ids": [ref]},
                          {"text": "PFS was 30.1 months.", "sentence_ids": [ref]},       # number not in sentence
                          {"text": "Something", "sentence_ids": ["E999-1", "N3"]}],        # unknown IDs
                "evidence": []}

    llm = FakeLLM(conn, WorkerNotes=notes)
    r = agent.full(llm, FakeSearch())
    assert all("[E" in u and "-1] " in u for u in llm.users("WorkerNotes"))
    dropped = [t["detail"]["dropped"] for t in trace_steps(r, "notes")]
    assert all(d.get("unverified_number") == 1 and d.get("no_valid_citation") == 1 for d in dropped)
    assert {n["text"] for n in storage.get_notes(conn, r["task"]["id"])} == {GOOD_TEXT}


# --- gate 1: numbers -----------------------------------------------------------------

def test_number_normalization():
    sent = ["The hazard ratio was 0.70 (95% CI, 0.58 to 0.85; P < .001) with 12-month PFS of 61 percent in 1,024 "
            "patients, reported on June 1, 2025."]
    assert numbers.check("HR .70 (95% CI .58-.85), p<0.001, 12 months, 61%, n=1024, June 2025", sent).status == "pass"
    assert numbers.check("HR 0.70 at 12%", sent).status == "fail"  # 12 months is not 12%
    assert numbers.check("p=0.001", sent).status == "fail"  # comparator matters
    assert numbers.check("approved in July 2025", sent).status == "fail"
    assert numbers.check("Phase 3 trial in stage IV, exon 19, NCT01234567", []).status == "n/a"
    assert numbers.check("HR .55", sent).status == "fail"  # leading-dot decimals are checked


def test_ci_ordering_violation_caught():
    sent = ["HR 0.70 (95% CI 0.58-0.85) overall."]
    bad = numbers.check("HR 0.70 (95% CI 0.85-0.58)", sent)
    assert bad.status == "fail" and "out of order" in bad.problems[0]
    outside = numbers.check("HR 0.70 (95% CI 0.72 to 0.85)", ["HR 0.70; 95% CI 0.72 to 0.85."])
    assert outside.status == "fail" and "lower 0.72 < point 0.7" in outside.problems[0]
    assert numbers.check("HR 0.70 (95% CI 0.58-0.85)", sent).status == "pass"


def test_number_mismatch_caught_before_entailment_and_repaired(agent, conn):
    llm = FakeLLM(conn, Draft=draft_with("PFS hazard ratio was 0.68 with drugx."))
    r = agent.full(llm, FakeSearch())
    first = gate_events(r, "numbers")[0]
    assert first["phase"] == "initial" and first["status"] == "fail" and "'0.68'" in first["problems"][0]
    assert all("0.68" not in u for u in llm.users("Entailment"))  # no LLM call for a failed number
    claim = active(r)[0]
    assert claim["revised"] and claim["text"] == GOOD_TEXT and claim["verdict"] == "supported"
    assert claim["original_text"] == "PFS hazard ratio was 0.68 with drugx."
    assert [e["action"] for e in claim["repair_log"]] == ["recite"]


def test_repair_recites_in_code_when_numbers_exist_elsewhere(agent, conn):
    """Claim's numbers are right but it cites only the population sentence: code re-cites, no repair LLM call."""
    llm = FakeLLM(conn, Draft=draft_with(GOOD_TEXT, lambda c: [sid(c, num_notes(c)[0], POP_SENTENCE)]))
    r = agent.full(llm, FakeSearch())
    claim = active(r)[0]
    assert llm.count("RepairAction") == 0 and claim["verdict"] == "supported"
    assert "re-cited in code" in claim["repair_log"][0]["detail"]
    assert NUM_SENTENCE in [s["text"] for x in claim["citations"] for s in x["sentences"]]


def test_repair_deletes_unsourceable_number(agent, conn):
    bad = "PFS hazard ratio was 0.68 with drugx."

    def stubborn(call):  # keeps the unsourceable number every time
        n = num_notes(call.llm.conn)[0]
        return {"action": "recite", "text": bad, "sentence_ids": n["sentence_ids"], "reason": "try again"}

    llm = FakeLLM(conn, Draft=draft_with(bad), RepairAction=stubborn)
    r = agent.full(llm, FakeSearch())
    claim = next(c for c in r["claims"] if c["section"] == "new")
    assert claim["status"] == "deleted"
    assert llm.count("RepairAction") == agent.settings.limits.max_repair_rounds == 2
    assert claim["repair_log"][-1]["action"] == "delete" and "could not be sourced" in claim["repair_log"][-1]["detail"]
    assert "0.68" not in r["task"]["final_answer"].split("Removed during verification")[0]
    assert "Removed during verification: 1 claim" in r["task"]["final_answer"]
    assert r["task"]["status"] == "insufficient_evidence"


# --- citation check (before the gates) ----------------------------------------------

def test_citation_to_unknown_sentence_id_rejected_and_repaired(agent, conn):
    llm = FakeLLM(conn, Draft=draft_with(GOOD_TEXT, lambda c: ["E999-1"]))
    r = agent.full(llm, FakeSearch())
    first = [t for t in trace_steps(r, "verify") if t["detail"].get("gate") == "citation"]
    assert first and first[0]["detail"]["rejected"] == [{"id": "E999-1", "reason": "sentence ID not in the evidence store"}]
    assert llm.count("RepairAction") == 1
    claim = active(r)[0]
    assert claim["verdict"] == "supported" and claim["rejected"] == []


def test_fabricated_url_in_claim_is_rejected(agent, conn):
    text = GOOD_TEXT + " See https://www.nejm.org/doi/full/10.1056/FAKE123."
    r = agent.full(FakeLLM(conn, Draft=draft_with(text), RepairAction=lambda c: {"action": "delete", "reason": "x"}),
                   FakeSearch())
    claim = next(c for c in r["claims"] if c["section"] == "new")
    assert claim["status"] == "deleted"
    assert any(x["reason"] == "URL not in the evidence store" for x in claim["rejected"])


def test_out_of_range_sentence_index_rejected(agent, conn):
    def refs(c):
        e = num_notes(c)[0]["sentence_ids"][0].split("-")[0]
        return [f"{e}-500"]
    r = agent.full(FakeLLM(conn, Draft=draft_with(GOOD_TEXT, refs), RepairAction=lambda c: {"action": "unverified"}),
                   FakeSearch())
    claim = next(c for c in r["claims"] if c["section"] == "new")
    assert claim["status"] == "unverified" and "[UNVERIFIED]" in r["task"]["final_answer"]


# --- gate 2: identifiers -------------------------------------------------------------

def test_identifier_parsing_and_title_match():
    assert ids_in_text("see PMID: 12345678 and NCT01234567; doi 10.1056/NEJMoa2402.") == [
        Identifier("pmid", "12345678"), Identifier("nct", "NCT01234567"), Identifier("doi", "10.1056/nejmoa2402")]
    assert ids_in_url("https://pubmed.ncbi.nlm.nih.gov/38912345/") == [Identifier("pmid", "38912345")]
    assert ids_in_url("https://clinicaltrials.gov/study/NCT04035486") == [Identifier("nct", "NCT04035486")]
    assert ids_in_url("https://www.nejm.org/doi/full/10.1056/NEJMoa2403614") == [Identifier("doi", "10.1056/nejmoa2403614")]
    assert title_matches("Amivantamab plus Lazertinib in EGFR-Mutated NSCLC",
                         "Amivantamab plus Lazertinib in EGFR-Mutated NSCLC - PubMed")
    assert not title_matches("Adjuvant chemotherapy in colon cancer", "Amivantamab plus Lazertinib - PubMed")


def test_unresolvable_pmid_caught(agent, conn):
    llm = FakeLLM(conn, Draft=draft_with(GOOD_TEXT + " (PMID 12345678)"))
    r = agent.full(llm, FakeSearch(), ids=FakeIds())
    first = gate_events(r, "identifiers")[0]
    assert first["status"] == "fail" and first["checks"][0]["status"] == "not_found"
    assert all("12345678" not in u for u in llm.users("Entailment"))
    claim = active(r)[0]
    assert claim["text"] == GOOD_TEXT and claim["verdict"] == "supported"  # repaired by re-citing without the PMID


def test_pmid_that_resolves_but_is_not_a_cited_source_fails(agent, conn):
    ids = FakeIds({"pmid:12345678": "Some real but unrelated paper"})
    r = agent.full(FakeLLM(conn, Draft=draft_with(GOOD_TEXT + " (PMID 12345678)")), FakeSearch(), ids=ids)
    assert gate_events(r, "identifiers")[0]["checks"][0]["status"] == "not_in_evidence"


PUBMED = "https://pubmed.ncbi.nlm.nih.gov/38912345"


def _pubmed_search():
    hit = SearchHit(url=PUBMED, title="Drugx versus comparator in EGFR NSCLC - PubMed",
                    content="drugx egfr nsclc fda approval", score=0.95, published_date="2025-06-01")
    return FakeSearch(results={PLAN[0]: [hit], PLAN[1]: [], PLAN[2]: []})


def test_title_mismatch_caught(agent, conn):
    ids = FakeIds({"pmid:38912345": "Adjuvant chemotherapy for stage III colon cancer"})
    r = agent.full(FakeLLM(conn, RepairAction=lambda c: {"action": "unverified", "reason": "id mismatch"}),
                   _pubmed_search(), ids=ids)
    ev = gate_events(r, "identifiers")[0]
    assert ev["status"] == "fail" and ev["checks"][0]["status"] == "title_mismatch"
    claim = next(c for c in r["claims"] if c["section"] == "new")
    assert claim["citations"][0]["id_gate"]["status"] == "fail"
    assert claim["citations"][0]["entail_gate"]["status"] == "skipped"
    assert claim["status"] == "unverified"


def test_matching_source_identifier_passes(agent, conn):
    ids = FakeIds({"pmid:38912345": "Drugx versus comparator in EGFR NSCLC"})
    r = agent.full(FakeLLM(conn), _pubmed_search(), ids=ids)
    claim = active(r)[0]
    assert claim["citations"][0]["id_gate"]["status"] == "pass" and claim["verdict"] == "supported"
    assert set(ids.calls) == {"pmid:38912345"}  # once at ingest (tier), once in gate 2 (cached in production)
    ev = next(e for e in r["evidence"] if e["kept"])
    assert ev["trust_tier"] == 2 and ev["id_status"] == "confirmed" and ev["identifier"] == "PMID 38912345"


def test_identifier_service_error_is_recorded_not_failed(agent, conn):
    r = agent.full(FakeLLM(conn), _pubmed_search(), ids=FakeIds(error=True))
    claim = active(r)[0]
    assert claim["citations"][0]["id_gate"]["status"] == "error" and claim["verdict"] == "supported"


def test_resolver_caches_results(conn):
    from research_agent.backends.identifiers import IdentifierResolver

    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"result": {"123456": {"title": "A paper", "pubdate": "2025 Jun"}}})

    http = HTTP(client=None)
    res = IdentifierResolver(conn, http, attempts=1, base_delay=0, max_delay=0)

    async def go():
        http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        http._loop = asyncio.get_running_loop()
        a = await res.resolve(Identifier("pmid", "123456"))
        b = await res.resolve(Identifier("pmid", "123456"))
        return a, b

    a, b = asyncio.run(go())
    assert a.status == "found" and a.title == "A paper" and not a.cached and b.cached and len(calls) == 1


# --- gate 3: entailment --------------------------------------------------------------

def test_multi_sentence_support_accepted(agent, conn):
    """The endpoint is in one sentence and the population in another: support comes from their combination."""
    text = "In adults aged 18 years or older, median progression-free survival was 23.7 months with drugx."
    seen = []

    def judge(call):
        seen.append(call.user)
        from conftest import sentence_lines
        cited = [t for r, t in sentence_lines(call.user) if f"[{r}] (context)" not in call.user]
        ok = NUM_SENTENCE in cited and POP_SENTENCE in cited
        return {"verdict": "supported" if ok else "unsupported", "note": "combined"}

    llm = FakeLLM(conn, Draft=draft_with(text), Entailment=judge)
    r = agent.full(llm, FakeSearch())
    claim = active(r)[0]
    assert claim["verdict"] == "supported"
    assert "COMBINATION" in next(c.system for c in llm.calls if c.name == "Entailment")
    # neighbours are shown as context only
    assert "(context)" in seen[0]
    assert seen[0].index("</untrusted_source>") < seen[0].index(f"Claim: {text}")  # data first, claim last


def test_partial_on_numeric_claim_escalates_to_strong(agent, conn, settings):
    def judge(call):
        return {"verdict": "partial" if call.model == settings.models.fast else "supported", "note": "n"}

    llm = FakeLLM(conn, Entailment=judge)
    r = agent.full(llm, FakeSearch())
    models = [c.model for c in llm.calls if c.name == "Entailment"]
    assert models.count(settings.models.strong) == len(active(r))
    g = active(r)[0]["citations"][0]["entail_gate"]
    assert g["status"] == "supported" and "partial" in g["escalated"] and g["model"] == settings.models.strong


def test_fast_failure_escalates(agent, conn, settings):
    def judge(call):
        if call.model == settings.models.fast:
            return RuntimeError("fast down")
        return {"verdict": "supported", "note": "ok"}

    r = agent.full(FakeLLM(conn, Entailment=judge), FakeSearch())
    g = active(r)[0]["citations"][0]["entail_gate"]
    assert g["status"] == "supported" and "fast model failed" in g["escalated"]


def test_unsupported_claim_marked_unverified_after_two_rounds(agent, conn):
    llm = FakeLLM(conn, Entailment=lambda c: {"verdict": "unsupported", "note": "not stated"})
    r = agent.full(llm, FakeSearch())
    claims = active(r)
    assert claims and all(c["status"] == "unverified" and c["claim_type"] == "uncertain" for c in claims)
    assert llm.count("RepairAction") == 2 * len(claims)  # one call per claim per round, never the whole brief
    assert all("Failed claim:" in u for u in llm.users("RepairAction"))
    assert r["task"]["answer_json"]["bottom_line"] is None and r["task"]["status"] == "insufficient_evidence"


def test_targeted_repair_search_uses_reserved_quota(agent, conn):
    n = {"i": 0}

    def repair(call):
        n["i"] += 1
        if n["i"] == 1:
            return {"action": "search", "query": "drugx label update fda 2025", "reason": "need the label"}
        from conftest import h_repair
        return h_repair(call)

    settings = make_settings(budget={"tavily_calls": 12, "tavily_repair_reserve": 8})
    search = FakeSearch()
    llm = FakeLLM(conn, Draft=draft_with(GOOD_TEXT), RepairAction=repair,
                  Entailment=lambda c: {"verdict": "supported" if n["i"] >= 2 else "unsupported", "note": "n"})
    r = agent.full(llm, search, settings=settings)
    usage = r["task"]["usage"]
    assert "drugx label update fda 2025" in [c["query"] for c in search.search_calls]
    assert usage["tavily_repair_used"] >= 2 and usage["used"]["tavily"] <= 12
    assert any(s["text"] == "drugx label update fda 2025" for s in r["sub_queries"])
    assert active(r)[0]["verdict"] == "supported"


# --- Tavily: Extract failed_results fallback, settings, 429 --------------------------

def test_extract_failed_results_retried_with_advanced(agent, conn):
    url = f"https://www.fda.gov/{PLAN[0].replace(' ', '-')}"
    search = FakeSearch(basic_fails=(url,))
    r = agent.full(FakeLLM(conn), search)
    assert search.extract_depths == ["basic", "advanced"] and search.extract_calls[1] == [url]
    e = next(e for e in r["evidence"] if e["url"] == "https://fda.gov/" + PLAN[0].replace(" ", "-"))
    assert e["kept"] and e["fetched_via"] == "tavily_extract_advanced"


def test_extract_failure_after_advanced_is_a_coverage_gap(agent, conn):
    url = f"https://www.fda.gov/{PLAN[0].replace(' ', '-')}"
    r = agent.full(FakeLLM(conn), FakeSearch(login_blocked=(url,)))
    e = next(e for e in r["evidence"] if "fda.gov" in e["url"] and PLAN[0].split()[0] in e["url"])
    assert not e["kept"] and e["drop_reason"].startswith("source_unavailable") and "advanced" in e["drop_reason"]


def test_tavily_payload_settings_and_failed_results_on_http_200():
    seen = []

    def handler(request):
        import json
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path == "/search":
            return httpx.Response(200, json={"results": [{"url": "https://fda.gov/a", "title": "t", "score": 0.9}],
                                             "usage": {"credits": 2}})
        return httpx.Response(200, json={"results": [{"url": "https://fda.gov/a", "raw_content": "Full text here."}],
                                         "failed_results": [{"url": "https://nccn.org/x", "error": "login"}]})

    async def go():
        http = HTTP()
        http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        http._loop = asyncio.get_running_loop()
        tb = TavilyBackend(TavilyHTTP("k", http))
        s = await tb.search("q", depth="advanced", max_results=5, include_domains=["fda.gov"], start_date="2025-01-01")
        e = await tb.extract(["https://fda.gov/a", "https://nccn.org/x"])
        return s, e

    s, e = asyncio.run(go())
    search_body = seen[0][1]
    assert search_body["search_depth"] == "advanced" and search_body["auto_parameters"] is False
    assert search_body["chunks_per_source"] == 3 and search_body["include_answer"] is False
    assert search_body["include_domains"] == ["fda.gov"] and "query" in search_body
    extract_body = seen[1][1]
    assert extract_body["extract_depth"] == "basic" and "query" not in extract_body
    assert s.credits == 2 and e.pages == {"https://fda.gov/a": "Full text here."}
    assert e.failed == {"https://nccn.org/x": "login"}
    assert all(p != "/research" for p, _ in seen)


def test_429_retry_after_honored_through_http_layer():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.05"}, json={"detail": "rate limited"})
        return httpx.Response(200, json={"results": []})

    delays = []

    async def go():
        http = HTTP()
        http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        http._loop = asyncio.get_running_loop()
        client = TavilyHTTP("k", http)
        return await with_retry(lambda: client.search(query="q", search_depth="basic"), attempts=3, base_delay=0,
                                max_delay=0, on_retry=lambda exc, a, d: delays.append((type(exc).__name__, d)))

    out = asyncio.run(go())
    assert out == {"results": []} and len(attempts) == 2
    assert delays[0][0] == "RateLimitedError" and delays[0][1] >= 0.05


def test_retry_after_is_capped():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RateLimitedError("429", retry_after=999)
        return "ok"

    delays = []
    assert asyncio.run(with_retry(flaky, attempts=2, base_delay=0, max_delay=0, retry_after_max=0.01,
                                  on_retry=lambda e, a, d: delays.append(d))) == "ok"
    assert delays == [0.01]


def test_concurrency_semaphore_is_five(settings):
    assert settings.limits.max_concurrency == 5
