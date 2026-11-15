"""Regression tests for the verification, date, number-check and cache fixes (review of the v2 pipeline)."""

import re
from dataclasses import replace

from conftest import NUM_SENTENCE, FakeLLM, FakeSearch, blocks, good_item, notes_with_numbers, trace_steps

from research_agent.backends import cache
from research_agent.engine.checks import (POPULATION_NOT_STATED, before_anchor, check_design, check_item_date,
                                          check_numbers, check_population, contains, numbers_in, population_supported)
from research_agent.evidence import near_duplicate, normalize_query, site_of
from research_agent.metrics import task_counts
from research_agent.render import claim_verdict, is_flagged

# --- verdict rollup ------------------------------------------------------------------


def _claim(*verdicts):
    return {"citations": [{"evidence_id": i, "verdict": v} for i, v in enumerate(verdicts, 1)]}


def test_claim_verdict_is_worst_checked_citation():
    assert claim_verdict(_claim("supported", "unsupported")) == "unsupported"
    assert claim_verdict(_claim("supported", "partial")) == "partial"
    assert claim_verdict(_claim("supported", "source_unavailable")) == "supported"  # login wall ignored
    assert claim_verdict(_claim("source_unavailable")) == "source_unavailable"
    assert claim_verdict({"citations": []}) == "uncited"


def test_partial_is_flagged_not_verified():
    assert is_flagged(_claim("partial"))
    assert is_flagged(_claim("supported", "unsupported"))
    assert not is_flagged(_claim("supported", "supported"))


def _two_source_draft(call):
    notes = [n for n in notes_with_numbers(call.llm.conn) if NUM_SENTENCE in str(n["details"])]
    item = good_item(notes[0])
    item["evidence_ids"] = [notes[0]["evidence_ids"][0], notes[1]["evidence_ids"][0]]
    return {"sufficient": True, "bottom_line": {"text": "Drugx improved PFS.", "confidence": "high", "reason": "r"},
            "items": [item]}


def _second_source_disagrees(call):
    ids = sorted(e for e, _ in blocks(call.user))
    return {"verdicts": [{"evidence_id": e, "verdict": "supported" if e == ids[0] or len(ids) == 1 else "unsupported",
                          "note": "n"} for e in ids]}


def test_mixed_verdict_claim_goes_to_repair_and_is_fixed(agent, conn):
    llm = FakeLLM(conn, Draft=_two_source_draft, ClaimVerification=_second_source_disagrees)
    r = agent.full(llm, FakeSearch())
    assert llm.count("Repair") == 1  # one supported citation no longer hides an unsupported one
    claim = r["claims"][0]
    assert claim["revised"] and len(claim["citations"]) == 1 and claim_verdict(claim) == "supported"
    assert r["task"]["status"] == "done"


def test_mixed_verdict_claim_stays_flagged_when_repair_keeps_bad_citation(agent, conn):
    def keep_both(call):
        draft = _two_source_draft(call)["items"][0]
        ids = [int(x) for x in re.findall(r"^claim_id (\d+):", call.user, re.M)]
        return {"claims": [{**draft, "claim_id": i} for i in ids]}

    llm = FakeLLM(conn, Draft=_two_source_draft, ClaimVerification=_second_source_disagrees, Repair=keep_both)
    r = agent.full(llm, FakeSearch())
    assert "[FLAGGED: not verified]" in r["task"]["final_answer"]
    assert r["task"]["status"] == "insufficient_evidence"
    assert r["task"]["answer_json"]["bottom_line"] is None


def test_partial_after_repair_is_not_done(agent, conn):
    def partial(call):
        return {"verdicts": [{"evidence_id": e, "verdict": "partial", "note": "different population"}
                             for e, _ in blocks(call.user)]}

    r = agent.full(FakeLLM(conn, ClaimVerification=partial), FakeSearch())
    assert r["task"]["status"] == "insufficient_evidence"
    assert "[FLAGGED: not verified]" in r["task"]["final_answer"]


# --- bottom line ---------------------------------------------------------------------


def test_bottom_line_kept_when_all_items_verified_unchanged(agent, conn):
    llm = FakeLLM(conn)
    r = agent.full(llm, FakeSearch())
    assert llm.count("BottomLine") == 0
    assert r["task"]["answer_json"]["bottom_line"]["text"] == "Drugx improved progression-free survival."


def test_bottom_line_with_unverified_number_is_rewritten(agent, conn):
    def draft(call):
        from conftest import h_draft
        d = h_draft(call)
        d["bottom_line"]["text"] = "Drugx doubled survival to 99 months."
        return d

    llm = FakeLLM(conn, Draft=draft)
    r = agent.full(llm, FakeSearch())
    assert llm.count("BottomLine") == 1
    bl = r["task"]["answer_json"]["bottom_line"]
    assert "99" not in bl["text"] and "23.7" in bl["text"]
    prompt = llm.users("BottomLine")[0]
    assert "23.7 months versus 16.6 months with drugx" in prompt and "99 months" not in prompt


def test_bottom_line_rewrite_with_bad_number_falls_back_to_plain(agent, conn):
    def draft(call):
        from conftest import h_draft
        d = h_draft(call)
        d["bottom_line"]["text"] = "Survival was 99 months."
        return d

    bad = lambda call: {"text": "Survival was 42 months.", "confidence": "high", "reason": "r"}  # noqa: E731
    r = agent.full(FakeLLM(conn, Draft=draft, BottomLine=bad), FakeSearch())
    bl = r["task"]["answer_json"]["bottom_line"]
    assert "42" not in bl["text"] and bl["text"].startswith("2 verified development(s) since 2025-01-01")
    assert bl["confidence"] == "low"


# --- item dates ----------------------------------------------------------------------


def test_writer_date_not_in_sources_is_replaced(agent, conn):
    def draft(call):
        from conftest import h_draft
        d = h_draft(call)
        for it in d["items"]:
            it["date"] = "2026-09-09"
        return d

    r = agent.full(FakeLLM(conn, Draft=draft), FakeSearch())
    assert {c["item_date"] for c in r["claims"]} == {"2025-06-01"}
    assert any("2026-09-09 not found" in t["message"] for t in trace_steps(r, "check"))


def test_item_dated_before_anchor_is_uncertain_and_not_new(agent, conn):
    search = FakeSearch()
    base = search.default_hits
    search.default_hits = lambda q: [replace(h, published_date="2024-05-01") for h in base(q)]

    def draft(call):
        from conftest import h_draft
        d = h_draft(call)
        for it in d["items"]:
            it["date"] = "2024-05-01"
        return d

    r = agent.full(FakeLLM(conn, Draft=draft), search)
    assert all(c["claim_type"] == "uncertain" and c["item_date"] == "2024-05-01" for c in r["claims"])
    assert "[DATED BEFORE 2025-01-01: not new]" in r["task"]["final_answer"]
    assert r["task"]["status"] == "insufficient_evidence"  # nothing new since the anchor was verified


def test_date_helpers():
    assert check_item_date("2025-06", ["2025-06-01"]) == "2025-06"
    assert check_item_date("2023-01-01", ["2025-06-01", "2025-02-01"]) == "2025-06-01"
    assert check_item_date("2025-06-01", [None]) is None
    assert before_anchor("2024", "2025-01-01") and before_anchor("2025-05", "2025-06-01")
    assert not before_anchor("2025", "2025-06-01") and not before_anchor(None, "2025-01-01")


# --- deterministic checks --------------------------------------------------------------


def test_range_and_ci_upper_bounds_are_checked():
    assert numbers_in("ORR 30-40% (95% CI 0.58-0.85)") == {"30", "40", "95", "0.58", "0.85"}
    assert numbers_in("PD-1, PD-L1, COVID-19, HER2-low, CDK4/6") == set()
    assert numbers_in("approved on 2025-06-01") == {"2025"}
    cit = [{"evidence_id": 1, "excerpt": "The ORR was 30-45% across cohorts.",
            "source_sentence": "The ORR was 30-45% across cohorts."}]
    ok, problems = check_numbers("ORR was 30-40%", cit)
    assert not ok and "40" in problems[-1]


def test_number_must_match_whole_number_and_full_sentence():
    assert not contains("ORR was 45%", "5%")
    cit = [{"evidence_id": 1, "excerpt": "ORR was 45% overall.", "source_sentence": "5%"}]
    assert not check_numbers("ORR was 5%", cit)[0]
    cit = [{"evidence_id": 1, "excerpt": "Overall the ORR was 45% in this trial.",
            "source_sentence": "ORR was 5% in this trial"}]
    assert not check_numbers("ORR was 5%", cit)[0]


def test_design_ignores_negated_words():
    assert check_design("phase 2 randomized", ["a phase 2 non-randomized study"]) == "not stated"
    assert check_design("phase 2 randomized", ["patients were not randomized in this phase 2 study"]) == "not stated"
    assert check_design("phase 2 non-randomized", ["a phase 2 non-randomized study"]) == "phase 2 non-randomized"
    assert check_design("phase 3 randomized trial", ["A phase III randomised trial"]) == "phase 3 randomized trial"


def test_population_ignores_negated_terms():
    assert not population_supported("adults with brain metastases", "We enrolled adults without brain metastases.")
    assert not population_supported("small cell lung cancer", "Adults with non-small cell lung cancer were enrolled.")
    assert population_supported("adults aged 18 or older with NSCLC", "Adults aged 18 years or older with NSCLC.")
    notes = [{"evidence_ids": [1], "details": {"population": "adults without brain metastases",
                                               "population_sentence": "We enrolled adults without brain metastases."}}]
    assert check_population("adults with brain metastases", notes, {1}) == POPULATION_NOT_STATED


# --- cache and query similarity -------------------------------------------------------


def test_cache_keys_keep_clinically_different_queries_apart():
    pairs = [("pembrolizumab stage II NSCLC", "pembrolizumab stage IV NSCLC"),
             ("PD-1 inhibitor melanoma", "PD-L1 inhibitor melanoma"),
             ("EGFR exon 19 deletion", "EGFR exon 20 insertion"),
             ("HR-positive HER2-negative breast cancer", "HR-negative HER2-positive breast cancer")]
    for a, b in pairs:
        assert normalize_query(a) != normalize_query(b)
        assert cache.search_key(a, "basic", ["fda.gov"], None, 10) != cache.search_key(b, "basic", ["fda.gov"], None, 10)
        assert not near_duplicate(a, [b], 0.7)
    assert normalize_query("  EGFR  NSCLC ") == normalize_query("egfr nsclc")
    assert near_duplicate("drugx egfr nsclc fda approval", ["drugx EGFR NSCLC FDA approval 2025"], 0.7)


def test_empty_search_results_are_not_cached(conn):
    key = cache.search_key("q", "basic", [], None, 10)
    cache.put_search(conn, key, "q", {}, [])
    assert cache.get_search(conn, key, 7) is None


def test_subdomains_of_one_site_are_not_independent():
    assert site_of("pubmed.ncbi.nlm.nih.gov") == site_of("ncbi.nlm.nih.gov") == "nih.gov"
    assert site_of("accessdata.fda.gov") == site_of("fda.gov")
    assert site_of("nejm.org") != site_of("fda.gov")


# --- eval metric ------------------------------------------------------------------------


def test_failed_run_is_not_counted_as_correct():
    report = {"claims": [], "trace": [], "task": {"status": "failed"},
              "totals": {"credits": 0, "input_tokens": 0, "output_tokens": 0}}
    assert task_counts(report, expect_insufficient=False)["insufficient_correct"] == 0
    report["task"]["status"] = "done"
    assert task_counts(report, expect_insufficient=False)["insufficient_correct"] == 1
