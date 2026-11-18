"""Structured ground-truth sources: PubMed/PMC, ClinicalTrials.gov, openFDA, Crossref; routing and dates."""

import asyncio
import json

import httpx
from conftest import PLAN, FakeLLM, FakeSearch, FakeStructured, make_settings, trace_steps

from research_agent.backends.http import HTTP
from research_agent.backends.search_base import SearchHit
from research_agent.backends.structured import (CTGovBackend, OpenFDABackend, PubMedBackend, drug_names,
                                                study_text)
from research_agent.engine.research import pubmed_date

PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/39389055"
CTGOV_URL = "https://clinicaltrials.gov/study/NCT03653546"
PUBMED_TEXT = ("First-line zorifertinib for EGFR-mutant NSCLC with CNS metastases: the phase 3 EVEREST trial.\n"
               "Published in Med on 2024-10-09 (PMID 39389055).\n"
               "Median progression-free survival was 9.6 months versus 6.9 months (HR 0.719; p=0.024).")
CTGOV_TEXT = ("A Phase II/III Study of AZD3759 (NCT03653546).\nNCT03653546 overall status is completed; phase: phase 2, "
              "phase 3.\nNCT03653546 last update posted: 2024-11-04.\nNCT03653546 has results posted: no.")


def mocked(handler):
    http = HTTP()

    async def bind():
        http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        http._loop = asyncio.get_running_loop()
    return http, bind


# --- backends against mocked APIs ----------------------------------------------------

def test_pubmed_record_and_pmc_full_text():
    efetch_pubmed = """<PubmedArticleSet><PubmedArticle><MedlineCitation><Article>
      <Journal><Title>Med</Title><JournalIssue><PubDate><Year>2024</Year><Month>Dec</Month></PubDate></JournalIssue></Journal>
      <ArticleTitle>Zorifertinib in EGFR NSCLC.</ArticleTitle>
      <Abstract><AbstractText Label="RESULTS">Median PFS was 9.6 months.</AbstractText></Abstract>
      <ArticleDate><Year>2024</Year><Month>10</Month><Day>09</Day></ArticleDate>
      <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
      </Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1016/x</ArticleId>
      <ArticleId IdType="pmc">PMC123</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
    efetch_pmc = """<pmc-articleset><article><front><article-meta><title-group><article-title>Zorifertinib full
      </article-title></title-group></article-meta></front><body><sec><p>Full text paragraph one.</p>
      <p>Second paragraph with 9.6 months.</p></sec></body></article></pmc-articleset>"""

    def handler(req):
        db = req.url.params.get("db")
        return httpx.Response(200, text=efetch_pmc if db == "pmc" else efetch_pubmed)

    http, bind = mocked(handler)

    async def go():
        await bind()
        return await PubMedBackend(http, attempts=1, base_delay=0, max_delay=0).extract([PUBMED_URL])

    res = asyncio.run(go())
    text = res.pages[PUBMED_URL]
    assert "Published in Med on 2024-10-09 (PMID 39389055; doi:10.1016/x)." in text
    assert "Results: Median PFS was 9.6 months." in text and "Full text paragraph one." in text


def test_ctgov_record_as_sentences():
    data = {"hasResults": True, "protocolSection": {
        "identificationModule": {"nctId": "NCT03653546", "briefTitle": "Brief", "officialTitle": "Official title"},
        "statusModule": {"overallStatus": "ACTIVE_NOT_RECRUITING", "whyStopped": None,
                         "lastUpdatePostDateStruct": {"date": "2025-03-04"},
                         "resultsFirstPostDateStruct": {"date": "2025-02-01"}},
        "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 439}},
        "outcomesModule": {"primaryOutcomes": [{"measure": "PFS", "timeFrame": "36 months"}]}}}
    text = study_text(data)
    assert "NCT03653546 overall status is active not recruiting; phase: phase 3." in text
    assert "NCT03653546 last update posted: 2025-03-04." in text and "has results posted: yes" in text
    assert "NCT03653546 enrollment: 439 participants." in text

    http, bind = mocked(lambda req: httpx.Response(200, json=data))

    async def go():
        await bind()
        return await CTGovBackend(http, attempts=1, base_delay=0, max_delay=0).extract([CTGOV_URL])

    assert "phase 3" in asyncio.run(go()).pages[CTGOV_URL]


def test_openfda_approval_history():
    data = {"results": [{"application_number": "NDA219008", "openfda": {"brand_name": ["LAZCLUZE"]},
                         "submissions": [{"submission_type": "ORIG", "submission_number": "1",
                                          "submission_status": "AP", "submission_status_date": "20240819",
                                          "submission_class_code_description": "Type 1 - New Molecular Entity"},
                                         {"submission_type": "SUPPL", "submission_number": "2",
                                          "submission_status": "TA", "submission_status_date": "20250101"}]}]}
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(200, json=data)

    http, bind = mocked(handler)

    async def go():
        await bind()
        b = OpenFDABackend(http, attempts=1, base_delay=0, max_delay=0)
        return await b.extract([b.approvals_url("lazertinib")])

    text = next(iter(asyncio.run(go()).pages.values()))
    assert "LAZCLUZE (NDA219008) original approval ORIG-1 (type 1 - new molecular entity) was approved on 2024-08-19." in text
    assert "SUPPL-2" not in text  # only approved (AP) submissions
    assert drug_names("FDA approves lazertinib with amivantamab-vmjw and osimertinib") == [
        "lazertinib", "amivantamab", "osimertinib"]


def test_pubmed_dates_from_the_record():
    assert pubmed_date({"sortpubdate": "2024/10/09 00:00"}) == "2024-10-09"
    assert pubmed_date({"pubdate": "2024 Oct-Dec"}) == "2024-10" and pubmed_date({"pubdate": "2024"}) == "2024"


# --- routing in the pipeline ---------------------------------------------------------

def _hits(*urls):
    return {PLAN[0]: [SearchHit(url=u, title=t, content="drugx egfr nsclc fda approval", score=0.95,
                                published_date="2023-01-01") for u, t in urls], PLAN[1]: [], PLAN[2]: []}


def test_pubmed_and_ctgov_urls_read_from_apis_not_tavily(agent, conn):
    structured = FakeStructured(pubmed={"39389055": PUBMED_TEXT}, ctgov={"NCT03653546": CTGOV_TEXT})
    search = FakeSearch(results=_hits((PUBMED_URL, "Zorifertinib EVEREST - PubMed"), (CTGOV_URL, "Study NCT03653546")))
    from conftest import FakeIds
    ids = FakeIds({"pmid:39389055": "First-line zorifertinib for EGFR-mutant NSCLC with CNS metastases",
                   "nct:NCT03653546": "A Phase II/III Study of AZD3759"})
    r = agent.full(FakeLLM(conn), search, structured=structured, ids=ids)
    assert search.extract_calls == []  # nothing went to Tavily Extract
    via = {e["url"]: e["fetched_via"] for e in r["evidence"] if e["kept"]}
    assert via == {"https://pubmed.ncbi.nlm.nih.gov/39389055": "pubmed_api",
                   "https://clinicaltrials.gov/study/NCT03653546": "ctgov_api"}
    assert r["task"]["usage"]["used"]["pubmed"] >= 1 and r["task"]["usage"]["used"]["ctgov"] >= 1


def test_dates_come_from_the_source_not_search_metadata(agent, conn):
    """Tavily said 2023-01-01; PubMed and ClinicalTrials.gov dates win."""
    from conftest import FakeIds

    class Ids(FakeIds):
        async def resolve(self, ident, take=None):
            from research_agent.backends.identifiers import Resolution
            if ident.kind == "pmid":
                return Resolution(ident, "found", "Zorifertinib EVEREST", {"sortpubdate": "2024/10/09 00:00"})
            return Resolution(ident, "found", "A Phase II/III Study of AZD3759", {"last_update_posted": "2024-11-04"})

    structured = FakeStructured(pubmed={"39389055": PUBMED_TEXT}, ctgov={"NCT03653546": CTGOV_TEXT})
    search = FakeSearch(results=_hits((PUBMED_URL, "Zorifertinib EVEREST - PubMed"), (CTGOV_URL, "Study NCT03653546")))
    r = agent.full(FakeLLM(conn), search, structured=structured, ids=Ids())
    ev = {e["url"]: e for e in r["evidence"] if e["kept"]}
    assert ev[PUBMED_URL]["source_date"] == "2024-10-09" and ev[PUBMED_URL]["date_source"] == "pubmed"
    assert ev[CTGOV_URL]["source_date"] == "2024-11-04"
    assert ev[CTGOV_URL]["date_source"] == "clinicaltrials.gov last update posted"
    assert ev[PUBMED_URL]["evidence_date"] == "2023-01-01"  # Tavily's date is kept only as a hint
    assert ev[PUBMED_URL]["trust_tier"] == 2 and ev[CTGOV_URL]["trust_tier"] == 1


def test_extract_failure_falls_back_to_pubmed_central(agent, conn):
    nejm = "https://www.nejm.org/doi/full/10.1056/nejmoa2403614"
    full = "Full text: median PFS was 23.7 months versus 16.6 months in the drugx group overall."
    structured = FakeStructured(pmc={"10.1056/nejmoa2403614": full})
    search = FakeSearch(results=_hits((nejm, "Drugx trial report nejmoa2403614")), login_blocked=(nejm,))
    r = agent.full(FakeLLM(conn), search, structured=structured)
    e = next(e for e in r["evidence"] if e["url"] == "https://nejm.org/doi/full/10.1056/nejmoa2403614")
    assert search.extract_depths == ["basic", "advanced"]
    assert e["kept"] and e["fetched_via"] == "pubmed_central"
    assert any("PubMed Central" in t["message"] for t in trace_steps(r, "extract"))


def test_fda_page_gets_drugs_at_fda_approval_record(agent, conn):
    fda = "https://www.fda.gov/drugs/fda-approves-lazertinib"
    approvals = ("Drugs@FDA (openFDA) approval history for lazertinib.\n"
                 "LAZCLUZE (NDA219008) original approval ORIG-1 (type 1 - new molecular entity) was approved on 2024-08-19.")
    structured = FakeStructured(openfda={"lazertinib": approvals})
    search = FakeSearch(results=_hits((fda, "FDA approves lazertinib with amivantamab-vmjw")))
    r = agent.full(FakeLLM(conn), search, structured=structured)
    rec = next(e for e in r["evidence"] if e["url"].startswith("https://api.fda.gov/drug/drugsfda.json"))
    assert rec["kept"] and rec["trust_tier"] == 1 and rec["source_type"] == "regulatory_action"
    assert rec["source_date"] == "2024-08-19" and rec["date_source"].startswith("Drugs@FDA")
    assert r["task"]["usage"]["used"]["openfda"] >= 1


def test_structured_quota_falls_back_to_tavily(agent, conn):
    structured = FakeStructured(pubmed={"39389055": PUBMED_TEXT})
    search = FakeSearch(results=_hits((PUBMED_URL, "Zorifertinib - PubMed")))
    settings = make_settings(budget={"pubmed_calls": 0})
    r = agent.full(FakeLLM(conn), search, structured=structured, settings=settings)
    assert structured.pubmed.calls == [] and search.extract_calls == [[PUBMED_URL]]
    assert any("pubmed quota reached" in t["message"] for t in trace_steps(r, "extract"))


def test_crossref_search_backend():
    from research_agent.backends.structured import CrossrefBackend
    body = {"message": {"items": [{"DOI": "10.1056/NEJMoa1", "title": ["A trial"]}]}}
    http, bind = mocked(lambda req: httpx.Response(200, content=json.dumps(body)))

    async def go():
        await bind()
        return await CrossrefBackend(http, attempts=1, base_delay=0, max_delay=0).search("a trial", start_date="2024-01-01")

    hits = asyncio.run(go()).hits
    assert hits[0].url == "https://doi.org/10.1056/NEJMoa1" and hits[0].title == "A trial"



# --- conflicts between tier 1-2 sources ----------------------------------------------

def test_detect_within_one_claim():
    from research_agent.engine.conflicts import detect

    def cit(eid, tier, text, date):
        return {"evidence_id": eid, "verdict": "supported", "trust_tier": tier, "url": f"https://x/{eid}",
                "title": f"S{eid}", "source_date": date, "sentences": [{"id": f"E{eid}-1", "text": text}]}

    claim = {"id": 1, "verdict": "supported", "status": "active",
             "text": "Median PFS with zorifertinib was 9.6 months.",
             "citations": [cit(1, 2, "Median PFS was 9.6 months with zorifertinib.", "2024-10-09"),
                           cit(2, 1, "Median progression-free survival was 9.2 months in the label.", "2025-01-01"),
                           cit(3, 4, "Median PFS was 12 months, the company said.", "2025-02-01")]}
    k = detect([claim])
    assert len(k) == 1 and k[0]["endpoint"] == "PFS" and k[0]["kind"] == "within_claim"
    assert {s["value"] for s in k[0]["sides"]} == {"9.6", "9.2"}  # the tier 4 source is not part of it
    same = dict(claim, citations=claim["citations"][:1] + [cit(2, 1, "Median PFS was 9.6 months.", "2025-01-01")])
    assert detect([same]) == []


def test_conflicting_tier_1_2_numbers_shown_side_by_side(agent, conn):
    from conftest import FakeIds, default_url, latest_task
    from research_agent import storage

    fda = "https://www.fda.gov/zorifertinib-label"
    nejm = default_url("nejm.org", "zorifertinib trial")
    a = "In the label, median progression-free survival with zorifertinib was 9.2 months in the phase 3 study."
    b = "In the publication, median progression-free survival with zorifertinib was 9.6 months in the phase 3 study."
    hits = {PLAN[0]: [SearchHit(url=fda, title="Zorifertinib label", content="drugx egfr nsclc fda approval",
                                score=0.95, published_date="2025-03-01")],
            PLAN[1]: [SearchHit(url=nejm, title="Drugx trial report zorifertinib-trial",
                                content="drugx egfr nsclc phase 3 trial results", score=0.95,
                                published_date="2025-04-01")], PLAN[2]: []}
    search = FakeSearch(results=hits, pages={fda: f"Label update. {a} Other text here.",
                                             nejm: f"Abstract. {b} Other text here."})

    def notes(call):
        from conftest import sentence_lines
        return {"notes": [{"text": t, "sentence_ids": [r]} for r, t in sentence_lines(call.user) if "zorifertinib" in t],
                "evidence": []}

    def draft(call):
        ns = storage.get_notes(call.llm.conn, latest_task(call.llm.conn))
        return {"sufficient": True, "bottom_line": {"text": "Zorifertinib improved PFS.", "confidence": "high",
                                                    "reason": "r"},
                "items": [{"text": n["text"], "claim_type": "fact", "sentence_ids": n["sentence_ids"],
                           "date": None, "change_type": "new trial data"} for n in ns]}

    r = agent.full(FakeLLM(conn, WorkerNotes=notes, Draft=draft), search, ids=FakeIds())
    k = r["task"]["answer_json"]["conflicts_detected"]
    assert len(k) == 1 and k[0]["endpoint"] == "PFS" and k[0]["kind"] == "across_claims"
    assert sorted((s["value"], s["tier"]) for s in k[0]["sides"]) == [("9.2", 1), ("9.6", 2)]
    ans = r["task"]["final_answer"]
    sec3 = ans.split("3. SOURCES DISAGREE")[1].split("4. NOT YET")[0]
    assert "9.2 per E" in sec3 and "9.6 per E" in sec3 and "neither is picked" in sec3
    assert "2025-03-01" in sec3 and "2025-04-01" in sec3  # each side with its date
    assert ans.count("[SOURCES DISAGREE: see section 3]") == 2
    assert r["task"]["answer_json"]["bottom_line"]["confidence"] == "moderate"  # capped: sources disagree
