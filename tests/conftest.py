"""Fakes for Token Factory (LLMClient), Tavily (SearchBackend) and the identifier resolver, plus fixtures.

FakeSearch mimics Tavily's include_domains restrict mode and returns deterministic oncology pages (full text).
FakeLLM answers each schema with a default handler that reads its prompt like a real model would: sentence lines
look like "[E3-2] text" inside <untrusted_source id="E3"> blocks.
FakeIds resolves identifiers from a dict; anything not in it is not_found.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from research_agent import storage
from research_agent.backends.identifiers import Resolution
from research_agent.backends.llm import LLMResult
from research_agent.backends.search_base import ExtractResult, SearchHit, SearchResult
from research_agent.config import Budget, Limits, Network, Settings
from research_agent.engine import prepare, run, run_all

TODAY = date(2026, 11, 15)
QUESTION = "What has changed for first-line EGFR-mutant metastatic NSCLC since January 2025?"
PLAN = ["drugx egfr nsclc fda approval", "drugx egfr nsclc phase 3 trial results",
        "egfr nsclc first line guideline update"]

NUM_SENTENCE = "Median progression-free survival was 23.7 months versus 16.6 months (HR 0.70; p<0.001)."
POP_SENTENCE = ("Approval was based on a phase 3 randomized trial in adults aged 18 years or older "
                "in North America and Asia.")
POPULATION = "adults aged 18 years or older in North America and Asia"
TIER3_SENTENCE = "Drugx showed an objective response rate of 61% in an early analysis."
GOOD_TEXT = "Progression-free survival was 23.7 months versus 16.6 months with drugx (HR 0.70)."

BLOCK_RE = re.compile(r'<untrusted_source id="E(\d+)"([^>]*)>\n(.*?)\n</untrusted_source>', re.S)
LINE_RE = re.compile(r"^\[(E\d+-\d+)\](?: \(context\))? (.*)$", re.M)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def page_for(url: str) -> str:
    if "onclive.com" in url:
        return f"{TIER3_SENTENCE} The company said more data are expected soon. Source page {url}."
    return (f"On June 1, 2025, regulators and investigators reported new data on drugx for EGFR-mutated "
            f"non-small cell lung cancer. {POP_SENTENCE} {NUM_SENTENCE} Source page {url}.")


def default_url(domain: str, query: str) -> str:
    """Journal pages carry a DOI in the URL (as nejm.org does), so their identifier can be confirmed."""
    if domain == "nejm.org":
        return f"https://www.nejm.org/doi/full/10.1056/{slug(query)}"
    return f"https://www.{domain}/{slug(query)}"


def _domain_ok(url: str, domains: list[str]) -> bool:
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    return any(host == d or host.endswith("." + d) for d in domains)


class FakeSearch:
    name = "fake"

    def __init__(self, *, results: dict[str, list[SearchHit]] | None = None, fail: dict[str, Exception] | None = None,
                 domains: tuple[str, ...] = ("fda.gov", "nejm.org", "onclive.com"), pages: dict[str, str] | None = None,
                 login_blocked: tuple[str, ...] = (), basic_fails: tuple[str, ...] = (),
                 extract_error: Exception | None = None, empty: bool = False):
        self.results = results or {}
        self.fail = fail or {}
        self.domains = domains
        self.pages = pages or {}
        self.login_blocked = set(login_blocked)  # fail at every depth
        self.basic_fails = set(basic_fails)  # fail at basic depth, succeed at advanced
        self.extract_error = extract_error
        self.empty = empty
        self.search_calls: list[dict] = []
        self.extract_calls: list[list[str]] = []
        self.extract_depths: list[str] = []

    def default_hits(self, query: str) -> list[SearchHit]:
        return [SearchHit(url=default_url(d, query), title=f"Drugx trial report {slug(query)}" if d == "nejm.org"
                          else f"{d} {query}", content=f"{query} snippet", score=0.8, published_date="2025-06-01")
                for d in self.domains]

    async def search(self, query, *, depth, max_results, include_domains, start_date):
        self.search_calls.append(dict(query=query, depth=depth, include_domains=list(include_domains),
                                      start_date=start_date))
        if query in self.fail:
            raise self.fail[query]
        hits = [] if self.empty else self.results.get(query, self.default_hits(query))
        if include_domains:  # Tavily restrict mode; no list means no domain filter
            hits = [h for h in hits if _domain_ok(h.url, include_domains)]
        return SearchResult(hits=hits[:max_results], credits=2.0 if depth == "advanced" else 1.0)

    async def extract(self, urls, *, depth="basic"):
        self.extract_calls.append(list(urls))
        self.extract_depths.append(depth)
        if self.extract_error:
            raise self.extract_error
        blocked = self.login_blocked | (self.basic_fails if depth == "basic" else set())
        pages = {u: self.pages.get(u, page_for(u)) for u in urls if u not in blocked}
        failed = {u: "403 Forbidden: login required" for u in urls if u in blocked}
        return ExtractResult(pages=pages, failed=failed, credits=float(math.ceil(len(pages) / 5)))


class FakeIds:
    """Identifier resolver: {"pmid:123": "Title", ...}; unknown identifiers are not_found."""

    def __init__(self, known: dict[str, str] | None = None, error: bool = False, nejm: bool = True):
        self.known = known or {}
        self.nejm = nejm  # resolve the fake nejm.org DOIs (10.1056/<slug>) to their page titles
        self.error = error
        self.calls: list[str] = []

    async def resolve(self, ident, take=None):
        self.calls.append(f"{ident.kind}:{ident.value}")
        if take is not None and not take({"pmid": "pubmed", "nct": "ctgov", "doi": "crossref"}[ident.kind]):
            return Resolution(ident, "skipped", detail="quota")
        if self.error:
            return Resolution(ident, "error", detail="service down")
        title = self.known.get(f"{ident.kind}:{ident.value}")
        if title is None and self.nejm and ident.kind == "doi" and ident.value.startswith("10.1056/"):
            title = f"Drugx trial report {ident.value.split('/', 1)[1]}"
        return Resolution(ident, "found", title) if title else Resolution(ident, "not_found", detail="no record")


class FakeAPI:
    """A structured backend (PubMed / ClinicalTrials.gov / openFDA) serving fixed texts by URL substring."""

    def __init__(self, name, api, marker, texts=None, pmc=None):
        self.name, self.api, self.marker = name, api, marker
        self.texts = texts or {}
        self.pmc = pmc or {}  # identifier value -> full text (PubMed Central fallback)
        self.calls: list[str] = []

    def handles(self, url):
        return self.marker in (url or "")

    async def extract(self, urls, *, depth="basic"):
        self.calls += list(urls)
        pages = {u: t for u in urls for k, t in self.texts.items() if k in u}
        return ExtractResult(pages=pages, failed={u: "no record" for u in urls if u not in pages})

    async def pmc_for(self, kind, value):
        self.calls.append(f"pmc_for:{value}")
        return f"PMC{abs(hash(value)) % 10**6}" if value in self.pmc else None

    async def pmc_text(self, pmcid):
        for v, t in self.pmc.items():
            if pmcid == f"PMC{abs(hash(v)) % 10**6}":
                return t
        return None

    @staticmethod
    def approvals_url(generic):
        return f'https://api.fda.gov/drug/drugsfda.json?search=openfda.generic_name:"{generic}"&limit=5'


class FakeStructured:
    def __init__(self, pubmed=None, ctgov=None, openfda=None, pmc=None):
        self.pubmed = FakeAPI("pubmed", "pubmed", "pubmed.ncbi.nlm.nih.gov", pubmed, pmc)
        self.ctgov = FakeAPI("ctgov", "ctgov", "clinicaltrials.gov/study/", ctgov)
        self.openfda = FakeAPI("openfda", "openfda", "api.fda.gov/drug/drugsfda.json", openfda)
        self.crossref = None

    def for_url(self, url):
        for b in (self.pubmed, self.ctgov, self.openfda):
            if b.handles(url):
                return b
        return None


# --- fake LLM ------------------------------------------------------------------

def blocks(user: str) -> list[tuple[int, str]]:
    """(evidence id, block body) for each source block in a prompt."""
    return [(int(m.group(1)), m.group(3)) for m in BLOCK_RE.finditer(user)]


def sentence_lines(user: str) -> list[tuple[str, str]]:
    """(sentence ID, text) for every sentence line in a prompt."""
    return LINE_RE.findall(user)


def latest_task(conn) -> int:
    return conn.execute("SELECT max(id) FROM tasks").fetchone()[0]


def num_notes(conn) -> list[dict]:
    """Notes whose cited sentence is NUM_SENTENCE (the drugx result)."""
    return [n for n in storage.get_notes(conn, latest_task(conn)) if n["text"] == GOOD_TEXT]


def sid(conn, note: dict, sentence: str) -> str:
    """The sentence ID in a note's citations whose text is `sentence`."""
    tid = latest_task(conn)
    for ref in note["sentence_ids"]:
        e, i = ref[1:].split("-")
        if storage.evidence_sentences(conn, tid, int(e), [int(i)]).get(int(i)) == sentence:
            return ref
    raise AssertionError(f"{sentence!r} not cited by note {note['id']}")


def h_brief(call):
    return {"cancer_type": "non-small cell lung cancer", "setting": "metastatic, first line, EGFR mutation",
            "question": "What changed for first-line therapy?", "anchor_date": "2025-01-01",
            "aspects": ["regulatory approvals", "phase 3 trial results"]}


def h_plan(call):
    return {"queries": PLAN}


def h_notes(call):
    notes, meta = [], []
    for ref, text in sentence_lines(call.user):
        if text == NUM_SENTENCE:
            pop = next((r for r, t in sentence_lines(call.user) if t == POP_SENTENCE and r.split("-")[0] == ref.split("-")[0]), None)
            notes.append({"text": GOOD_TEXT, "sentence_ids": [ref] + ([pop] if pop else [])})
        elif text == TIER3_SENTENCE:
            notes.append({"text": "Drugx showed an objective response rate of 61% in an early analysis.",
                          "sentence_ids": [ref]})
    for eid, _ in blocks(call.user):
        meta.append({"evidence_id": eid, "source_type": None, "evidence_date": None})
    return {"notes": notes, "evidence": meta}


def h_coverage_complete(call):
    conn = call.llm.conn
    tid = latest_task(conn)
    ids = sorted({i for n in storage.get_notes(conn, tid) for i in n["evidence_ids"]})
    aspects = storage.get_task(conn, tid)["brief"]["aspects"]
    return {"aspects": [{"aspect": a, "status": "covered" if ids else "thin", "evidence_ids": ids} for a in aspects],
            "follow_up_queries": []}


def good_item(conn, note: dict) -> dict:
    return {"text": GOOD_TEXT, "claim_type": "fact", "sentence_ids": note["sentence_ids"], "date": "2025-06-01",
            "evidence_level": "phase 3 randomized trial", "population": POPULATION}


def h_draft(call):
    conn = call.llm.conn
    return {"sufficient": True,
            "bottom_line": {"text": "Drugx improved progression-free survival.", "confidence": "high",
                            "reason": "phase 3 data"},
            "items": [good_item(conn, n) for n in num_notes(conn)[:2]],
            "not_established": ["overall survival benefit"],
            "check_before_acting": ["confirm label wording on fda.gov"]}


def h_repair(call):
    """Re-cite the claim to the first drugx note, with the good text."""
    n = num_notes(call.llm.conn)[0]
    return {"action": "recite", "text": GOOD_TEXT, "sentence_ids": n["sentence_ids"], "reason": "fixed"}


def h_bottom_line(call):
    return {"text": "Drugx improved progression-free survival to 23.7 months.", "confidence": "high",
            "reason": "rewritten from verified items"}


def h_verify(call):
    return {"verdict": "supported", "note": "stated"}


class FakeLLM:
    def __init__(self, conn, **overrides: Callable):
        self.conn = conn
        self.handlers: dict[str, Callable] = {
            "Brief": h_brief, "Plan": h_plan, "WorkerNotes": h_notes, "Coverage": h_coverage_complete,
            "Draft": h_draft, "RepairAction": h_repair, "Entailment": h_verify, "BottomLine": h_bottom_line,
            **overrides,
        }
        self.calls: list[SimpleNamespace] = []

    async def complete_json(self, *, model, system, user, schema, context=None):
        call = SimpleNamespace(name=schema.__name__, model=model, system=system, user=user, context=context, llm=self)
        self.calls.append(call)
        out = self.handlers[schema.__name__](call)
        if isinstance(out, Exception):
            raise out
        return LLMResult(data=schema.model_validate(out, context=context), input_tokens=100, output_tokens=20,
                         latency_ms=5, model=model)

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c.name == name)

    def users(self, name: str) -> list[str]:
        return [c.user for c in self.calls if c.name == name]


# --- fixtures ------------------------------------------------------------------

def make_settings(budget: dict | None = None, **limits: Any) -> Settings:
    s = Settings(network=Network(max_retries=2, backoff_base_s=0, backoff_max_s=0, search_timeout_s=2,
                                 extract_timeout_s=2, llm_timeout_s=2, patient_waits_s=(0, 0)),
                 budget=Budget(time_s=60, **(budget or {})))
    return replace(s, limits=replace(Limits(), **limits)) if limits else s


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def conn():
    c = storage.connect(":memory:")
    yield c
    c.close()


class Agent:
    def __init__(self, conn, settings):
        self.conn, self.settings = conn, settings
        self.ids = FakeIds()
        self.structured = FakeStructured()

    def kw(self, llm, search, settings=None, events=None, ids=None, structured=None):
        return dict(llm=llm, search=search, conn=self.conn, settings=settings or self.settings,
                    on_event=events.append if events is not None else None, ids=ids or self.ids,
                    structured=structured or self.structured)

    def prepare(self, llm, search, question=QUESTION, **kw):
        kw = self.kw(llm, search, **kw)
        kw.pop("ids")
        kw.pop("structured")
        return asyncio.run(prepare(question, today=TODAY, **kw))

    def run(self, task_id, llm, search, **kw):
        return asyncio.run(run(task_id, **self.kw(llm, search, **kw)))

    def full(self, llm, search, question=QUESTION, **kw):
        return asyncio.run(run_all(question, today=TODAY, **self.kw(llm, search, **kw)))


@pytest.fixture
def agent(conn, settings) -> Agent:
    return Agent(conn, settings)


def trace_steps(report, step: str) -> list[dict]:
    return [t for t in report["trace"] if t["step"] == step]


def active(report, section: str = "new") -> list[dict]:
    return [c for c in report["claims"] if c["section"] == section and c["status"] != "deleted"]
