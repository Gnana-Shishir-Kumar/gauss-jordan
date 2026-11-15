"""Fakes for Token Factory (LLMClient) and Tavily (SearchBackend), plus fixtures.

FakeSearch mimics Tavily's include_domains restrict mode and returns deterministic oncology pages.
FakeLLM answers each schema with a default handler that reads the DB like a real model would read its prompt.
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


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def page_for(url: str) -> str:
    if "onclive.com" in url:
        return f"{TIER3_SENTENCE} The company said more data are expected. Source page {url}."
    return (f"On June 1, 2025, regulators and investigators reported new data on drugx for EGFR-mutated "
            f"non-small cell lung cancer. {POP_SENTENCE} {NUM_SENTENCE} Source page {url}.")


def _domain_ok(url: str, domains: list[str]) -> bool:
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    return any(host == d or host.endswith("." + d) for d in domains)


class FakeSearch:
    name = "fake"

    def __init__(self, *, results: dict[str, list[SearchHit]] | None = None, fail: dict[str, Exception] | None = None,
                 domains: tuple[str, ...] = ("fda.gov", "nejm.org", "onclive.com"), pages: dict[str, str] | None = None,
                 login_blocked: tuple[str, ...] = (), extract_error: Exception | None = None, empty: bool = False):
        self.results = results or {}
        self.fail = fail or {}
        self.domains = domains
        self.pages = pages or {}
        self.login_blocked = set(login_blocked)
        self.extract_error = extract_error
        self.empty = empty
        self.search_calls: list[dict] = []
        self.extract_calls: list[list[str]] = []

    def default_hits(self, query: str) -> list[SearchHit]:
        return [SearchHit(url=f"https://www.{d}/{slug(query)}", title=f"{d} {query}", content=f"{query} snippet",
                          score=0.8, published_date="2025-06-01") for d in self.domains]

    async def search(self, query, *, depth, max_results, include_domains, start_date):
        self.search_calls.append(dict(query=query, depth=depth, include_domains=list(include_domains),
                                      start_date=start_date))
        if query in self.fail:
            raise self.fail[query]
        hits = [] if self.empty else self.results.get(query, self.default_hits(query))
        hits = [h for h in hits if _domain_ok(h.url, include_domains)]  # Tavily restrict mode
        return SearchResult(hits=hits[:max_results], credits=2.0 if depth == "advanced" else 1.0)

    async def extract(self, urls, *, query=None):
        self.extract_calls.append(list(urls))
        if self.extract_error:
            raise self.extract_error
        pages = {u: self.pages.get(u, page_for(u)) for u in urls if u not in self.login_blocked}
        failed = {u: "403 Forbidden: login required" for u in urls if u in self.login_blocked}
        return ExtractResult(pages=pages, failed=failed, credits=float(math.ceil(len(pages) / 5)))


# --- fake LLM ------------------------------------------------------------------

def blocks(user: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(3)) for m in BLOCK_RE.finditer(user)]


def latest_task(conn) -> int:
    return conn.execute("SELECT max(id) FROM tasks").fetchone()[0]


def notes_with_numbers(conn) -> list[dict]:
    return [n for n in storage.get_notes(conn, latest_task(conn)) if n["details"].get("numbers")]


def h_brief(call):
    return {"cancer_type": "non-small cell lung cancer", "setting": "metastatic, first line, EGFR mutation",
            "question": "What changed for first-line therapy?", "anchor_date": "2025-01-01",
            "aspects": ["regulatory approvals", "phase 3 trial results"]}


def h_plan(call):
    return {"queries": PLAN}


def h_notes(call):
    notes, meta = [], []
    for eid, text in blocks(call.user):
        n: dict[str, Any] = {"text": f"Finding from E{eid}", "evidence_ids": [eid]}
        if NUM_SENTENCE in text:
            n["numbers"] = [{"value": v, "sentence": NUM_SENTENCE} for v in ("23.7", "16.6", "0.70")]
        if TIER3_SENTENCE in text:
            n["numbers"] = [{"value": "61%", "sentence": TIER3_SENTENCE}]
        if POP_SENTENCE in text:
            n.update(design="phase 3 randomized trial", population=POPULATION, population_sentence=POP_SENTENCE,
                     date="2025-06-01")
        notes.append(n)
        meta.append({"evidence_id": eid, "source_type": None, "evidence_date": None})
    return {"notes": notes, "evidence": meta}


def h_coverage_complete(call):
    conn = call.llm.conn
    tid = latest_task(conn)
    ids = sorted({i for n in storage.get_notes(conn, tid) for i in n["evidence_ids"]})
    aspects = storage.get_task(conn, tid)["brief"]["aspects"]
    return {"aspects": [{"aspect": a, "status": "covered" if ids else "thin", "evidence_ids": ids} for a in aspects],
            "follow_up_queries": []}


def good_item(note: dict) -> dict:
    eid = note["evidence_ids"][0]
    return {"text": GOOD_TEXT, "claim_type": "fact", "evidence_ids": [eid], "date": "2025-06-01",
            "evidence_level": "phase 3 randomized trial", "population": note["details"].get("population"),
            "numbers": [{"value": v, "sentence": NUM_SENTENCE, "evidence_id": eid} for v in ("23.7", "16.6", "0.70")]}


def h_draft(call):
    notes = [n for n in notes_with_numbers(call.llm.conn) if NUM_SENTENCE in str(n["details"])]
    return {"sufficient": True,
            "bottom_line": {"text": "Drugx improved progression-free survival.", "confidence": "high",
                            "reason": "phase 3 data"},
            "items": [good_item(n) for n in notes[:2]],
            "not_established": ["overall survival benefit"],
            "check_before_acting": ["confirm label wording on fda.gov"]}


def h_repair(call):
    notes = [n for n in notes_with_numbers(call.llm.conn) if NUM_SENTENCE in str(n["details"])]
    ids = [int(x) for x in re.findall(r"^claim_id (\d+):", call.user, re.M)]
    return {"claims": [{**good_item(notes[0]), "claim_id": i, "text": GOOD_TEXT + " (revised)"} for i in ids]}


def h_verify(call):
    return {"verdicts": [{"evidence_id": eid, "verdict": "supported", "note": "stated"} for eid, _ in blocks(call.user)]}


class FakeLLM:
    def __init__(self, conn, **overrides: Callable):
        self.conn = conn
        self.handlers: dict[str, Callable] = {
            "Brief": h_brief, "Plan": h_plan, "WorkerNotes": h_notes, "Coverage": h_coverage_complete,
            "Draft": h_draft, "Repair": h_repair, "ClaimVerification": h_verify, **overrides,
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

def make_settings(**limits: Any) -> Settings:
    s = Settings(network=Network(max_retries=2, backoff_base_s=0, backoff_max_s=0, search_timeout_s=2,
                                 extract_timeout_s=2, llm_timeout_s=2),
                 budget=Budget(time_s=60, credits=1000))
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

    def kw(self, llm, search, settings=None, events=None):
        return dict(llm=llm, search=search, conn=self.conn, settings=settings or self.settings,
                    on_event=events.append if events is not None else None)

    def prepare(self, llm, search, question=QUESTION, **kw):
        return asyncio.run(prepare(question, today=TODAY, **self.kw(llm, search, **kw)))

    def run(self, task_id, llm, search, **kw):
        return asyncio.run(run(task_id, **self.kw(llm, search, **kw)))

    def full(self, llm, search, question=QUESTION, **kw):
        return asyncio.run(run_all(question, today=TODAY, **self.kw(llm, search, **kw)))


@pytest.fixture
def agent(conn, settings) -> Agent:
    return Agent(conn, settings)


def trace_steps(report, step: str) -> list[dict]:
    return [t for t in report["trace"] if t["step"] == step]
