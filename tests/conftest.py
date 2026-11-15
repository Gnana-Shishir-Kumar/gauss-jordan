"""Fakes for Token Factory (LLMClient) and Tavily (SearchBackend), plus fixtures."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

import pytest

from research_agent import storage
from research_agent.config import Network, Settings
from research_agent.llm import LLMResult
from research_agent.pipeline import run_research
from research_agent.search.base import ExtractResult, SearchHit

ID_RE = re.compile(r'id="E(\d+)"')


def ids_in(prompt: str) -> list[int]:
    return [int(x) for x in ID_RE.findall(prompt)]


def default_plan(system, user, ctx):
    return {"sub_queries": ["alpha topic facts", "beta topic facts", "gamma topic facts"]}


def default_follow_up(system, user, ctx):
    weak = re.findall(r"^- (.+?)  \(problem", user, flags=re.M)
    return {"queries": [f"{q} alternative" for q in weak]}


def default_draft(system, user, ctx):
    return {
        "sufficient": True,
        "claims": [{"text": f"Claim from E{i}", "claim_type": "fact", "evidence_ids": [i]} for i in ids_in(user)[:3]],
    }


def default_verify(system, user, ctx):
    return {"verdicts": [{"evidence_id": i, "verdict": "supported", "note": "stated"} for i in ids_in(user)]}


class FakeLLM:
    def __init__(self, **overrides: Callable):
        self.handlers: dict[str, Callable] = {
            "Plan": default_plan,
            "FollowUpPlan": default_follow_up,
            "Draft": default_draft,
            "ClaimVerification": default_verify,
            **overrides,
        }
        self.calls: list[tuple[str, str, str]] = []

    async def complete_json(self, *, model, system, user, schema, context=None):
        self.calls.append((schema.__name__, model, user))
        out = self.handlers[schema.__name__](system, user, context)
        return LLMResult(data=schema.model_validate(out, context=context),
                         input_tokens=100, output_tokens=20, latency_ms=5)

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)


def make_hits(query: str, n: int = 2, prefix: str | None = None) -> list[SearchHit]:
    slug = prefix or query.split()[0]
    return [
        SearchHit(url=f"https://{slug}{i}.example.com/page", title=f"{slug} {i}",
                  content=f"{query} snippet {i}", score=0.8)
        for i in range(n)
    ]


class FakeSearch:
    name = "fake"

    def __init__(self, results: dict[str, Any] | None = None, pages: dict[str, str] | None = None,
                 extract_error: Exception | None = None, default_hits: int = 2):
        self.results = results or {}
        self.pages = pages
        self.extract_error = extract_error
        self.default_hits = default_hits
        self.search_calls: list[str] = []
        self.extract_calls: list[list[str]] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.search_calls.append(query)
        r = self.results.get(query, "default")
        if isinstance(r, Exception):
            raise r
        if r == "default":
            r = make_hits(query, self.default_hits)
        return list(r)[:max_results]

    async def extract(self, urls: list[str], query: str | None = None) -> ExtractResult:
        self.extract_calls.append(urls)
        if self.extract_error:
            raise self.extract_error
        pages = {}
        for u in urls:
            if self.pages is not None and u in self.pages:
                pages[u] = self.pages[u]
            elif self.pages is None:
                pages[u] = f"Full article at {u}. {query} explained in depth with {query} details."
        return ExtractResult(pages=pages, failed={u: "not found" for u in urls if u not in pages})


@pytest.fixture
def settings() -> Settings:
    return Settings(network=Network(max_retries=2, backoff_base_s=0, backoff_max_s=0,
                                    search_timeout_s=2, extract_timeout_s=2, llm_timeout_s=2))


@pytest.fixture
def conn():
    c = storage.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def run(conn, settings):
    def _run(llm, search, question="What are the topic facts?", events=None):
        cb = events.append if events is not None else None
        return asyncio.run(run_research(question, llm=llm, search=search, conn=conn,
                                        settings=settings, on_event=cb))
    return _run
