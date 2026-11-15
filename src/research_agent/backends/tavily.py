"""Tavily implementation of SearchBackend: search + extract only (never /research).

Domain filtering: include_domains (restrict mode is Tavily's default). Date filtering: start_date (YYYY-MM-DD,
publish date). Credits come from include_usage; if Tavily omits usage we fall back to its published pricing
(search: basic 1, advanced 2; extract: 1 per 5 successful URLs) and mark it estimated.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import Settings
from .search_base import ExtractResult, SearchHit, SearchResult

_SEARCH_COST = {"basic": 1.0, "fast": 1.0, "ultra-fast": 1.0, "advanced": 2.0}


def _credits(resp: dict, fallback: float) -> float:
    usage = (resp or {}).get("usage") or {}
    try:
        return float(usage["credits"])
    except (KeyError, TypeError, ValueError):
        return fallback


class TavilyBackend:
    name = "tavily"

    def __init__(self, client: Any, *, search_timeout_s: float = 30.0, extract_timeout_s: float = 60.0,
                 chunks_per_source: int = 3):
        self.client = client
        self.search_timeout_s = search_timeout_s
        self.extract_timeout_s = extract_timeout_s
        self.chunks_per_source = chunks_per_source

    @classmethod
    def from_settings(cls, settings: Settings) -> "TavilyBackend":
        from tavily import AsyncTavilyClient

        return cls(AsyncTavilyClient(api_key=settings.tavily_api_key),
                   search_timeout_s=settings.network.search_timeout_s,
                   extract_timeout_s=settings.network.extract_timeout_s)

    async def search(self, query: str, *, depth: str, max_results: int, include_domains: list[str],
                     start_date: str | None) -> SearchResult:
        kwargs: dict[str, Any] = dict(query=query, search_depth=depth, max_results=max_results, topic="general",
                                      include_usage=True, timeout=self.search_timeout_s)
        if include_domains:
            kwargs["include_domains"] = list(include_domains)
        if start_date:
            kwargs["start_date"] = start_date
        resp = await self.client.search(**kwargs) or {}
        hits: list[SearchHit] = []
        for r in resp.get("results", []) or []:
            url = r.get("url")
            if not url:
                continue
            try:
                score = float(r.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            hits.append(SearchHit(url=url, title=r.get("title") or "", content=r.get("content") or "",
                                  score=max(0.0, min(1.0, score)), published_date=r.get("published_date")))
        return SearchResult(hits=hits[:max_results], credits=_credits(resp, _SEARCH_COST.get(depth, 1.0)))

    async def extract(self, urls: list[str], *, query: str | None = None) -> ExtractResult:
        if not urls:
            return ExtractResult()
        kwargs: dict[str, Any] = dict(urls=list(urls), extract_depth="basic", format="text", include_usage=True,
                                      timeout=self.extract_timeout_s)
        if query:
            kwargs.update(query=query, chunks_per_source=self.chunks_per_source)
        resp = await self.client.extract(**kwargs) or {}
        pages = {r["url"]: r.get("raw_content") or "" for r in resp.get("results", []) or [] if r.get("url")}
        pages = {u: t for u, t in pages.items() if t.strip()}
        failed = {f.get("url", ""): str(f.get("error") or "extraction failed")
                  for f in resp.get("failed_results", []) or []}
        for u in urls:
            if u not in pages and u not in failed:
                failed[u] = "no content returned"
        return ExtractResult(pages=pages, failed=failed, credits=_credits(resp, math.ceil(len(pages) / 5)))
