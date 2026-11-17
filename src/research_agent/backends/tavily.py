"""Tavily implementation of SearchBackend: /search and /extract only (never /research).

Called over HTTP directly (httpx) rather than through tavily-python so that a 429's Retry-After header reaches the
shared retry policy. Every call sets its parameters explicitly:
  search:  search_depth explicit, auto_parameters off (so cost never silently doubles), include_answer and
           include_raw_content off (we never verify against Tavily's answer or snippets), chunks_per_source only on
           advanced depth, include_domains as a strict filter, start_date from the brief's anchor.
  extract: full page text (no query, so no chunking), extract_depth explicit; failed_results is always read, even on
           HTTP 200, and returned per URL so the engine can retry with advanced extraction and then record a gap.
Credits come from include_usage; if Tavily omits usage we fall back to its published pricing and mark it estimated.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import Settings
from .http import HTTP
from .search_base import ExtractResult, SearchHit, SearchResult

API = "https://api.tavily.com"
_SEARCH_COST = {"basic": 1.0, "fast": 1.0, "ultra-fast": 1.0, "advanced": 2.0}
_EXTRACT_PER = {"basic": 5, "advanced": 5}  # URLs per credit (advanced costs 2 per 5)
_EXTRACT_COST = {"basic": 1.0, "advanced": 2.0}


def _credits(resp: dict, fallback: float) -> tuple[float, bool]:
    usage = (resp or {}).get("usage") or {}
    try:
        return float(usage["credits"]), False
    except (KeyError, TypeError, ValueError):
        return fallback, True


class TavilyHTTP:
    """Minimal async client for the two endpoints we use. Same call shape as tavily-python's client."""

    def __init__(self, api_key: str, http: HTTP | None = None):
        self._key = api_key
        self.http = http or HTTP()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"}

    async def search(self, **payload: Any) -> dict:
        payload.pop("timeout", None)
        return await self.http.post_json(f"{API}/search", payload, self._headers(), label="Tavily search")

    async def extract(self, **payload: Any) -> dict:
        payload.pop("timeout", None)
        return await self.http.post_json(f"{API}/extract", payload, self._headers(), label="Tavily extract")


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
        net = settings.network
        http = HTTP(timeout_s=max(net.search_timeout_s, net.extract_timeout_s))
        return cls(TavilyHTTP(settings.tavily_api_key, http), search_timeout_s=net.search_timeout_s,
                   extract_timeout_s=net.extract_timeout_s, chunks_per_source=settings.limits.chunks_per_source)

    async def search(self, query: str, *, depth: str, max_results: int, include_domains: list[str],
                     start_date: str | None) -> SearchResult:
        kwargs: dict[str, Any] = dict(query=query, search_depth=depth, max_results=max_results, topic="general",
                                      auto_parameters=False, include_answer=False, include_raw_content=False,
                                      include_usage=True, timeout=self.search_timeout_s)
        if depth == "advanced":
            kwargs["chunks_per_source"] = self.chunks_per_source
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
        credits, estimated = _credits(resp, _SEARCH_COST.get(depth, 1.0))
        return SearchResult(hits=hits[:max_results], credits=credits, estimated=estimated)

    async def extract(self, urls: list[str], *, depth: str = "basic") -> ExtractResult:
        if not urls:
            return ExtractResult()
        kwargs: dict[str, Any] = dict(urls=list(urls), extract_depth=depth, format="text", include_images=False,
                                      include_usage=True, timeout=self.extract_timeout_s)
        resp = await self.client.extract(**kwargs) or {}
        pages = {r["url"]: r.get("raw_content") or "" for r in resp.get("results", []) or [] if r.get("url")}
        failed = {u: "empty content" for u, t in pages.items() if not t.strip()}
        pages = {u: t for u, t in pages.items() if t.strip()}
        # failed_results is inspected even when the HTTP call succeeded.
        for f in resp.get("failed_results", []) or []:
            if isinstance(f, dict) and f.get("url"):
                failed[f["url"]] = str(f.get("error") or "extraction failed")
            elif isinstance(f, str):
                failed[f] = "extraction failed"
        for u in urls:
            if u not in pages and u not in failed:
                failed[u] = "no content returned"
        fallback = math.ceil(len(pages) / _EXTRACT_PER.get(depth, 5)) * _EXTRACT_COST.get(depth, 1.0)
        credits, estimated = _credits(resp, fallback)
        return ExtractResult(pages=pages, failed=failed, credits=credits, estimated=estimated)
