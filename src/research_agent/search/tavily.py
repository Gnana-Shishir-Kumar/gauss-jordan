"""Tavily implementation of SearchBackend (search + extract)."""

from __future__ import annotations

from typing import Any

from ..config import Settings
from .base import ExtractResult, SearchHit


class TavilyBackend:
    name = "tavily"

    def __init__(
        self,
        client: Any,
        *,
        search_depth: str = "advanced",
        chunks_per_source: int = 3,
        search_timeout_s: float = 30.0,
        extract_timeout_s: float = 45.0,
    ):
        self.client = client
        self.search_depth = search_depth
        self.chunks_per_source = chunks_per_source
        self.search_timeout_s = search_timeout_s
        self.extract_timeout_s = extract_timeout_s

    @classmethod
    def from_settings(cls, settings: Settings) -> "TavilyBackend":
        from tavily import AsyncTavilyClient

        return cls(
            AsyncTavilyClient(api_key=settings.tavily_api_key),
            search_depth=settings.tavily_search_depth,
            chunks_per_source=settings.tavily_chunks_per_source,
            search_timeout_s=settings.network.search_timeout_s,
            extract_timeout_s=settings.network.extract_timeout_s,
        )

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        resp = await self.client.search(
            query=query,
            max_results=max_results,
            search_depth=self.search_depth,
            timeout=self.search_timeout_s,
        )
        hits: list[SearchHit] = []
        for r in (resp or {}).get("results", []) or []:
            url = r.get("url")
            if not url:
                continue
            try:
                score = float(r.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            hits.append(
                SearchHit(
                    url=url,
                    title=r.get("title") or "",
                    content=r.get("content") or "",
                    score=max(0.0, min(1.0, score)),
                    published_date=r.get("published_date"),
                )
            )
        return hits[:max_results]

    async def extract(self, urls: list[str], query: str | None = None) -> ExtractResult:
        if not urls:
            return ExtractResult()
        kwargs: dict[str, Any] = dict(urls=urls, format="text", timeout=self.extract_timeout_s)
        if query:
            kwargs.update(query=query, chunks_per_source=self.chunks_per_source)
        resp = await self.client.extract(**kwargs) or {}
        pages = {
            r["url"]: r.get("raw_content") or ""
            for r in resp.get("results", []) or []
            if r.get("url")
        }
        failed = {
            f.get("url", ""): str(f.get("error") or "extraction failed")
            for f in resp.get("failed_results", []) or []
        }
        return ExtractResult(pages=pages, failed=failed)
