"""Search-backend interface. Add a new source by implementing SearchBackend; the engine is unchanged."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SearchHit:
    url: str
    title: str = ""
    content: str = ""  # snippet, used for relevance scoring before extraction
    score: float = 0.0  # backend relevance, 0..1
    published_date: str | None = None


@dataclass
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    credits: float = 0.0
    estimated: bool = False  # credits from published pricing because the API did not report usage


@dataclass
class ExtractResult:
    pages: dict[str, str] = field(default_factory=dict)  # url -> extracted text
    failed: dict[str, str] = field(default_factory=dict)  # url -> error (e.g. login wall), from failed_results
    credits: float = 0.0
    estimated: bool = False


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, *, depth: str, max_results: int, include_domains: list[str],
                     start_date: str | None) -> SearchResult: ...

    async def extract(self, urls: list[str], *, depth: str = "basic") -> ExtractResult:
        """Full page text per URL (no query-based chunking: sentences are numbered from the whole page)."""
        ...
