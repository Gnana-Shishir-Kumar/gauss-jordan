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


@dataclass
class ExtractResult:
    pages: dict[str, str] = field(default_factory=dict)  # url -> extracted text
    failed: dict[str, str] = field(default_factory=dict)  # url -> error (e.g. login wall)
    credits: float = 0.0


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, *, depth: str, max_results: int, include_domains: list[str],
                     start_date: str | None) -> SearchResult: ...

    async def extract(self, urls: list[str], *, query: str | None = None) -> ExtractResult: ...
