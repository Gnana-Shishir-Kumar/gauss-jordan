"""Search-backend interface. Add a new source by implementing SearchBackend; nothing else changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SearchHit:
    url: str
    title: str = ""
    content: str = ""  # search snippet; used as fallback if extraction fails
    score: float = 0.0  # backend relevance, 0..1
    published_date: str | None = None


@dataclass
class ExtractResult:
    pages: dict[str, str] = field(default_factory=dict)  # url -> extracted text
    failed: dict[str, str] = field(default_factory=dict)  # url -> error


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...

    async def extract(self, urls: list[str], query: str | None = None) -> ExtractResult: ...
