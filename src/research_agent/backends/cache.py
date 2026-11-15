"""Cache checked before every Tavily call: search results by normalized query + params, and
extracted pages by URL, both reused for `cache_ttl_days` (default 7)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from .. import storage
from ..evidence import normalize_query
from .search_base import SearchHit


def search_key(query: str, depth: str, domains: list[str], start_date: str | None, max_results: int) -> str:
    payload = json.dumps([normalize_query(query), depth, sorted(domains), start_date, max_results])
    return hashlib.sha256(payload.encode()).hexdigest()


def get_search(conn, key: str, ttl_days: int) -> list[SearchHit] | None:
    rows = storage.get_search_cache(conn, key, ttl_days)
    return None if rows is None else [SearchHit(**r) for r in rows]


def put_search(conn, key: str, query: str, params: dict, hits: list[SearchHit]) -> None:
    storage.put_search_cache(conn, key, query, params, [asdict(h) for h in hits])


def get_pages(conn, urls: list[str], ttl_days: int) -> dict[str, str]:
    return storage.get_extract_cache(conn, urls, ttl_days)


def put_pages(conn, pages: dict[str, str]) -> None:
    if pages:
        storage.put_extract_cache(conn, pages)
