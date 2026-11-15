"""Cache checked before every Tavily call: search results by exact (case/space-normalized) query + params, and
extracted pages by URL + the extraction query (Tavily returns only the chunks relevant to that query). Both are
reused for `cache_ttl_days` (default 7). Empty search results are never cached."""

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
    return [SearchHit(**r) for r in rows] if rows else None  # empty rows (older DBs) count as a miss


def put_search(conn, key: str, query: str, params: dict, hits: list[SearchHit]) -> None:
    if hits:  # an empty result may be a transient miss; don't pin it for a week
        storage.put_search_cache(conn, key, query, params, [asdict(h) for h in hits])


def extract_query_key(query: str | None) -> str:
    return normalize_query(query or "")


def get_pages(conn, urls: list[str], query: str | None, ttl_days: int) -> dict[str, str]:
    return storage.get_extract_cache(conn, urls, extract_query_key(query), ttl_days)


def put_pages(conn, pages: dict[str, str], query: str | None) -> None:
    if pages:
        storage.put_extract_cache(conn, pages, extract_query_key(query))
