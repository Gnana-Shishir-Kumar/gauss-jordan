"""Search cache checked before every Tavily search: results by exact (case/space-normalized) query + params, reused
for `cache_ttl_days` (default 7). Empty search results are never cached. Extracted pages are cached as the sources'
stored raw text (storage.cached_texts), keyed by normalized URL."""

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
