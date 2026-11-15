"""Research step: run sub-queries concurrently (search -> extract), then score/dedupe evidence."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field

from . import storage
from .context import RunContext, describe_error
from .evidence import Candidate, assess, domain_of, normalize_url
from .retry import with_retry


@dataclass
class FetchOutcome:
    sub_query_id: int
    text: str
    hit_count: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    error: str | None = None


@dataclass
class SubQueryResult:
    kept: int
    status: str  # done | weak | empty | failed

    @property
    def weakness(self) -> str:
        return {
            "failed": "search failed",
            "empty": "no results",
        }.get(self.status, f"only {self.kept} usable source(s)")


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


async def _fetch(ctx: RunContext, sq_id: int, text: str, sem: asyncio.Semaphore) -> FetchOutcome:
    net, lim = ctx.settings.network, ctx.settings.limits
    retry = dict(attempts=net.max_retries, base_delay=net.backoff_base_s, max_delay=net.backoff_max_s)

    def on_retry(step: str):
        def _cb(exc: BaseException, attempt: int, delay: float) -> None:
            ctx.note(step, f"Retry {attempt} for '{text}' after {describe_error(exc)}",
                     sub_query_id=sq_id, delay_s=round(delay, 2))
        return _cb

    async with sem:
        storage.update_sub_query(ctx.conn, sq_id, "searching")
        t0 = time.perf_counter()
        try:
            hits = await with_retry(
                lambda: ctx.search.search(text, lim.max_results_per_search),
                timeout=net.search_timeout_s + 5, on_retry=on_retry("search"), **retry,
            )
        except Exception as exc:
            err = describe_error(exc)
            await ctx.emit("search", f"Search failed for '{text}': {err}", level="error",
                           detail={"sub_query_id": sq_id, "query": text, "error": err}, latency_ms=_ms(t0))
            return FetchOutcome(sq_id, text, error=err)

        unique, seen = [], set()
        for h in hits[: lim.max_results_per_search]:
            n = normalize_url(h.url)
            if n not in seen:
                seen.add(n)
                unique.append((n, h))
        await ctx.emit(
            "search", f"{len(unique)} results for '{text}'",
            level="info" if unique else "warning",
            detail={"sub_query_id": sq_id, "query": text, "backend": ctx.search.name,
                    "results": [{"url": h.url, "title": h.title, "score": h.score} for _, h in unique]},
            latency_ms=_ms(t0),
        )
        if not unique:
            return FetchOutcome(sq_id, text)

        pages: dict[str, str] = {}
        t1 = time.perf_counter()
        try:
            ext = await with_retry(
                lambda: ctx.search.extract([h.url for _, h in unique], query=text),
                timeout=net.extract_timeout_s + 5, on_retry=on_retry("extract"), **retry,
            )
            pages = {normalize_url(u): t for u, t in ext.pages.items() if t and t.strip()}
            await ctx.emit(
                "extract", f"Extracted {len(pages)}/{len(unique)} pages for '{text}'",
                level="info" if len(pages) == len(unique) else "warning",
                detail={"sub_query_id": sq_id, "failed": ext.failed}, latency_ms=_ms(t1),
            )
        except Exception as exc:
            err = describe_error(exc)
            await ctx.emit("extract", f"Extraction failed for '{text}', using search snippets: {err}",
                           level="error", detail={"sub_query_id": sq_id, "error": err}, latency_ms=_ms(t1))

        candidates = []
        for n, h in unique:
            body = pages.get(n, "")
            candidates.append(Candidate(
                sub_query_id=sq_id, url=n, title=h.title, published_date=h.published_date,
                content=body or h.content, backend_score=h.score, extracted=bool(body),
            ))
        return FetchOutcome(sq_id, text, len(unique), candidates)


async def _record(ctx: RunContext, out: FetchOutcome, seen_urls: set[str], seen_hashes: set[str]) -> SubQueryResult:
    lim = ctx.settings.limits
    decisions = assess(out.candidates, out.text, seen_urls, seen_hashes,
                       min_relevance=lim.min_relevance, excerpt_chars=lim.excerpt_chars)
    kept = 0
    for d in decisions:
        c = d.candidate
        src = storage.upsert_source(ctx.conn, c.url, domain_of(c.url), c.title or None, c.published_date, d.content_hash)
        storage.add_evidence(ctx.conn, ctx.task_id, out.sub_query_id, src, d.excerpt, d.score, d.kept, d.drop_reason)
        kept += int(d.kept)
    if out.error:
        status = "failed"
    elif out.hit_count == 0:
        status = "empty"
    elif kept < lim.min_kept_evidence_per_sub_query:
        status = "weak"
    else:
        status = "done"
    storage.update_sub_query(ctx.conn, out.sub_query_id, status, out.hit_count)
    if decisions:
        dropped = Counter(d.drop_reason for d in decisions if not d.kept)
        await ctx.emit(
            "evidence", f"Kept {kept}/{len(decisions)} evidence items for '{out.text}'",
            level="info" if status == "done" else "warning",
            detail={"sub_query_id": out.sub_query_id, "kept": kept, "dropped": dict(dropped),
                    "snippet_fallbacks": sum(not d.candidate.extracted for d in decisions)},
        )
    return SubQueryResult(kept, status)


async def research_round(
    ctx: RunContext, sub_queries: list[tuple[int, str]], seen_urls: set[str], seen_hashes: set[str]
) -> dict[int, SubQueryResult]:
    sem = asyncio.Semaphore(ctx.settings.limits.max_concurrency)
    outcomes = await asyncio.gather(*(_fetch(ctx, i, t, sem) for i, t in sub_queries), return_exceptions=True)
    results: dict[int, SubQueryResult] = {}
    # Evidence is recorded sequentially in plan order so dedupe is deterministic.
    for (sq_id, text), out in zip(sub_queries, outcomes):
        if isinstance(out, BaseException):  # defensive: _fetch catches network errors itself
            err = describe_error(out)
            await ctx.emit("search", f"Sub-query crashed for '{text}': {err}", level="error",
                           detail={"sub_query_id": sq_id, "error": err})
            out = FetchOutcome(sq_id, text, error=err)
        results[sq_id] = await _record(ctx, out, seen_urls, seen_hashes)
    return results
