"""Step 4: parallel research workers.

Per round:
  1. each worker searches its query (cache first; tier-filtered domains; start_date = anchor date),
  2. scores hits on snippets and picks its top URLs,
  3. the round's picks are extracted in shared batches (cache first; <= 20 URLs per call),
  4. evidence is scored and deduped (URL + content hash); dropped items keep a reason,
  5. each worker writes structured notes (FAST) whose numbers/design/population are checked against excerpts.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .. import prompts, storage
from ..backends import cache
from ..backends.search_base import SearchHit
from ..evidence import Candidate, assess, domain_of, normalize_url, relevance
from ..retry import with_retry
from ..schemas import WorkerNotes, parse_partial_date
from ..sources import classify, domains_for, tier_of
from .checks import contains, is_sentence, numbers_in, population_supported
from .context import RunContext, describe_error


@dataclass
class ResearchState:
    seen_urls: set[str] = field(default_factory=set)
    seen_hashes: set[str] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)


@dataclass
class Pick:
    sq_id: int
    query: str
    url: str  # normalized
    hit: SearchHit
    rel: float
    tier: int


@dataclass
class WorkerHits:
    sq_id: int
    query: str
    picks: list[Pick] = field(default_factory=list)
    off_allowlist: list[tuple[str, SearchHit]] = field(default_factory=list)
    error: str | None = None


@dataclass
class RoundResult:
    new_kept: int
    kept_by_sq: dict[int, int]


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _retry_kwargs(ctx: RunContext, step: str, label: str, sq_id: int | None = None) -> dict:
    net = ctx.settings.network

    def on_retry(exc: BaseException, attempt: int, delay: float) -> None:
        ctx.note(step, f"Retry {attempt} for {label} after {describe_error(exc)}", sub_query_id=sq_id,
                 delay_s=round(delay, 2))

    return dict(attempts=net.max_retries, base_delay=net.backoff_base_s, max_delay=net.backoff_max_s,
                on_retry=on_retry)


def _record(ctx: RunContext, sq_id: int, url: str, hit: SearchHit, tier: int | None, excerpt: str, rel: float,
            kept: bool, reason: str | None, content_hash: str | None = None) -> int:
    domain = domain_of(url)
    src = storage.upsert_source(ctx.conn, url, domain, hit.title or None, hit.published_date, content_hash, tier,
                                classify(url, domain))
    return storage.add_evidence(ctx.conn, ctx.task_id, sq_id, src, excerpt, rel, kept, reason,
                                parse_partial_date((hit.published_date or "")[:10]))


async def _search(ctx: RunContext, brief: dict, sq_id: int, query: str, depth: str, tiers: tuple[int, ...],
                  sem: asyncio.Semaphore) -> WorkerHits:
    lim, net = ctx.settings.limits, ctx.settings.network
    out = WorkerHits(sq_id, query)
    async with sem:
        if (b := ctx.budget_exhausted()):
            storage.update_sub_query(ctx.conn, sq_id, "skipped_budget")
            await ctx.emit("search", f"Skipped '{query}': {b} reached", level="warning", detail={"sub_query_id": sq_id})
            out.error = b
            return out
        storage.update_sub_query(ctx.conn, sq_id, "searching")
        domains = domains_for(tiers)
        start = brief.get("anchor_date")
        key = cache.search_key(query, depth, domains, start, lim.search_results)
        detail = {"sub_query_id": sq_id, "query": query, "depth": depth, "tiers": list(tiers), "start_date": start}
        t0 = time.perf_counter()
        hits = cache.get_search(ctx.conn, key, ctx.settings.cache_ttl_days)
        if hits is not None:
            await ctx.emit("search", f"Cache hit: {len(hits)} results for '{query}'", credits=0.0, latency_ms=_ms(t0),
                           detail={**detail, "cached": True,
                                   "results": [{"url": h.url, "score": h.score} for h in hits]})
        else:
            try:
                res = await with_retry(
                    lambda: ctx.search.search(query, depth=depth, max_results=lim.search_results,
                                              include_domains=domains, start_date=start),
                    timeout=net.search_timeout_s + 5, **_retry_kwargs(ctx, "search", f"'{query}'", sq_id))
            except Exception as exc:
                err = describe_error(exc)
                storage.update_sub_query(ctx.conn, sq_id, "failed", 0)
                await ctx.emit("search", f"Search failed for '{query}': {err}", level="error",
                               detail={**detail, "error": err}, latency_ms=_ms(t0))
                out.error = err
                return out
            hits = res.hits
            cache.put_search(ctx.conn, key, query, {"depth": depth, "tiers": list(tiers), "start_date": start}, hits)
            await ctx.emit("search", f"{len(hits)} results for '{query}' ({depth}, tiers {list(tiers)})",
                           level="info" if hits else "warning", credits=res.credits, latency_ms=_ms(t0),
                           detail={**detail, "results": [{"url": h.url, "score": h.score} for h in hits]})

    seen: set[str] = set()
    for h in hits:
        url = normalize_url(h.url)
        if url in seen:
            continue
        seen.add(url)
        tier = tier_of(domain_of(url))
        if tier is None or tier not in tiers:
            out.off_allowlist.append((url, h))
            continue
        rel = relevance(h.score, f"{query} {brief.get('question', '')}", f"{h.title} {h.content}")
        out.picks.append(Pick(sq_id, query, url, h, rel, tier))
    storage.update_sub_query(ctx.conn, sq_id, "searched" if out.picks else "empty", len(out.picks))
    return out


async def _extract(ctx: RunContext, brief: dict, picks: list[Pick]) -> tuple[dict[str, str], dict[str, str]]:
    """Cache first, then batched Tavily extract calls (<= batch size URLs each). Keyed by normalized URL + query."""
    lim, net = ctx.settings.limits, ctx.settings.network
    urls = [p.url for p in picks]
    pages = cache.get_pages(ctx.conn, urls, brief.get("question"), ctx.settings.cache_ttl_days)
    failed: dict[str, str] = {}
    if pages:
        await ctx.emit("extract", f"Cache hit: {len(pages)} extracted pages reused", credits=0.0,
                       detail={"cached_urls": sorted(pages)})
    todo = [p for p in picks if p.url not in pages]
    for i in range(0, len(todo), lim.extract_batch_size):
        batch = todo[i: i + lim.extract_batch_size]
        if (b := ctx.budget_exhausted()):
            failed.update({p.url: f"skipped: {b}" for p in batch})
            await ctx.emit("extract", f"Skipped extracting {len(batch)} URLs: {b} reached", level="warning")
            continue
        by_raw = {p.hit.url: p.url for p in batch}
        t0 = time.perf_counter()
        try:
            res = await with_retry(lambda: ctx.search.extract(list(by_raw), query=brief.get("question")),
                                   timeout=net.extract_timeout_s + 5,
                                   **_retry_kwargs(ctx, "extract", f"batch of {len(batch)}"))
        except Exception as exc:
            err = describe_error(exc)
            failed.update({p.url: err for p in batch})
            await ctx.emit("extract", f"Extraction batch failed ({len(batch)} URLs): {err}", level="error",
                           latency_ms=_ms(t0))
            continue
        got = {by_raw.get(u, normalize_url(u)): t for u, t in res.pages.items()}
        bad = {by_raw.get(u, normalize_url(u)): e for u, e in res.failed.items()}
        pages.update(got)
        failed.update(bad)
        cache.put_pages(ctx.conn, got, brief.get("question"))
        await ctx.emit("extract", f"Extracted {len(got)}/{len(batch)} URLs in one batch",
                       level="info" if not bad else "warning", credits=res.credits, latency_ms=_ms(t0),
                       detail={"urls": len(batch), "failed": bad})
    return pages, failed


async def _take_notes(ctx: RunContext, brief: dict, sq_id: int, query: str, rows: list[dict],
                      sem: asyncio.Semaphore) -> int:
    if not rows:
        return 0
    model = ctx.settings.models.fast
    try:
        async with sem:
            res = await ctx.llm.complete_json(model=model, system=prompts.NOTES_SYSTEM,
                                              user=prompts.notes_user(brief, query, rows), schema=WorkerNotes)
    except Exception as exc:
        await ctx.emit("notes", f"Notes failed for '{query}', skipped: {describe_error(exc)}", level="error",
                       detail={"sub_query_id": sq_id}, model_id=model)
        return 0

    by_id = {r["id"]: r for r in rows}
    for meta in res.data.evidence:
        r = by_id.get(meta.evidence_id)
        if not r:
            continue
        if meta.source_type and not r.get("source_type"):
            storage.set_source_type(ctx.conn, r["source_id"], meta.source_type)
        if meta.evidence_date and not r.get("evidence_date") and meta.evidence_date[:4] in r["excerpt"]:
            storage.set_evidence_date(ctx.conn, r["id"], meta.evidence_date)
    # Anything still unclassified falls back to the most conservative type (not peer reviewed).
    for r in rows:
        storage.set_source_type(ctx.conn, r["source_id"], "news")

    dropped: Counter[str] = Counter()
    kept = 0
    for note in res.data.notes:
        ids = [i for i in note.evidence_ids if i in by_id]
        if not ids:
            dropped["no_valid_evidence"] += 1
            continue
        excerpts = [by_id[i]["excerpt"] for i in ids]
        numbers = []
        for n in note.numbers:
            if (is_sentence(n.sentence) and any(contains(x, n.sentence) for x in excerpts)
                    and numbers_in(n.value) <= numbers_in(n.sentence)):
                numbers.append(n.model_dump())
            else:
                dropped["unverified_number"] += 1
        design = note.design if note.design and any(contains(x, note.design) for x in excerpts) else None
        if note.design and not design:
            dropped["unverified_design"] += 1
        population = population_sentence = None
        if note.population:
            s = note.population_sentence or ""
            if s and any(contains(x, s) for x in excerpts) and population_supported(note.population, s):
                population, population_sentence = note.population, s
            else:
                dropped["unverified_population"] += 1
        date = note.date if note.date and note.date[:4] in " ".join(excerpts) else None
        storage.add_note(ctx.conn, ctx.task_id, sq_id, note.text, ids,
                         {"numbers": numbers, "design": design, "population": population,
                          "population_sentence": population_sentence, "date": date})
        kept += 1
    await ctx.emit_llm("notes", f"{kept} notes for '{query}'", res,
                       detail={"sub_query_id": sq_id, "dropped": dict(dropped)})
    return kept


async def research_round(ctx: RunContext, brief: dict, queries: list[tuple[int, str]], round_: int, depth: str,
                         tiers: tuple[int, ...], state: ResearchState) -> RoundResult:
    lim = ctx.settings.limits
    sem = asyncio.Semaphore(lim.max_concurrency)
    await ctx.emit("round", f"Round {round_}: {len(queries)} queries, {depth} depth, tiers {list(tiers)}",
                   detail={"round": round_, "queries": [q for _, q in queries]})
    workers = await asyncio.gather(*(_search(ctx, brief, i, q, depth, tiers, sem) for i, q in queries))

    # Off-allowlist results (defensive; include_domains should prevent them) are recorded, not used.
    for w in workers:
        for url, h in w.off_allowlist:
            _record(ctx, w.sq_id, url, h, None, h.content or "", 0.0, False, "off_allowlist")

    # Select: each worker's top URLs by snippet relevance, then a per-round cap.
    chosen: list[Pick] = []
    for w in workers:
        n = 0
        for p in sorted(w.picks, key=lambda p: p.rel, reverse=True):
            if p.url in state.seen_urls:
                _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False, "duplicate_url")
            elif n < lim.extract_per_query and p.rel >= lim.min_relevance:
                chosen.append(p)
                n += 1
            else:
                _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False,
                        "low_relevance" if p.rel < lim.min_relevance else "not_selected_for_extraction")
    final: list[Pick] = []
    in_round: set[str] = set()
    for p in sorted(chosen, key=lambda p: p.rel, reverse=True):
        if p.url in in_round:
            _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False, "duplicate_url")
        elif len(final) >= lim.extract_per_round:
            _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False,
                    "not_selected_for_extraction")
        else:
            final.append(p)
            in_round.add(p.url)

    pages, failed = await _extract(ctx, brief, final)

    # Evidence per worker, in plan order so dedupe is deterministic.
    by_worker: dict[int, list[Pick]] = defaultdict(list)
    for p in final:
        by_worker[p.sq_id].append(p)
    kept_rows: dict[int, list[int]] = defaultdict(list)
    for w in workers:
        cands, picks = [], []
        for p in by_worker.get(w.sq_id, []):
            body = pages.get(p.url)
            if not body:
                # Login walls and failed extractions: recorded as source_unavailable, never used for claims.
                _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False,
                        f"source_unavailable: {failed.get(p.url, 'not extracted')}")
                state.seen_urls.add(p.url)
                continue
            cands.append(Candidate(p.sq_id, p.url, p.hit.title, p.hit.published_date, body, p.hit.score, True))
            picks.append(p)
        decisions = assess(cands, f"{w.query} {brief.get('question', '')}", state.seen_urls, state.seen_hashes,
                           min_relevance=lim.min_relevance, excerpt_chars=lim.excerpt_chars)
        for p, d in zip(picks, decisions):
            eid = _record(ctx, p.sq_id, p.url, p.hit, p.tier, d.excerpt, d.score, d.kept, d.drop_reason,
                          d.content_hash)
            if d.kept:
                kept_rows[w.sq_id].append(eid)
        if decisions:
            dropped = Counter(d.drop_reason for d in decisions if not d.kept)
            await ctx.emit("evidence", f"Kept {len(kept_rows[w.sq_id])}/{len(decisions)} for '{w.query}'",
                           detail={"sub_query_id": w.sq_id, "dropped": dict(dropped)})

    note_sem = asyncio.Semaphore(lim.max_concurrency)
    await asyncio.gather(*(
        _take_notes(ctx, brief, w.sq_id, w.query, storage.get_evidence(ctx.conn, ctx.task_id, ids=kept_rows[w.sq_id]),
                    note_sem)
        for w in workers if kept_rows.get(w.sq_id)
    ))

    kept_by_sq = {w.sq_id: len(kept_rows.get(w.sq_id, [])) for w in workers}
    for w in workers:
        if w.error:
            continue
        n = kept_by_sq[w.sq_id]
        status = "done" if n >= 1 else ("empty" if not w.picks else "weak")
        storage.update_sub_query(ctx.conn, w.sq_id, status)
    return RoundResult(sum(kept_by_sq.values()), kept_by_sq)
