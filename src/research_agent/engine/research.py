"""Step 4: parallel research workers.

Per round:
  1. each worker searches its query (cache first; tier-filtered domains as a strict filter; start_date = anchor),
  2. scores hits on snippets and picks its top URLs,
  3. the round's picks are fetched in shared Tavily Extract batches as FULL page text (cache first). failed_results
     is always read; failed URLs are retried once with advanced extraction, then recorded as source_unavailable,
  4. raw text is stored once per source and split into numbered sentences (E12-4 = source E12, sentence 4);
     evidence is deduped by URL and content hash; dropped items keep a reason,
  5. each worker (FAST) sees only its sources' most relevant sentences, with IDs, and returns compressed notes:
     a short finding plus the sentence IDs it rests on. Notes citing unknown IDs, or stating numbers that are not in
     their cited sentences, are dropped in code.
Raw page text never leaves this step: the planner, coverage check and writer only see notes and metadata.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .. import prompts, storage
from ..backends import cache
from ..backends.search_base import SearchHit
from ..evidence import content_hash, domain_of, normalize_url, relevance
from ..retry import with_retry
from ..schemas import WorkerNotes, parse_partial_date
from ..sentences import parse_ref, select, split_sentences
from ..sources import classify, domains_for, tier_of
from . import numbers
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
    new_sentences: int = 0


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _retry_kwargs(ctx: RunContext, step: str, label: str, sq_id: int | None = None) -> dict:
    net = ctx.settings.network

    def on_retry(exc: BaseException, attempt: int, delay: float) -> None:
        ctx.note(step, f"Retry {attempt} for {label} after {describe_error(exc)}", sub_query_id=sq_id,
                 delay_s=round(delay, 2))

    return dict(attempts=net.max_retries, base_delay=net.backoff_base_s, max_delay=net.backoff_max_s,
                on_retry=on_retry, retry_after_max=net.retry_after_max_s)


def _source(ctx: RunContext, url: str, hit: SearchHit, tier: int | None, chash: str | None = None) -> int:
    domain = domain_of(url)
    return storage.upsert_source(ctx.conn, url, domain, hit.title or None, hit.published_date, chash, tier,
                                 classify(url, domain))


def _record(ctx: RunContext, sq_id: int, url: str, hit: SearchHit, tier: int | None, excerpt: str, rel: float,
            kept: bool, reason: str | None, chash: str | None = None, shown: list[int] | None = None) -> int:
    src = _source(ctx, url, hit, tier, chash)
    return storage.add_evidence(ctx.conn, ctx.task_id, sq_id, src, excerpt, rel, kept, reason,
                                parse_partial_date((hit.published_date or "")[:10]), shown)


def _quota_reason(ctx: RunContext, phase: str) -> str | None:
    if phase == "research":
        return ctx.budget_exhausted()
    return None if ctx.quota.left("tavily", "repair") > 0 else "quota_reached"


async def _search(ctx: RunContext, brief: dict, sq_id: int, query: str, depth: str, tiers: tuple[int, ...],
                  sem: asyncio.Semaphore, phase: str = "research", max_results: int | None = None) -> WorkerHits:
    lim, net = ctx.settings.limits, ctx.settings.network
    n_results = max_results or lim.search_results
    out = WorkerHits(sq_id, query)
    async with sem:
        storage.update_sub_query(ctx.conn, sq_id, "searching")
        domains = domains_for(tiers)
        start = brief.get("anchor_date")
        key = cache.search_key(query, depth, domains, start, n_results)
        detail = {"sub_query_id": sq_id, "query": query, "depth": depth, "tiers": list(tiers), "start_date": start,
                  "phase": phase}
        t0 = time.perf_counter()
        hits = cache.get_search(ctx.conn, key, ctx.settings.cache_ttl_days)
        if hits is not None:
            await ctx.emit("search", f"Cache hit: {len(hits)} results for '{query}'", credits=0.0, latency_ms=_ms(t0),
                           detail={**detail, "cached": True,
                                   "results": [{"url": h.url, "score": h.score} for h in hits]})
        else:
            if (b := _quota_reason(ctx, phase)) or not ctx.take("tavily", phase):
                b = b or "quota_reached"
                storage.update_sub_query(ctx.conn, sq_id, "skipped_budget")
                await ctx.emit("search", f"Skipped '{query}': {b}", level="warning", detail=detail)
                out.error = b
                return out
            try:
                res = await with_retry(
                    lambda: ctx.search.search(query, depth=depth, max_results=n_results,
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
                           detail={**detail, "credits_estimated": res.estimated,
                                   "results": [{"url": h.url, "score": h.score} for h in hits]})

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


async def _extract_batch(ctx: RunContext, by_raw: dict[str, str], depth: str, phase: str
                         ) -> tuple[dict[str, str], dict[str, str]]:
    """One Tavily Extract call. Returns (pages, failed) keyed by normalized URL."""
    net = ctx.settings.network
    if (b := _quota_reason(ctx, phase)) or not ctx.take("tavily", phase):
        b = b or "quota_reached"
        await ctx.emit("extract", f"Skipped extracting {len(by_raw)} URLs ({depth}): {b}", level="warning")
        return {}, {u: f"skipped: {b}" for u in by_raw.values()}
    t0 = time.perf_counter()
    try:
        res = await with_retry(lambda: ctx.search.extract(list(by_raw), depth=depth),
                               timeout=net.extract_timeout_s + 5,
                               **_retry_kwargs(ctx, "extract", f"{depth} batch of {len(by_raw)}"))
    except Exception as exc:
        err = describe_error(exc)
        await ctx.emit("extract", f"Extraction batch failed ({len(by_raw)} URLs, {depth}): {err}", level="error",
                       latency_ms=_ms(t0))
        return {}, {u: err for u in by_raw.values()}
    got = {by_raw.get(u, normalize_url(u)): t for u, t in res.pages.items()}
    bad = {by_raw.get(u, normalize_url(u)): e for u, e in res.failed.items()}
    await ctx.emit("extract", f"Extracted {len(got)}/{len(by_raw)} URLs ({depth})",
                   level="info" if not bad else "warning", credits=res.credits, latency_ms=_ms(t0),
                   detail={"urls": len(by_raw), "depth": depth, "failed": bad, "credits_estimated": res.estimated,
                           "phase": phase})
    return got, bad


async def fetch_pages(ctx: RunContext, picks: list[Pick], phase: str = "research"
                      ) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Full text per normalized URL: {url: (text, fetched_via)}, plus {url: error} for coverage gaps.

    Cache first (raw text already stored within the TTL). Then batched basic extraction; failed_results are retried
    once with advanced extraction; whatever still fails is a coverage gap (source_unavailable)."""
    lim = ctx.settings.limits
    urls = [p.url for p in picks]
    pages = {u: (t, "cache") for u, t in storage.cached_texts(ctx.conn, urls, ctx.settings.cache_ttl_days).items()}
    if pages:
        await ctx.emit("extract", f"Cache hit: {len(pages)} pages reused", credits=0.0,
                       detail={"cached_urls": sorted(pages)})
    failed: dict[str, str] = {}
    todo = [p for p in picks if p.url not in pages]
    for i in range(0, len(todo), lim.extract_batch_size):
        batch = todo[i: i + lim.extract_batch_size]
        got, bad = await _extract_batch(ctx, {p.hit.url: p.url for p in batch}, "basic", phase)
        pages.update({u: (t, "tavily_extract_basic") for u, t in got.items()})
        failed.update(bad)
    retry = [p for p in todo if p.url in failed and not failed[p.url].startswith("skipped")]
    if retry:
        got, bad = await _extract_batch(ctx, {p.hit.url: p.url for p in retry}, "advanced", phase)
        for u, t in got.items():
            pages[u] = (t, "tavily_extract_advanced")
            failed.pop(u, None)
        for u, e in bad.items():
            failed[u] = f"{failed.get(u, '')}; advanced: {e}".strip("; ")
    # Next fallback for articles (PubMed Central full text) plugs in here; until then the failure is a coverage gap.
    return pages, failed


def _store_text(ctx: RunContext, src_id: int, text: str, via: str) -> list[str]:
    sentences = split_sentences(text)
    if via != "cache" or not storage.get_sentences(ctx.conn, src_id, [1]):
        storage.set_source_text(ctx.conn, src_id, text, sentences, via)
    return sentences


def store_picks(ctx: RunContext, brief: dict, query: str, picks: list[Pick], pages: dict[str, tuple[str, str]],
                failed: dict[str, str], state: ResearchState) -> tuple[list[int], Counter]:
    """Record each pick as evidence: kept (with its shown sentences) or dropped with a reason."""
    lim = ctx.settings.limits
    kept: list[int] = []
    dropped: Counter[str] = Counter()
    q = f"{query} {brief.get('question', '')}"
    for p in picks:
        got = pages.get(p.url)
        if not got:
            # Login walls and failed extractions: recorded as source_unavailable, never used for claims.
            reason = f"source_unavailable: {failed.get(p.url, 'not extracted')}"
            _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False, reason)
            state.seen_urls.add(p.url)
            dropped["source_unavailable"] += 1
            continue
        text, via = got
        chash = content_hash(text)
        src_id = _source(ctx, p.url, p.hit, p.tier, chash)
        sentences = _store_text(ctx, src_id, text, via)
        shown = select(sentences, q, lim.sentences_per_source, lim.sentence_chars_per_source)
        excerpt = "\n".join(sentences[i - 1] for i in shown)
        score = relevance(p.hit.score, q, excerpt)
        reason = None
        if not sentences:
            reason = "empty_content"
        elif p.url in state.seen_urls:
            reason = "duplicate_url"
        elif chash in state.seen_hashes:
            reason = "duplicate_content"
        elif score < lim.min_relevance:
            reason = "low_relevance"
        if reason is None:
            state.seen_urls.add(p.url)
            state.seen_hashes.add(chash)
        eid = storage.add_evidence(ctx.conn, ctx.task_id, p.sq_id, src_id, excerpt, score, reason is None, reason,
                                   parse_partial_date((p.hit.published_date or "")[:10]), shown)
        if reason is None:
            kept.append(eid)
        else:
            dropped[reason] += 1
    return kept, dropped


async def take_notes(ctx: RunContext, brief: dict, sq_id: int | None, query: str, rows: list[dict],
                     sem: asyncio.Semaphore) -> int:
    if not rows:
        return 0
    model = ctx.settings.models.fast
    sents_by_eid = {r["id"]: storage.get_sentences(ctx.conn, r["source_id"]) for r in rows}
    blocks = [prompts.source_block(r, {i: sents_by_eid[r["id"]][i] for i in r["shown"] if i in sents_by_eid[r["id"]]})
              for r in rows]
    try:
        async with sem:
            res = await ctx.llm.complete_json(model=model, system=prompts.NOTES_SYSTEM,
                                              user=prompts.notes_user(brief, query, blocks), schema=WorkerNotes)
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
        refs, texts = [], []
        for raw in note.sentence_ids:
            ref = parse_ref(raw)
            if ref is None or ref.evidence_id not in by_id or ref.idx not in sents_by_eid[ref.evidence_id]:
                dropped["unknown_sentence_id"] += 1
                continue
            if str(ref) not in [str(x) for x in refs]:
                refs.append(ref)
                texts.append(sents_by_eid[ref.evidence_id][ref.idx])
        if not refs:
            dropped["no_valid_citation"] += 1
            continue
        num = numbers.check(note.text, texts)
        if not num.passed:
            dropped["unverified_number"] += 1
            continue
        storage.add_note(ctx.conn, ctx.task_id, sq_id, note.text, sorted({r.evidence_id for r in refs}), {},
                         [str(r) for r in refs])
        kept += 1
    await ctx.emit_llm("notes", f"{kept} notes for '{query}'", res,
                       detail={"sub_query_id": sq_id, "dropped": dict(dropped)})
    return kept


def _select(ctx: RunContext, workers: list[WorkerHits], state: ResearchState, per_query: int, per_round: int
            ) -> list[Pick]:
    """Each worker's top URLs by snippet relevance, then a per-round cap. Unselected hits are recorded with a reason."""
    lim = ctx.settings.limits
    chosen: list[Pick] = []
    for w in workers:
        n = 0
        for p in sorted(w.picks, key=lambda p: p.rel, reverse=True):
            if p.url in state.seen_urls:
                _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False, "duplicate_url")
            elif n < per_query and p.rel >= lim.min_relevance:
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
        elif len(final) >= per_round:
            _record(ctx, p.sq_id, p.url, p.hit, p.tier, p.hit.content or "", p.rel, False,
                    "not_selected_for_extraction")
        else:
            final.append(p)
            in_round.add(p.url)
    return final


async def research_round(ctx: RunContext, brief: dict, queries: list[tuple[int, str]], round_: int, depth: str,
                         tiers: tuple[int, ...], state: ResearchState) -> RoundResult:
    lim = ctx.settings.limits
    before = storage.kept_sentence_total(ctx.conn, ctx.task_id)
    sem = asyncio.Semaphore(lim.max_concurrency)
    await ctx.emit("round", f"Round {round_}: {len(queries)} queries, {depth} depth, tiers {list(tiers)}",
                   detail={"round": round_, "queries": [q for _, q in queries]})
    workers = await asyncio.gather(*(_search(ctx, brief, i, q, depth, tiers, sem) for i, q in queries))

    # Off-allowlist results (defensive; include_domains should prevent them) are recorded, not used.
    for w in workers:
        for url, h in w.off_allowlist:
            _record(ctx, w.sq_id, url, h, None, h.content or "", 0.0, False, "off_allowlist")

    final = _select(ctx, workers, state, lim.extract_per_query, lim.extract_per_round)
    pages, failed = await fetch_pages(ctx, final)

    by_worker: dict[int, list[Pick]] = defaultdict(list)
    for p in final:
        by_worker[p.sq_id].append(p)
    kept_rows: dict[int, list[int]] = {}
    for w in workers:  # plan order, so dedupe is deterministic
        picks = by_worker.get(w.sq_id, [])
        kept, dropped = store_picks(ctx, brief, w.query, picks, pages, failed, state)
        kept_rows[w.sq_id] = kept
        if picks:
            await ctx.emit("evidence", f"Kept {len(kept)}/{len(picks)} for '{w.query}'",
                           detail={"sub_query_id": w.sq_id, "dropped": dict(dropped)})

    note_sem = asyncio.Semaphore(lim.max_concurrency)
    await asyncio.gather(*(
        take_notes(ctx, brief, w.sq_id, w.query, storage.get_evidence(ctx.conn, ctx.task_id, ids=kept_rows[w.sq_id]),
                   note_sem)
        for w in workers if kept_rows.get(w.sq_id)
    ))

    kept_by_sq = {w.sq_id: len(kept_rows.get(w.sq_id, [])) for w in workers}
    for w in workers:
        if w.error:
            continue
        n = kept_by_sq[w.sq_id]
        storage.update_sub_query(ctx.conn, w.sq_id, "done" if n >= 1 else ("empty" if not w.picks else "weak"))
    new_sentences = storage.kept_sentence_total(ctx.conn, ctx.task_id) - before
    return RoundResult(sum(kept_by_sq.values()), kept_by_sq, new_sentences)


async def targeted_search(ctx: RunContext, brief: dict, query: str, state: ResearchState, round_: int) -> list[int]:
    """Repair: ONE targeted search from the reserved quota, its top URLs fetched and noted. Returns new evidence ids."""
    lim = ctx.settings.limits
    sq_id = storage.add_sub_query(ctx.conn, ctx.task_id, query, round_, lim.follow_up_depth, "planned")
    w = await _search(ctx, brief, sq_id, query, lim.follow_up_depth, lim.follow_up_tiers, asyncio.Semaphore(1),
                      phase="repair", max_results=lim.repair_search_results)
    if w.error or not w.picks:
        return []
    final = _select(ctx, [w], state, lim.repair_extract_per_search, lim.repair_extract_per_search)
    pages, failed = await fetch_pages(ctx, final, phase="repair")
    kept, dropped = store_picks(ctx, brief, query, final, pages, failed, state)
    await ctx.emit("evidence", f"Repair search kept {len(kept)}/{len(final)} for '{query}'",
                   detail={"sub_query_id": sq_id, "dropped": dict(dropped), "phase": "repair"})
    if kept:
        await take_notes(ctx, brief, sq_id, query, storage.get_evidence(ctx.conn, ctx.task_id, ids=kept),
                         asyncio.Semaphore(1))
    storage.update_sub_query(ctx.conn, sq_id, "done" if kept else "weak")
    return kept
