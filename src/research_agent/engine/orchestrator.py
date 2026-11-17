"""Orchestrator: prepare (brief + plan, then wait for approval) and run (research loop, write, verify, repair).

Plain async functions; no UI or web framework code. State is persisted after every step.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date
from typing import Any

from .. import storage
from ..backends.llm import LLMClient, TokenFactoryLLM
from ..backends.search_base import SearchBackend
from ..backends.tavily import TavilyBackend
from ..config import Settings
from ..privacy import check_question
from ..prompts import PROMPT_VERSION
from ..backends.identifiers import IdentifierResolver
from ..render import SUPPORTING, format_answer, is_flagged
from ..retry import is_transient
from ..sources import is_peer_reviewed, is_preliminary
from .brief import make_brief
from .context import EventCallback, RunContext, describe_error
from .coverage import assess_coverage
from .planner import plan_queries
from .repair import repair_loop
from .research import ResearchState, research_round
from .verifier import verify_claims
from .writer import persist_draft, settle_bottom_line, verified_items, write_draft

STOP_REASONS = ("coverage_complete", "no_new_evidence", "quota_reached", "time_budget")


def default_clients(settings: Settings) -> tuple[LLMClient, SearchBackend]:
    missing = settings.missing_keys()
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (see .env.example)")
    return TokenFactoryLLM.from_settings(settings), TavilyBackend.from_settings(settings)


def load_report(conn: sqlite3.Connection, task_id: int) -> dict[str, Any]:
    return {
        "task": storage.get_task(conn, task_id),
        "sub_queries": storage.get_sub_queries(conn, task_id),
        "evidence": storage.get_evidence(conn, task_id),
        "notes": storage.get_notes(conn, task_id),
        "claims": storage.get_claims(conn, task_id),
        "trace": storage.get_trace(conn, task_id),
        "totals": storage.task_totals(conn, task_id),
        "steps": storage.step_summary(conn, task_id),
    }


def _ctx(conn, task_id, settings, llm, search, on_event, ids=None) -> RunContext:
    return RunContext(conn, task_id, settings, llm, search, on_event, ids=ids)


async def prepare(question: str, *, llm: LLMClient, search: SearchBackend, conn: sqlite3.Connection,
                  settings: Settings, on_event: EventCallback | None = None, today: date | None = None,
                  ids: Any = None) -> dict:
    """Brief + plan. Ends in awaiting_approval, needs_clarification or failed.

    Raises privacy.IdentifierError (before anything is stored) if the question looks like it has identifiers."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    check_question(question)
    task_id = storage.create_task(conn, question, settings.model_ids(), PROMPT_VERSION)
    ctx = _ctx(conn, task_id, settings, llm, search, on_event)
    await ctx.emit("start", f"Task {task_id}", detail={"models": settings.model_ids()})
    try:
        brief = await make_brief(ctx, question, today)
        if brief.get("needs_clarification"):
            await ctx.emit("brief", f"Clarifying question: {brief.get('clarifying_question')}", level="warning")
            return load_report(conn, task_id)
        queries = await plan_queries(ctx, brief)
        storage.update_task(conn, task_id, status="awaiting_approval")
        await ctx.emit("approval", "Waiting for approval of the brief and queries",
                       detail={"queries": [q for _, q in queries]})
    except Exception as exc:
        err = describe_error(exc, limit=1000)
        storage.finish_task(conn, task_id, "failed", error=err)
        await ctx.emit("error", f"Run failed: {err}", level="error")
    return load_report(conn, task_id)


async def run(task_id: int, *, llm: LLMClient, search: SearchBackend, conn: sqlite3.Connection, settings: Settings,
              on_event: EventCallback | None = None, ids: Any = None) -> dict:
    """Research, write, verify and repair an approved task. Never raises for API failures.

    ids: identifier resolver for verifier gate 2; defaults to the live PubMed/ClinicalTrials.gov/Crossref resolver."""
    task = storage.get_task(conn, task_id)
    if not task or task["status"] != "awaiting_approval":
        raise ValueError(f"task {task_id} is not awaiting approval")
    if ids is None:
        ids = IdentifierResolver.from_settings(conn, settings)
    ctx = _ctx(conn, task_id, settings, llm, search, on_event, ids)
    storage.update_task(conn, task_id, status="researching")
    await ctx.emit("approval", "Approved; research started")
    try:
        status, answer = await _run(ctx, task["brief"])
        storage.finish_task(conn, task_id, status, final_answer=answer, usage=ctx.quota.snapshot())
        await ctx.emit("done", f"Finished: {status}", level="info" if status == "done" else "warning",
                       detail={"status": status, "credits": round(ctx.credits_used, 2), "quota": ctx.quota.snapshot()})
    except Exception as exc:
        err = describe_error(exc, limit=1000)
        storage.finish_task(conn, task_id, "failed", error=err, usage=ctx.quota.snapshot())
        hint = f". Resume with: research --resume {task_id}" if can_resume(conn, task_id) else ""
        await ctx.emit("error", f"Run failed: {err}{hint}", level="error")
    return load_report(conn, task_id)


def can_resume(conn: sqlite3.Connection, task_id: int) -> bool:
    """A failed task whose research finished (stop reason recorded, notes stored) can be resumed from writing."""
    task = storage.get_task(conn, task_id)
    return bool(task and task["status"] == "failed" and task.get("stop_reason") and storage.get_notes(conn, task_id))


async def resume(task_id: int, *, llm: LLMClient, search: SearchBackend, conn: sqlite3.Connection,
                 settings: Settings, on_event: EventCallback | None = None, ids: Any = None) -> dict:
    """Re-run writing, verification and repair for a task that failed after research, reusing its stored evidence
    (no new research calls; repair still has its reserved Tavily quota). Earlier claims of the task are discarded."""
    if not can_resume(conn, task_id):
        raise ValueError(f"task {task_id} cannot be resumed (it must have failed after research finished)")
    if ids is None:
        ids = IdentifierResolver.from_settings(conn, settings)
    ctx = _ctx(conn, task_id, settings, llm, search, on_event, ids)
    task = storage.get_task(conn, task_id)
    storage.delete_claims(conn, task_id)
    storage.update_task(conn, task_id, status="writing", error=None, finished_at=None)
    await ctx.emit("resume", f"Resuming from stored research ({task.get('stop_reason')}); writing again")
    state = ResearchState()
    for e in storage.get_evidence(conn, task_id):
        state.seen_urls.add(e["url"])
        if e["kept"] and e.get("content_hash"):
            state.seen_hashes.add(e["content_hash"])
    try:
        status, answer = await _write_verify(ctx, task["brief"], state)
        storage.finish_task(conn, task_id, status, final_answer=answer, usage=ctx.quota.snapshot())
        await ctx.emit("done", f"Finished: {status}", level="info" if status == "done" else "warning",
                       detail={"status": status, "credits": round(ctx.credits_used, 2), "quota": ctx.quota.snapshot()})
    except Exception as exc:
        err = describe_error(exc, limit=1000)
        storage.finish_task(conn, task_id, "failed", error=err, usage=ctx.quota.snapshot())
        await ctx.emit("error", f"Run failed: {err}. Resume with: research --resume {task_id}", level="error")
    return load_report(conn, task_id)


async def _patient(ctx: RunContext, step: str, fn):
    """Extra attempts for a long LLM step after its client-level retries hit a transient error."""
    waits = list(ctx.settings.network.patient_waits_s)
    while True:
        try:
            return await fn()
        except Exception as exc:
            if not waits or not is_transient(exc):
                raise
            wait = waits.pop(0)
            await ctx.emit(step, f"{step} failed ({describe_error(exc)}); trying again in {wait:g}s", level="warning")
            await asyncio.sleep(wait)


async def run_all(question: str, **kw: Any) -> dict:
    """prepare + run without a pause (CLI --yes, evals)."""
    today = kw.pop("today", None)
    report = await prepare(question, today=today, **{k: v for k, v in kw.items() if k != "ids"})
    if report["task"]["status"] != "awaiting_approval":
        return report
    return await run(report["task"]["id"], **kw)


async def _research_loop(ctx: RunContext, brief: dict) -> tuple[str, ResearchState]:
    """Rounds until one of the four stop reasons:
      coverage_complete  every aspect has >= 2 independent sources;
      no_new_evidence    two consecutive rounds added no new sentence IDs (or there is nothing left to search:
                         no follow-up queries, or the coverage check failed; the trace says which);
      quota_reached      the research share of the Tavily quota is spent (or the round safety cap was reached);
      time_budget        the research time budget is spent."""
    lim = ctx.settings.limits
    planned = [(s["id"], s["text"]) for s in storage.get_sub_queries(ctx.conn, ctx.task_id) if s["round"] == 1]
    state = ResearchState(queries=[q for _, q in planned])
    queries, round_, depth, tiers = planned, 1, lim.round1_depth, lim.round1_tiers
    empty_rounds = 0
    why = ""
    while True:
        rr = await research_round(ctx, brief, queries, round_, depth, tiers, state)
        empty_rounds = empty_rounds + 1 if rr.new_sentences == 0 else 0
        await ctx.emit("round", f"Round {round_} added {rr.new_sentences} new sentence IDs from {rr.new_kept} sources",
                       detail={"round": round_, "new_sentences": rr.new_sentences, "new_sources": rr.new_kept,
                               "quota": ctx.quota.snapshot()})
        if (stop := ctx.budget_exhausted()):
            break
        if empty_rounds >= 2:
            stop, why = "no_new_evidence", "two consecutive rounds added no new sentence IDs"
            break
        cov = await assess_coverage(ctx, brief, state.queries)
        if cov.complete:
            stop = "coverage_complete"
        elif cov.failed:
            stop, why = "no_new_evidence", "the coverage check failed, so there are no follow-up queries"
        elif not cov.follow_ups:
            stop, why = "no_new_evidence", "no new follow-up queries for the thin aspects"
        elif round_ >= lim.max_research_rounds:
            stop, why = "quota_reached", f"round cap ({lim.max_research_rounds}) reached"
        elif (b := ctx.budget_exhausted()):
            stop = b
        if stop:
            break
        round_ += 1
        depth, tiers = lim.follow_up_depth, lim.follow_up_tiers
        queries = [(storage.add_sub_query(ctx.conn, ctx.task_id, q, round_, depth), q) for q in cov.follow_ups]
        state.queries += cov.follow_ups
    if not why:
        reserve = ctx.settings.budget.tavily_repair_reserve
        why = {"quota_reached": f"research share of the Tavily quota spent ({ctx.quota.used['tavily']} calls; "
                                f"{reserve} reserved for repair)",
               "time_budget": f"{ctx.settings.budget.time_s:g}s research time budget spent",
               "coverage_complete": "every aspect has independent sources"}.get(stop, "")
    storage.update_task(ctx.conn, ctx.task_id, stop_reason=stop)
    await ctx.emit("stop", f"Research stopped after round {round_}: {stop} ({why})",
                   detail={"stop_reason": stop, "why": why, "rounds": round_, "credits": round(ctx.credits_used, 2),
                           "quota": ctx.quota.snapshot()})
    return stop, state


async def _run(ctx: RunContext, brief: dict) -> tuple[str, str]:
    _, state = await _research_loop(ctx, brief)
    return await _write_verify(ctx, brief, state)


async def _write_verify(ctx: RunContext, brief: dict, state: ResearchState) -> tuple[str, str]:
    conn, tid = ctx.conn, ctx.task_id
    kept = storage.get_evidence(conn, tid, kept_only=True)
    notes = storage.get_notes(conn, tid)
    task = storage.get_task(conn, tid)
    if not kept or not notes:
        await ctx.emit("write", "Skipped writing: no usable evidence", level="warning")
        answer = {"insufficiency_note": "no usable sources were found on the allowlist since the anchor date."}
        storage.update_task(conn, tid, answer_json=answer)
        return "insufficient_evidence", format_answer(task, [], answer)

    storage.update_task(conn, tid, status="writing")
    draft = await _patient(ctx, "write", lambda: write_draft(ctx, brief))
    await persist_draft(ctx, draft)

    storage.update_task(conn, tid, status="verifying")
    await verify_claims(ctx, phase="initial")
    summary = await repair_loop(ctx, brief, state)
    await ctx.emit("repair", f"Repair finished after {summary['rounds']} round(s)", detail=summary)
    return await _finalize(ctx, brief, draft)


async def _finalize(ctx: RunContext, brief: dict, draft) -> tuple[str, str]:
    conn, tid = ctx.conn, ctx.task_id
    for c in storage.get_claims(conn, tid, include_deleted=False):
        supporting = [x for x in c["citations"] if x.get("verdict") in SUPPORTING]
        fields: dict[str, Any] = {"peer_reviewed": int(any(is_peer_reviewed(x.get("source_type")) for x in supporting))}
        if supporting and all(x.get("trust_tier") == 3 for x in supporting) and c["claim_type"] != "uncertain":
            fields["claim_type"] = "uncertain"
            await ctx.emit("check", f"Claim {c['position'] + 1}: supported only by tier 3 sources; marked uncertain",
                           level="warning", detail={"claim_id": c["id"]})
        storage.update_claim(conn, c["id"], **fields)

    claims = storage.get_claims(conn, tid, include_deleted=False)
    bl = await settle_bottom_line(ctx, brief, draft, claims)
    if bl:
        strong = any(x.get("verdict") == "supported" and x.get("trust_tier") in (1, 2)
                     and not is_preliminary(x.get("source_type")) for c in claims for x in c["citations"])
        caveat = any(is_flagged(c) for c in claims if c["citations"]) or any(c["section"] == "conflict" for c in claims)
        base = (bl.get("reason") or "").strip().rstrip(".")
        base = f"{base}. " if base else ""
        if not strong and bl["confidence"] != "low":
            bl.update(confidence="low",
                      reason=base + "Capped at low: no finding is supported by a tier 1-2, non-preliminary source.")
        elif caveat and bl["confidence"] == "high":
            bl.update(confidence="moderate",
                      reason=base + "Capped at moderate: some items are unverified or sources disagree.")
    verified = verified_items(claims, brief.get("anchor_date"))
    note = draft.insufficiency_note if not draft.sufficient else None
    if not verified and not note:
        note = "none of the developments could be verified against their cited sources."
    answer = {"bottom_line": bl, "not_established": draft.not_established,
              "check_before_acting": draft.check_before_acting, "sufficient": draft.sufficient,
              "insufficiency_note": note}
    storage.update_task(conn, tid, answer_json=answer)
    status = "done" if (draft.sufficient and verified) else "insufficient_evidence"
    return status, format_answer(storage.get_task(conn, tid), storage.get_claims(conn, tid), answer)
