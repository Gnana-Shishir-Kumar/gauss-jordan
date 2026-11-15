"""Orchestration: plan -> research (+ at most one follow-up round) -> write -> verify.

Plain async functions; no UI or web framework code. State is persisted after every step.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import storage
from .config import Settings
from .context import EventCallback, RunContext, describe_error
from .llm import LLMClient, TokenFactoryLLM
from .planner import plan, plan_follow_up
from .render import format_answer
from .researcher import research_round
from .search.base import SearchBackend
from .search.tavily import TavilyBackend
from .verifier import verify
from .writer import write


def default_clients(settings: Settings) -> tuple[LLMClient, SearchBackend]:
    missing = settings.missing_keys()
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (see .env.example)")
    return TokenFactoryLLM.from_settings(settings), TavilyBackend.from_settings(settings)


def load_report(conn: sqlite3.Connection, task_id: int) -> dict[str, Any]:
    trace = storage.get_trace(conn, task_id)
    for t in trace:
        try:
            d = json.loads(t["detail"]) if t["detail"] else {}
        except (TypeError, ValueError):
            d = {"message": t["detail"]}
        t["detail"] = d
        t["message"] = d.get("message", "") if isinstance(d, dict) else str(d)
        t["level"] = d.get("level", "info") if isinstance(d, dict) else "info"
    return {
        "task": storage.get_task(conn, task_id),
        "sub_queries": storage.get_sub_queries(conn, task_id),
        "evidence": storage.get_evidence(conn, task_id),
        "claims": storage.get_claims(conn, task_id),
        "trace": trace,
    }


async def _run(ctx: RunContext, question: str) -> tuple[str, str]:
    lim = ctx.settings.limits
    conn, tid = ctx.conn, ctx.task_id

    # 1. Plan
    sub_queries = await plan(ctx, question)

    # 2. Research
    storage.set_task_status(conn, tid, "researching")
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    results = await research_round(ctx, sub_queries, seen_urls, seen_hashes)

    # 3. Follow-up: at most max_follow_up_rounds, only for weak sub-queries of the previous round
    current = sub_queries
    for round_ in range(2, 2 + lim.max_follow_up_rounds):
        weak = [(i, t, results[i].weakness) for i, t in current
                if results[i].kept < lim.min_kept_evidence_per_sub_query]
        if not weak:
            break
        try:
            current = await plan_follow_up(ctx, question, weak, round_)
        except Exception as exc:
            await ctx.emit("follow_up", f"Follow-up planning failed: {describe_error(exc)}", level="error")
            break
        results.update(await research_round(ctx, current, seen_urls, seen_hashes))

    kept = storage.get_evidence(conn, tid, kept_only=True)
    rows = sorted(kept, key=lambda e: e["relevance_score"] or 0, reverse=True)[: lim.max_evidence_for_writer]
    if not rows:
        await ctx.emit("write", "Skipped writing: no usable evidence was found", level="warning")
        return "insufficient_evidence", format_answer(
            [], "Insufficient evidence: no usable sources were found for this question, so no answer is given."
        )

    # 4. Write
    storage.set_task_status(conn, tid, "writing")
    draft = await write(ctx, question, rows)
    if not storage.get_claims(conn, tid):
        note = draft.insufficiency_note or "The writer found the evidence insufficient to make any claim."
        return "insufficient_evidence", format_answer([], f"Insufficient evidence: {note}")

    # 5. Verify
    storage.set_task_status(conn, tid, "verifying")
    await verify(ctx, {r["id"] for r in rows})

    claims = storage.get_claims(conn, tid)
    any_supported = any(x["verdict"] in ("supported", "partial") for c in claims for x in c["citations"])
    if not draft.sufficient:
        note = "Insufficient evidence: " + (draft.insufficiency_note or "the sources only partly answer the question.")
        return "insufficient_evidence", format_answer(claims, note)
    if not any_supported:
        return "insufficient_evidence", format_answer(
            claims, "Insufficient evidence: none of the claims could be verified against their cited sources."
        )
    return "done", format_answer(claims)


async def run_research(
    question: str,
    *,
    llm: LLMClient,
    search: SearchBackend,
    conn: sqlite3.Connection,
    settings: Settings,
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    """Run the full pipeline. Never raises for API failures: the task ends as done,
    insufficient_evidence, or failed (with `error` set). Returns the full report from the DB."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    task_id = storage.create_task(conn, question)
    ctx = RunContext(conn, task_id, settings, llm, search, on_event)
    await ctx.emit("start", f"Task {task_id}: {question}")
    try:
        status, answer = await _run(ctx, question)
        storage.finish_task(conn, task_id, status, final_answer=answer)
        await ctx.emit("done", f"Finished: {status}", level="info" if status == "done" else "warning",
                       detail={"status": status})
    except Exception as exc:
        err = describe_error(exc, limit=1000)
        storage.finish_task(conn, task_id, "failed", error=err)
        await ctx.emit("error", f"Run failed: {err}", level="error", detail={"status": "failed"})
    return load_report(conn, task_id)
