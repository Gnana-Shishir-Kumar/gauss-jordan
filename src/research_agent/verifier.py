"""Verify step: check each claim against each cited excerpt; record a verdict per citation."""

from __future__ import annotations

import asyncio
from collections import Counter

from . import prompts, storage
from .context import RunContext, describe_error
from .schemas import ClaimVerification


async def _verify_claim(
    ctx: RunContext, claim: dict, ev_by_id: dict[int, dict], shown: set[int], sem: asyncio.Semaphore
) -> None:
    available: list[dict] = []
    for cit in claim["citations"]:
        eid = cit["evidence_id"]
        e = ev_by_id.get(eid)
        if e is None:
            note = "evidence not found"
        elif not e["kept"]:
            note = f"evidence was dropped ({e['drop_reason']}) and not shown to the writer"
        elif eid not in shown:
            note = "evidence was not among the excerpts given to the writer"
        elif not (e["excerpt"] or "").strip():
            note = "excerpt is empty"
        else:
            available.append(e)
            continue
        storage.set_verdict(ctx.conn, claim["id"], eid, "source_unavailable", note)
    if not available:
        await ctx.emit("verify", f"Claim {claim['position'] + 1}: no available sources to check",
                       level="warning", detail={"claim_id": claim["id"]})
        return

    try:
        async with sem:
            res = await ctx.llm.complete_json(
                model=ctx.settings.models.verifier,
                system=prompts.VERIFIER_SYSTEM,
                user=prompts.verifier_user(claim["text"], available),
                schema=ClaimVerification,
            )
    except Exception as exc:
        err = describe_error(exc)
        for e in available:
            storage.set_verdict(ctx.conn, claim["id"], e["id"], "unsupported",
                                f"verification failed ({err}); treated as unsupported")
        await ctx.emit("verify", f"Claim {claim['position'] + 1}: verification failed: {err}",
                       level="error", detail={"claim_id": claim["id"], "error": err})
        return

    got = {v.evidence_id: v for v in res.data.verdicts}
    counts: Counter[str] = Counter()
    for e in available:
        v = got.get(e["id"])
        if v is None:
            verdict, note = "unsupported", "verifier returned no verdict for this excerpt; treated as unsupported"
        else:
            verdict, note = v.verdict, v.note
        storage.set_verdict(ctx.conn, claim["id"], e["id"], verdict, note)
        counts[verdict] += 1
    await ctx.emit(
        "verify", f"Claim {claim['position'] + 1}: " + ", ".join(f"{n} {k}" for k, n in counts.items()),
        level="info" if counts.get("supported") else "warning",
        detail={"claim_id": claim["id"], "verdicts": dict(counts)},
        latency_ms=res.latency_ms, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
    )


async def verify(ctx: RunContext, shown_ids: set[int]) -> None:
    claims = storage.get_claims(ctx.conn, ctx.task_id)
    ev_by_id = {e["id"]: e for e in storage.get_evidence(ctx.conn, ctx.task_id)}
    sem = asyncio.Semaphore(ctx.settings.limits.max_concurrency)
    await asyncio.gather(*(
        _verify_claim(ctx, c, ev_by_id, shown_ids, sem) for c in claims if c["citations"]
    ))
