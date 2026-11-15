"""Step 7 (FAST): verify each claim-citation pair against the ORIGINAL evidence excerpt.

The deterministic number check runs first; a claim that fails it is marked unsupported without an LLM call
and goes to the repair loop.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from .. import prompts, storage
from ..schemas import ClaimVerification
from .checks import check_numbers, numbers_in
from .context import RunContext, describe_error


async def _verify_claim(ctx: RunContext, claim: dict, phase: str, sem: asyncio.Semaphore) -> None:
    label = f"Claim {claim['position'] + 1}"
    cits = claim["citations"]
    if numbers_in(claim["text"]):
        ok, problems = check_numbers(claim["text"], [c for c in cits if c.get("kept")])
        await ctx.emit("check", f"{label}: numeric check {'passed' if ok else 'failed'}",
                       level="info" if ok else "warning",
                       detail={"claim_id": claim["id"], "numeric": True, "passed": ok, "phase": phase,
                               "problems": problems})
        if not ok:
            for c in cits:
                storage.set_verdict(ctx.conn, claim["id"], c["evidence_id"], "unsupported",
                                    "numeric check failed: " + "; ".join(problems))
            return

    available = []
    for c in cits:
        if c.get("excerpt") is None:
            note = "evidence not found"
        elif not c.get("kept"):
            note = f"source unavailable ({c.get('drop_reason')})"
        elif not c["excerpt"].strip():
            note = "excerpt is empty"
        else:
            available.append(c | {"id": c["evidence_id"]})
            continue
        storage.set_verdict(ctx.conn, claim["id"], c["evidence_id"], "source_unavailable", note)
    if not available:
        await ctx.emit("verify", f"{label}: no available sources to check", level="warning",
                       detail={"claim_id": claim["id"], "phase": phase})
        return

    model = ctx.settings.models.fast
    try:
        async with sem:
            res = await ctx.llm.complete_json(model=model, system=prompts.VERIFIER_SYSTEM,
                                              user=prompts.verifier_user(claim, available), schema=ClaimVerification)
    except Exception as exc:
        err = describe_error(exc)
        for c in available:
            storage.set_verdict(ctx.conn, claim["id"], c["evidence_id"], "unsupported",
                                f"verification failed ({err}); treated as unsupported")
        await ctx.emit("verify", f"{label}: verification failed: {err}", level="error",
                       detail={"claim_id": claim["id"], "phase": phase}, model_id=model)
        return

    got = {v.evidence_id: v for v in res.data.verdicts}
    counts: Counter[str] = Counter()
    for c in available:
        v = got.get(c["evidence_id"])
        verdict, note = (v.verdict, v.note) if v else ("unsupported", "verifier returned no verdict; treated as unsupported")
        storage.set_verdict(ctx.conn, claim["id"], c["evidence_id"], verdict, note)
        counts[verdict] += 1
    await ctx.emit_llm("verify", f"{label}: " + ", ".join(f"{n} {k}" for k, n in counts.items()), res,
                       level="info" if counts.get("supported") else "warning",
                       detail={"claim_id": claim["id"], "phase": phase, "verdicts": dict(counts)})


async def verify_claims(ctx: RunContext, claim_ids: list[int] | None = None, phase: str = "initial") -> None:
    claims = storage.get_claims(ctx.conn, ctx.task_id, ids=claim_ids)
    sem = asyncio.Semaphore(ctx.settings.limits.max_concurrency)
    await asyncio.gather(*(_verify_claim(ctx, c, phase, sem) for c in claims if c["citations"]))
