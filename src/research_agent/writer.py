"""Write step: kept evidence -> typed claims with evidence citations, persisted."""

from __future__ import annotations

from . import prompts, storage
from .context import RunContext
from .schemas import Draft


async def write(ctx: RunContext, question: str, rows: list[dict]) -> Draft:
    res = await ctx.llm.complete_json(
        model=ctx.settings.models.writer,
        system=prompts.WRITER_SYSTEM,
        user=prompts.writer_user(question, rows),
        schema=Draft,
    )
    draft = res.data
    await ctx.emit(
        "write", f"Drafted {len(draft.claims)} claims (sufficient={draft.sufficient})",
        level="info" if draft.sufficient else "warning",
        detail={"model": ctx.settings.models.writer, "evidence_shown": [r["id"] for r in rows],
                "insufficiency_note": draft.insufficiency_note},
        latency_ms=res.latency_ms, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
    )

    known = {e["id"] for e in storage.get_evidence(ctx.conn, ctx.task_id)}
    for pos, c in enumerate(draft.claims):
        valid = [i for i in c.evidence_ids if i in known]
        unknown = [i for i in c.evidence_ids if i not in known]
        ctype = c.claim_type
        if unknown:
            await ctx.emit("write", f"Claim {pos + 1} cited nonexistent evidence {unknown}; ignored",
                           level="warning", detail={"position": pos, "unknown_ids": unknown})
        if ctype != "uncertain" and not valid:
            ctype = "uncertain"
            await ctx.emit("write", f"Claim {pos + 1} has no valid citation; downgraded to uncertain",
                           level="warning", detail={"position": pos})
        claim_id = storage.add_claim(ctx.conn, ctx.task_id, c.text, pos, ctype)
        for eid in valid:
            storage.add_citation(ctx.conn, claim_id, eid)
    return draft
