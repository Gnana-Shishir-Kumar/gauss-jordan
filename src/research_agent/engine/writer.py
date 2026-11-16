"""Step 6 (STRONG): notes -> dated update brief; step 8: repair rewrite for failed claims.

The writer sees notes and trusted evidence metadata only, never raw page text. Every item it returns goes
through the same deterministic guards (valid citations, design words, population) before it is stored.
"""

from __future__ import annotations

from .. import prompts, storage
from ..render import claim_verdict
from ..schemas import Draft, Item, Repair
from .checks import check_design, check_population, contains
from .context import RunContext, describe_error


def _index_rows(ctx: RunContext, notes: list[dict]) -> list[dict]:
    ids = sorted({i for n in notes for i in n["evidence_ids"]})
    return storage.get_evidence(ctx.conn, ctx.task_id, ids=ids)


async def write_draft(ctx: RunContext, brief: dict) -> Draft:
    notes = storage.get_notes(ctx.conn, ctx.task_id)[: ctx.settings.limits.max_notes_for_writer]
    res = await ctx.llm.complete_json(model=ctx.settings.models.strong, system=prompts.WRITER_SYSTEM,
                                      user=prompts.writer_user(brief, notes, _index_rows(ctx, notes)), schema=Draft)
    d = res.data
    await ctx.emit_llm("write", f"Drafted {len(d.items)} items, {len(d.conflicts)} conflicts (sufficient={d.sufficient})",
                       res, level="info" if d.sufficient else "warning",
                       detail={"insufficiency_note": d.insufficiency_note})
    return d


async def apply_item(ctx: RunContext, item: Item, notes: list[dict], ev_by_id: dict[int, dict], *,
                     position: int = 0, section: str = "new", claim_id: int | None = None) -> int:
    """Validate and store one item (new claim, or an in-place revision when claim_id is given)."""
    label = f"claim {claim_id}" if claim_id else f"{section} item {position + 1}"
    note_map = {n["id"]: n["evidence_ids"] for n in notes}
    shown = {i for ids in note_map.values() for i in ids}  # evidence behind the notes the writer saw
    remapped: dict[str, list[int]] = {}

    def resolve(i: int) -> list[int]:
        """Map a citation to evidence ids. Note references ("N7", or a bare number that is a note id but not
        evidence the writer saw) become that note's evidence; the mapping comes from our DB, not the model."""
        if i < 0 or (i not in shown and i in note_map):
            ids = note_map.get(abs(i), [])
            remapped[f"N{abs(i)}"] = ids
            return ids
        return [i]

    cited = list(dict.fromkeys(e for i in [*item.evidence_ids, *(n.evidence_id for n in item.numbers)]
                               for e in resolve(i)))
    if remapped:
        await ctx.emit("write", f"{label} cited notes instead of evidence; mapped "
                       + ", ".join(f"{k} -> {', '.join(f'E{e}' for e in v) or 'nothing'}" for k, v in remapped.items()),
                       level="warning", detail={"remapped": remapped})
    valid = [i for i in cited if i in ev_by_id]
    unknown = [i for i in cited if i not in ev_by_id]
    ctype = item.claim_type
    if unknown:
        await ctx.emit("write", f"{label} cited nonexistent evidence {unknown}; ignored", level="warning")
    if ctype != "uncertain" and not valid:
        ctype = "uncertain"
        await ctx.emit("write", f"{label} has no valid citation; marked uncertain", level="warning")
    excerpts = [ev_by_id[i]["excerpt"] for i in valid if ev_by_id[i]["kept"]]
    level = check_design(item.evidence_level, excerpts)
    if level != item.evidence_level:
        await ctx.emit("check", f"{label}: evidence level '{item.evidence_level}' not stated in sources; "
                       "set to 'not stated'", level="warning")
    population = check_population(item.population, notes, set(valid))
    dates = sorted(d for d in (ev_by_id[i].get("evidence_date") for i in valid) if d)
    item_date = item.date or (dates[-1] if dates else None)
    fields = dict(text=item.text, claim_type=ctype, evidence_level=level, population=population, item_date=item_date)
    if claim_id is None:
        claim_id = storage.add_claim(ctx.conn, ctx.task_id, position=position, section=section, **fields)
    else:
        storage.update_claim(ctx.conn, claim_id, revised=1, **fields)
        storage.delete_citations(ctx.conn, claim_id)
    # Attach each number's source sentence to the cited evidence whose excerpt contains it.
    sentences: dict[int, list[str]] = {}
    for n in item.numbers:
        targets = [e for e in resolve(n.evidence_id) if e in valid]
        holders = [e for e in targets if contains(ev_by_id[e]["excerpt"], n.sentence)] or targets[:1]
        for e in holders:
            sentences.setdefault(e, [])
            if n.sentence.strip() not in sentences[e]:
                sentences[e].append(n.sentence.strip())
    for eid in valid:
        storage.add_citation(ctx.conn, claim_id, eid, "\n".join(sentences.get(eid, [])) or None)
    return claim_id


async def persist_draft(ctx: RunContext, draft: Draft) -> list[int]:
    notes = storage.get_notes(ctx.conn, ctx.task_id)
    ev_by_id = {e["id"]: e for e in storage.get_evidence(ctx.conn, ctx.task_id)}
    ids, pos = [], 0
    for section, items in (("new", draft.items), ("conflict", draft.conflicts)):
        for item in items:
            ids.append(await apply_item(ctx, item, notes, ev_by_id, position=pos, section=section))
            pos += 1
    return ids


def failing_claims(ctx: RunContext) -> list[dict]:
    claims = storage.get_claims(ctx.conn, ctx.task_id)
    return [c for c in claims if c["citations"] and claim_verdict(c) in ("partial", "unsupported", "source_unavailable")]


async def repair_claims(ctx: RunContext, brief: dict) -> list[int]:
    """One repair pass: failed claims go back to the writer with the verifier's notes."""
    failed = failing_claims(ctx)
    if not failed:
        return []
    notes = storage.get_notes(ctx.conn, ctx.task_id)[: ctx.settings.limits.max_notes_for_writer]
    model = ctx.settings.models.strong
    try:
        res = await ctx.llm.complete_json(model=model, system=prompts.REPAIR_SYSTEM,
                                          user=prompts.repair_user(brief, failed, notes, _index_rows(ctx, notes)),
                                          schema=Repair)
    except Exception as exc:
        await ctx.emit("repair", f"Repair failed; {len(failed)} claims stay flagged: {describe_error(exc)}",
                       level="error", model_id=model)
        return []
    ev_by_id = {e["id"]: e for e in storage.get_evidence(ctx.conn, ctx.task_id)}
    allowed = {c["id"]: c for c in failed}
    revised: list[int] = []
    for rc in res.data.claims:
        if rc.claim_id in allowed and rc.claim_id not in revised:
            c = allowed[rc.claim_id]
            await apply_item(ctx, rc, notes, ev_by_id, position=c["position"], section=c["section"],
                             claim_id=rc.claim_id)
            revised.append(rc.claim_id)
    await ctx.emit_llm("repair", f"Revised {len(revised)}/{len(failed)} failed claims", res,
                       detail={"failed": sorted(allowed), "revised": revised})
    return revised
