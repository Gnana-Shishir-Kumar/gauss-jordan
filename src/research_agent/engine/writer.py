"""Step 6 (STRONG): notes -> dated update brief; step 8: repair rewrite for failed claims.

The writer sees notes and trusted evidence metadata only, never raw page text. Every item it returns goes
through the same deterministic guards (valid citations, design words, population) before it is stored.
"""

from __future__ import annotations

from .. import prompts, storage
from ..render import SUPPORTING, claim_verdict
from ..schemas import BottomLine, Draft, Item, Repair
from .checks import before_anchor, check_design, check_item_date, check_population, numbers_in
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
    cited = list(dict.fromkeys([*item.evidence_ids, *(n.evidence_id for n in item.numbers)]))
    valid = [i for i in cited if i in ev_by_id]
    unknown = [i for i in cited if i not in ev_by_id]
    label = f"claim {claim_id}" if claim_id else f"{section} item {position + 1}"
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
    source_dates = [ev_by_id[i].get("evidence_date") for i in valid]
    source_dates += [(n.get("details") or {}).get("date") for n in notes if set(valid) & set(n["evidence_ids"])]
    item_date = check_item_date(item.date, source_dates)
    if item.date and item_date != item.date:
        await ctx.emit("check", f"{label}: date {item.date} not found in cited sources; "
                       f"using {item_date or 'date not stated'}", level="warning")
    anchor = (storage.get_task(ctx.conn, ctx.task_id) or {}).get("anchor_date")
    if before_anchor(item_date, anchor) and ctype != "uncertain":
        ctype = "uncertain"
        await ctx.emit("check", f"{label}: dated {item_date}, before the anchor date {anchor}; marked uncertain",
                       level="warning")
    fields = dict(text=item.text, claim_type=ctype, evidence_level=level, population=population, item_date=item_date)
    if claim_id is None:
        claim_id = storage.add_claim(ctx.conn, ctx.task_id, position=position, section=section, **fields)
    else:
        storage.update_claim(ctx.conn, claim_id, revised=1, **fields)
        storage.delete_citations(ctx.conn, claim_id)
    for eid in valid:
        sentences = list(dict.fromkeys(n.sentence.strip() for n in item.numbers if n.evidence_id == eid))
        storage.add_citation(ctx.conn, claim_id, eid, "\n".join(sentences) or None)
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


def verified_items(claims: list[dict], anchor: str | None) -> list[dict]:
    """'What's new' items whose every checked citation supports them and that are dated on/after the anchor."""
    return [c for c in claims if c["section"] == "new" and claim_verdict(c) in SUPPORTING
            and not before_anchor(c.get("item_date"), anchor)]


def _allowed_numbers(items: list[dict]) -> set[str]:
    out: set[str] = set()
    for c in items:
        out |= numbers_in(c["text"])
        for x in c["citations"]:
            if x.get("verdict") in SUPPORTING:
                out |= numbers_in(x.get("source_sentence") or "")
    return out


async def settle_bottom_line(ctx: RunContext, brief: dict, draft: Draft, claims: list[dict]) -> dict | None:
    """The bottom line may only state what the verified items state.

    Kept as drafted when every 'new' item verified unchanged and its numbers all come from verified items.
    Otherwise it is rewritten from the verified items only (numbers checked again in code); if that fails, a plain
    bottom line naming the newest verified item is used. No verified items -> no bottom line."""
    anchor = brief.get("anchor_date")
    items = verified_items(claims, anchor)
    if not items:
        return None
    allowed = _allowed_numbers(items)
    new = [c for c in claims if c["section"] == "new"]
    unchanged = len(items) == len(new) and not any(c.get("revised") for c in new)
    bl = draft.bottom_line
    if bl and unchanged and numbers_in(bl.text) <= allowed:
        return bl.model_dump()

    why = "some items were revised, flagged or dated before the anchor" if bl and not unchanged else \
        "it states numbers not found in verified items" if bl else "the draft had none"
    model = ctx.settings.models.strong
    try:
        res = await ctx.llm.complete_json(model=model, system=prompts.BOTTOM_LINE_SYSTEM,
                                          user=prompts.bottom_line_user(brief, items), schema=BottomLine)
        extra = sorted(numbers_in(res.data.text) - allowed)
        if not extra:
            await ctx.emit_llm("write", f"Bottom line rewritten from {len(items)} verified items ({why})", res)
            return res.data.model_dump()
        await ctx.emit_llm("check", f"Rewritten bottom line has unverified number(s) {', '.join(extra)}; "
                           "using a plain bottom line", res, level="warning")
    except Exception as exc:
        await ctx.emit("write", f"Bottom line rewrite failed: {describe_error(exc)}; using a plain bottom line",
                       level="error", model_id=model)
    newest = max(items, key=lambda c: c.get("item_date") or "")
    return {"text": f"{len(items)} verified development(s) since {anchor}; most recent: {newest['text']}",
            "confidence": "low", "reason": "Built from the verified items only."}
