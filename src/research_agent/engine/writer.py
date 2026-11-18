"""Step 6 (STRONG): notes -> dated update brief.

The writer sees notes (short findings with the sentence IDs they rest on) and trusted evidence metadata only, never
raw page text. Every item it returns is checked in code before it is stored:
  - each citation must be a sentence ID (E12-4) that exists in this task's kept evidence; a note reference (N7) is
    mapped to that note's sentence IDs from our DB;
  - any URL in the claim text must be a URL of this task's evidence;
  - a claim with a citation or URL that fails either rule, or with no valid citation, is stored as "rejected" and
    goes straight to the repair loop. This closes fabricated references in code, not by prompt;
  - evidence level and population must be stated in the cited sentences; the date must match a cited source or a
    date in the cited sentences; items dated before the anchor are not "new".
"""

from __future__ import annotations

import re

from .. import prompts, storage
from ..render import SUPPORTING, claim_verdict
from ..schemas import BottomLine, Draft, Item
from ..sentences import parse_ref
from . import numbers
from .checks import before_anchor, check_design, check_item_date, contains, population_supported
from .context import RunContext, describe_error

POPULATION_NOT_STATED = "population not stated"
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.I)
_NOTE_RE = re.compile(r"^N(\d+)$", re.I)


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


def resolve_citations(ctx: RunContext, raw_ids: list[str], text: str, notes: list[dict]
                      ) -> tuple[dict[int, list[int]], list[dict], dict[str, list[str]]]:
    """Mechanical citation check. Returns ({evidence_id: [sentence idx]}, rejected, remapped note refs)."""
    note_map = {n["id"]: n.get("sentence_ids") or [] for n in notes}
    kept = {e["id"]: e for e in storage.get_evidence(ctx.conn, ctx.task_id, kept_only=True)}
    sent_cache: dict[int, set[int]] = {}
    valid: dict[int, list[int]] = {}
    rejected: list[dict] = []
    remapped: dict[str, list[str]] = {}

    def exists(eid: int, idx: int) -> bool:
        if eid not in kept:
            return False
        if eid not in sent_cache:
            sent_cache[eid] = set(storage.get_sentences(ctx.conn, kept[eid]["source_id"]))
        return idx in sent_cache[eid]

    queue = list(raw_ids)
    while queue:
        raw = queue.pop(0)
        if (m := _NOTE_RE.match(raw)) and int(m.group(1)) in note_map:
            remapped[raw.upper()] = note_map[int(m.group(1))]
            queue += note_map[int(m.group(1))]
            continue
        ref = parse_ref(raw)
        if ref is None:
            rejected.append({"id": raw, "reason": "not a sentence ID"})
        elif not exists(ref.evidence_id, ref.idx):
            rejected.append({"id": raw, "reason": "sentence ID not in the evidence store"})
        elif ref.idx not in valid.setdefault(ref.evidence_id, []):
            valid[ref.evidence_id].append(ref.idx)
    known_urls = {e["url"] for e in kept.values()} | storage.task_urls(ctx.conn, ctx.task_id)
    for url in _URL_RE.findall(text or ""):
        u = url.rstrip(".,;")
        if u not in known_urls and not any(u.rstrip("/") == k.rstrip("/") for k in known_urls):
            rejected.append({"id": u, "reason": "URL not in the evidence store"})
    return {e: sorted(v) for e, v in valid.items() if v}, rejected, remapped


def cited_texts(ctx: RunContext, cites: dict[int, list[int]]) -> list[str]:
    out = []
    for eid, idxs in cites.items():
        sents = storage.evidence_sentences(ctx.conn, ctx.task_id, eid, idxs)
        out += [sents[i] for i in idxs if i in sents]
    return out


async def apply_item(ctx: RunContext, item: Item, notes: list[dict], *, position: int = 0, section: str = "new",
                     claim_id: int | None = None) -> int:
    """Validate and store one item (a new claim, or an in-place revision when claim_id is given)."""
    label = f"claim {claim_id}" if claim_id else f"{section} item {position + 1}"
    cites, rejected, remapped = resolve_citations(ctx, item.sentence_ids, item.text, notes)
    if remapped:
        await ctx.emit("write", f"{label} cited notes instead of sentences; mapped "
                       + ", ".join(f"{k} -> {', '.join(v) or 'nothing'}" for k, v in remapped.items()),
                       level="warning", detail={"remapped": remapped})
    if not cites and not rejected:
        rejected.append({"id": "", "reason": "no citation"})
    if rejected:
        await ctx.emit("write", f"{label} rejected: " + "; ".join(f"{r['id'] or '-'} ({r['reason']})" for r in rejected),
                       level="warning", detail={"rejected": rejected})
    texts = cited_texts(ctx, cites)
    ev = {e["id"]: e for e in storage.get_evidence(ctx.conn, ctx.task_id, ids=list(cites))}

    ctype = item.claim_type
    level = check_design(item.evidence_level, texts)
    if level != item.evidence_level:
        await ctx.emit("check", f"{label}: evidence level '{item.evidence_level}' not stated in the cited sentences; "
                       "set to 'not stated'", level="warning")
    population = POPULATION_NOT_STATED
    if item.population and item.population.strip().lower() not in ("null", "none", "not stated", POPULATION_NOT_STATED):
        joined = " ".join(texts)
        if contains(joined, item.population) or any(population_supported(item.population, t) for t in texts):
            population = item.population
        else:
            await ctx.emit("check", f"{label}: population not stated in the cited sentences", level="warning")
    source_dates = [ev[e].get("source_date") or ev[e].get("evidence_date") for e in ev]
    source_dates += [q.value for t in texts for q in numbers.quantities(t) if q.kind == "date" and len(q.value) > 4]
    item_date = check_item_date(item.date, source_dates)
    if item.date and item_date != item.date:
        await ctx.emit("check", f"{label}: date {item.date} not found in cited sources; "
                       f"using {item_date or 'date not stated'}", level="warning")
    anchor = (storage.get_task(ctx.conn, ctx.task_id) or {}).get("anchor_date")
    if before_anchor(item_date, anchor) and ctype != "uncertain":
        ctype = "uncertain"
        await ctx.emit("check", f"{label}: dated {item_date}, before the anchor date {anchor}; marked uncertain",
                       level="warning")
    impact = item.impact
    if impact and numbers.check(impact, texts).status == "fail":
        await ctx.emit("check", f"{label}: practice-impact note states numbers not in the cited sentences; dropped",
                       level="warning", detail={"impact": impact})
        impact = None
    fields = dict(text=item.text, claim_type=ctype, evidence_level=level, population=population, item_date=item_date,
                  change_type=item.change_type, impact=impact)
    if claim_id is None:
        claim_id = storage.add_claim(ctx.conn, ctx.task_id, position=position, section=section, rejected=rejected,
                                     **fields)
    else:
        storage.update_claim(ctx.conn, claim_id, revised=1, rejected=rejected, verdict=None, **fields)
        storage.delete_citations(ctx.conn, claim_id)
    for eid, idxs in cites.items():
        storage.add_citation(ctx.conn, claim_id, eid, idxs)
    if rejected:
        storage.update_claim(ctx.conn, claim_id, verdict="rejected")
    return claim_id


async def persist_draft(ctx: RunContext, draft: Draft) -> list[int]:
    notes = storage.get_notes(ctx.conn, ctx.task_id)
    ids, pos = [], 0
    for section, items in (("new", draft.items), ("conflict", draft.conflicts)):
        for item in items:
            ids.append(await apply_item(ctx, item, notes, position=pos, section=section))
            pos += 1
    return ids


def verified_items(claims: list[dict], anchor: str | None) -> list[dict]:
    """'What's new' items that passed all three gates, are still active and are dated on/after the anchor."""
    return [c for c in claims if c["section"] == "new" and c.get("status", "active") == "active"
            and claim_verdict(c) in SUPPORTING and not before_anchor(c.get("item_date"), anchor)]


def _allowed_numbers(items: list[dict], anchor: str | None = None) -> set[str]:
    out: set[str] = {anchor} if anchor else set()
    for c in items:
        if c.get("item_date"):
            out.add(c["item_date"])
        out |= numbers.numbers_in(c["text"])
        for x in c["citations"]:
            for s in x.get("sentences", []):
                out |= numbers.numbers_in(s["text"])
    return out


def unverified_numbers(text: str, allowed: set[str]) -> list[str]:
    """Quantities in text not backed by `allowed`. A date matches a more precise allowed date ("June 2024" is backed
    by the anchor 2024-06-01), never the reverse."""
    out = []
    for q in numbers.quantities(text):
        ok = any(a.startswith(q.value) for a in allowed) if q.kind == "date" else q.value in allowed
        if not ok and q.value not in out:
            out.append(q.value)
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
    allowed = _allowed_numbers(items, anchor)
    new = [c for c in claims if c["section"] == "new"]
    unchanged = len(items) == len(new) and not any(c.get("revised") for c in new)
    bl = draft.bottom_line
    if bl and unchanged and not unverified_numbers(bl.text, allowed):
        return bl.model_dump()

    why = "some items were revised, removed, flagged or dated before the anchor" if bl and not unchanged else \
        "it states numbers not found in verified items" if bl else "the draft had none"
    model = ctx.settings.models.strong
    try:
        res = await ctx.llm.complete_json(model=model, system=prompts.BOTTOM_LINE_SYSTEM,
                                          user=prompts.bottom_line_user(brief, items), schema=BottomLine)
        extra = unverified_numbers(res.data.text, allowed)
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
