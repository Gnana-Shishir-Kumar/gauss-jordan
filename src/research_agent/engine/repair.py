"""Step 8: bounded repair loop, one failed claim per repair call (never a rewrite of the whole answer).

For each claim that failed a gate (or was rejected by the citation check), in up to `max_repair_rounds` rounds:
  1. If the number gate failed, code first looks for stored sentences of this brief that contain every number in
     the claim and re-cites to them (no LLM call). The claim then goes through all gates again.
  2. Otherwise the STRONG model picks ONE action for that claim alone: re-cite from existing evidence (notes), run
     one targeted search from the reserved Tavily quota, delete the claim, or mark it unverified.
After the last round, a claim still failing the number gate is deleted (a number that cannot be sourced from stored
evidence is removed, never invented); any other failing claim is marked unverified and stays flagged.
Claims are independent, so their repair calls run concurrently; each call sees only its own claim.
"""

from __future__ import annotations

import asyncio

from .. import prompts, storage
from ..schemas import Item, RepairAction
from . import numbers
from .context import RunContext, describe_error
from .research import ResearchState, targeted_search
from .verifier import verify_claims
from .writer import _index_rows, apply_item

FAILING = ("partial", "unsupported", "rejected")


def failing_claims(ctx: RunContext) -> list[dict]:
    return [c for c in storage.get_claims(ctx.conn, ctx.task_id, include_deleted=False)
            if c["status"] == "active" and c.get("verdict") in FAILING]


def _gate_failures(claim: dict) -> list[str]:
    out = []
    for r in claim.get("rejected") or []:
        out.append(f"citation check: {r['id'] or '(none)'} - {r['reason']}")
    seen = set()
    for x in claim["citations"]:
        for gate in ("number_gate", "id_gate", "entail_gate"):
            g = x.get(gate) or {}
            if gate == "number_gate" and g.get("status") == "fail":
                msg = "number gate: " + "; ".join(g.get("problems") or [])
            elif gate == "id_gate" and g.get("status") == "fail":
                msg = "identifier gate: " + "; ".join(
                    f"{e['id']} {e['status'].replace('_', ' ')}" for e in g.get("checks", [])
                    if e["status"] in ("not_found", "title_mismatch", "not_in_evidence"))
            elif gate == "entail_gate" and g.get("status") in ("partial", "unsupported"):
                msg = f"entailment: {g['status']} - {g.get('note') or ''}"
            else:
                continue
            if msg not in seen:
                seen.add(msg)
                out.append(msg)
    return out


def number_failed(claim: dict) -> bool:
    return any((x.get("number_gate") or {}).get("status") == "fail" for x in claim["citations"])


def find_number_support(ctx: RunContext, claim: dict, max_sentences: int = 3) -> dict[int, list[int]] | None:
    """Stored sentences of this brief that together contain every number in the claim (cited sources first)."""
    wanted, _ = numbers.extract(claim["text"])
    if not wanted:
        return None
    cited = [x["evidence_id"] for x in claim["citations"]]
    rows = storage.get_evidence(ctx.conn, ctx.task_id, kept_only=True)
    rows.sort(key=lambda r: (r["id"] not in cited, r["id"]))
    for r in rows:  # prefer support from a single source
        sents = storage.get_sentences(ctx.conn, r["source_id"])
        chosen: list[int] = []
        missing = list(wanted)
        while missing and len(chosen) < max_sentences:
            best, best_cover = None, []
            for i, t in sents.items():
                if i in chosen:
                    continue
                have = numbers.quantities(t)
                cover = [q for q in missing if any(numbers._matches(q, h) for h in have)]
                if len(cover) > len(best_cover):
                    best, best_cover = i, cover
            if best is None:
                break
            chosen.append(best)
            missing = [q for q in missing if q not in best_cover]
        if not missing and chosen:
            texts = [sents[i] for i in chosen]
            if numbers.check(claim["text"], texts).passed:
                return {r["id"]: sorted(chosen)}
    return None


def _as_item(claim: dict, text: str | None, sentence_ids: list[str]) -> Item:
    return Item(text=text or claim["text"], claim_type=claim["claim_type"], sentence_ids=sentence_ids,
                date=claim.get("item_date"), evidence_level=claim.get("evidence_level") or "not stated",
                population=claim.get("population"), change_type=claim.get("change_type") or "other",
                impact=claim.get("impact"))


async def _log(ctx: RunContext, claim: dict, round_: int, action: str, detail: str, level: str = "info", res=None):
    storage.append_repair_log(ctx.conn, claim["id"], {"round": round_, "action": action, "detail": detail})
    msg = f"Claim {claim['position'] + 1}: {action} - {detail}"
    info = {"claim_id": claim["id"], "round": round_, "action": action}
    if res is not None:
        await ctx.emit_llm("repair", msg, res, level=level, detail=info)
    else:
        await ctx.emit("repair", msg, level=level, detail=info)


async def repair_one(ctx: RunContext, brief: dict, claim: dict, round_: int, last_round: bool, state: ResearchState,
                     notes: list[dict], sem: asyncio.Semaphore) -> int | None:
    """One repair action for one claim. Returns the claim id if it was revised and must be verified again."""
    if number_failed(claim) and not claim.get("rejected"):
        support = find_number_support(ctx, claim)
        if support:
            ids = [f"E{e}-{i}" for e, idxs in support.items() for i in idxs]
            await apply_item(ctx, _as_item(claim, None, ids), notes, position=claim["position"],
                             section=claim["section"], claim_id=claim["id"])
            await _log(ctx, claim, round_, "recite", f"re-cited in code to {', '.join(ids)}, which state every number")
            return claim["id"]

    # Searches cost one search call plus at least one extract call; none are offered in the last round.
    searches_left = 0 if last_round else ctx.quota.left("tavily", "repair") // 2
    model = ctx.settings.models.strong
    try:
        async with sem:
            res = await ctx.llm.complete_json(
                model=model, system=prompts.REPAIR_SYSTEM, schema=RepairAction,
                user=prompts.repair_user(brief, claim, _gate_failures(claim), searches_left, notes,
                                         _index_rows(ctx, notes)))
    except Exception as exc:
        await _log(ctx, claim, round_, "error", f"repair call failed, claim stays flagged: {describe_error(exc)}",
                   level="error")
        return None
    a: RepairAction = res.data
    if a.action == "recite":
        await apply_item(ctx, _as_item(claim, a.text, a.sentence_ids), notes, position=claim["position"],
                         section=claim["section"], claim_id=claim["id"])
        await _log(ctx, claim, round_, "recite", f"re-cited to {', '.join(a.sentence_ids)}"
                   + (" with revised text" if a.text and a.text != claim["text"] else "") + f"; {a.reason}", res=res)
        return claim["id"]
    if a.action == "search":
        if searches_left < 1:
            await _log(ctx, claim, round_, "search_refused", "no reserved search quota left", "warning", res=res)
            return None
        await _log(ctx, claim, round_, "search", f"targeted search '{a.query}'; {a.reason}", res=res)
        new = await targeted_search(ctx, brief, a.query or "", state, round_)
        await _log(ctx, claim, round_, "search_result", f"{len(new)} new source(s) kept",
                   "info" if new else "warning")
        return None
    if a.action == "delete":
        storage.update_claim(ctx.conn, claim["id"], status="deleted")
        await _log(ctx, claim, round_, "delete", a.reason or "deleted by repair", "warning", res=res)
        return None
    storage.update_claim(ctx.conn, claim["id"], status="unverified", claim_type="uncertain")
    await _log(ctx, claim, round_, "unverified", a.reason or "marked unverified by repair", "warning", res=res)
    return None


async def repair_loop(ctx: RunContext, brief: dict, state: ResearchState) -> dict:
    rounds = ctx.settings.limits.max_repair_rounds
    sem = asyncio.Semaphore(ctx.settings.limits.max_concurrency)
    summary = {"rounds": 0, "deleted": 0, "unverified": 0}
    for r in range(1, rounds + 1):
        failed = failing_claims(ctx)
        if not failed:
            break
        summary["rounds"] = r
        await ctx.emit("repair", f"Repair round {r}/{rounds}: {len(failed)} failed claim(s)",
                       detail={"round": r, "claims": [c["id"] for c in failed]})
        notes = storage.get_notes(ctx.conn, ctx.task_id)[: ctx.settings.limits.max_notes_for_writer]
        results = await asyncio.gather(*(repair_one(ctx, brief, c, r, r == rounds, state, notes, sem) for c in failed))
        revised = [cid for cid in results if cid]
        if revised:
            await verify_claims(ctx, revised, phase=f"repair{r}")

    for c in failing_claims(ctx):
        if number_failed(c):
            storage.update_claim(ctx.conn, c["id"], status="deleted")
            summary["deleted"] += 1
            await _log(ctx, c, summary["rounds"], "delete",
                       "number(s) could not be sourced from stored evidence after repair; claim removed", "warning")
        else:
            storage.update_claim(ctx.conn, c["id"], status="unverified", claim_type="uncertain")
            summary["unverified"] += 1
            await _log(ctx, c, summary["rounds"], "unverified", "still failing after repair; marked unverified",
                       "warning")
    return summary
