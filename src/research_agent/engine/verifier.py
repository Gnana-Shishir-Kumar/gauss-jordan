"""Step 7: verify each claim with three gates, in order, and store each gate's result on every citation.

  Gate 1, numbers (deterministic, no LLM): every number, unit, percentage, hazard ratio, CI, p-value, date and sample
          size in the claim must appear in the cited sentences (verbatim or normalized); CIs must be ordered.
          A failure stops here: the claim is unsupported and goes to repair.
  Gate 2, identifiers: every PMID, NCT number and DOI in the claim, and each cited source's own identifier, must
          resolve (PubMed, ClinicalTrials.gov, Crossref; cached) and the resolved title must match the stored source.
          An identifier in the claim must also belong to a cited source. Service errors are recorded, not failed.
  Gate 3, entailment (FAST model): the judge sees only the claim and its cited sentences, plus one neighbouring
          sentence on each side marked as context, and may combine several cited sentences. Escalates to the STRONG
          model when the FAST call fails or returns invalid output, or when it says "partial" on a claim with numbers.
Claims rejected by the writer's citation check skip the gates and go straight to repair.
"""

from __future__ import annotations

import asyncio

from .. import prompts, storage
from ..backends.identifiers import Identifier, ids_in_text, ids_in_url, title_matches
from ..schemas import Entailment
from . import numbers
from .context import RunContext, describe_error

SKIPPED = {"status": "skipped"}


def _texts(claim: dict) -> list[str]:
    return [s["text"] for c in claim["citations"] for s in c.get("sentences", [])]


async def _resolve(ctx: RunContext, ident: Identifier):
    if ctx.ids is None:
        return None
    return await ctx.ids.resolve(ident, take=lambda api: ctx.take(api, "verify"))


async def id_gate(ctx: RunContext, claim: dict) -> tuple[str, dict[int, dict]]:
    """Returns (overall status, {evidence_id: gate dict}). Status: pass | fail | error | skipped | n/a."""
    per_cit: dict[int, list[dict]] = {}
    source_ids: dict[int, list[Identifier]] = {}
    for c in claim["citations"]:
        src = storage.get_source(ctx.conn, c["source_id"]) if c.get("source_id") else None
        ids = ids_in_url(c.get("url") or "")
        if not ids and src and c.get("trust_tier") == 2:
            ids = [i for i in ids_in_text((src.get("raw_text") or "")[:4000]) if i.kind == "doi"][:1]
        source_ids[c["evidence_id"]] = ids
        checks = []
        for ident in ids:
            r = await _resolve(ctx, ident)
            if r is None:
                checks.append({"id": str(ident), "role": "source", "status": "skipped", "detail": "no resolver"})
                continue
            entry = {"id": str(ident), "role": "source", "status": r.status, "resolved_title": r.title,
                     "detail": r.detail}
            if r.status == "found":
                ok = title_matches(r.title, c.get("title"), (src or {}).get("raw_text", "")[:3000])
                entry["title_match"] = ok
                if not ok:
                    entry["status"] = "title_mismatch"
                    entry["detail"] = f"resolved title '{r.title}' does not match stored title '{c.get('title')}'"
            checks.append(entry)
        per_cit[c["evidence_id"]] = checks

    claim_checks = []
    for ident in ids_in_text(claim["text"]):
        r = await _resolve(ctx, ident)
        if r is None:
            claim_checks.append({"id": str(ident), "role": "claim", "status": "skipped", "detail": "no resolver"})
            continue
        entry = {"id": str(ident), "role": "claim", "status": r.status, "resolved_title": r.title, "detail": r.detail}
        if r.status == "found":
            linked = any(ident in ids for ids in source_ids.values())
            if not linked:
                for c in claim["citations"]:
                    src = storage.get_source(ctx.conn, c["source_id"]) if c.get("source_id") else None
                    if src and ident in ids_in_text(src.get("raw_text") or ""):
                        linked = True
                        break
            if not linked:
                entry["status"] = "not_in_evidence"
                entry["detail"] = f"{ident} resolves but is not an identifier of any cited source"
        claim_checks.append(entry)

    every = [e for es in per_cit.values() for e in es] + claim_checks
    if not every:
        status = "n/a"
    elif any(e["status"] in ("not_found", "title_mismatch", "not_in_evidence") for e in every):
        status = "fail"
    elif any(e["status"] == "error" for e in every):
        status = "error"
    elif all(e["status"] == "skipped" for e in every):
        status = "skipped"
    else:
        status = "pass"
    out = {}
    for eid, checks in per_cit.items():
        mine = checks + claim_checks
        st = status if mine else "n/a"
        if mine and status == "fail" and not any(
                e["status"] in ("not_found", "title_mismatch", "not_in_evidence") for e in mine):
            st = "pass"
        out[eid] = {"status": st, "claim_status": status, "checks": mine}
    return status, out


def _failed_ids(gates: dict[int, dict]) -> list[str]:
    seen, out = set(), []
    for g in gates.values():
        for e in g["checks"]:
            if e["status"] in ("not_found", "title_mismatch", "not_in_evidence") and e["id"] not in seen:
                seen.add(e["id"])
                out.append(f"{e['id']}: {e['status'].replace('_', ' ')}" + (f" ({e['detail']})" if e.get("detail") else ""))
    return out


def _blocks(ctx: RunContext, claim: dict) -> list[str]:
    blocks = []
    for c in claim["citations"]:
        cited = set(c["sentence_ids"])
        want = sorted(cited | {i + d for i in cited for d in (-1, 1) if i + d >= 1})
        sents = storage.get_sentences(ctx.conn, c["source_id"], want) if c.get("source_id") else {}
        row = {"id": c["evidence_id"], "url": c.get("url"), "title": c.get("title"), "trust_tier": c.get("trust_tier"),
               "source_type": c.get("source_type"), "source_date": c.get("source_date"),
               "evidence_date": c.get("evidence_date")}
        blocks.append(prompts.source_block(row, sents, context=set(sents) - cited))
    return blocks


async def entailment(ctx: RunContext, claim: dict, sem: asyncio.Semaphore) -> tuple[dict, object | None]:
    """Gate 3 with escalation. Returns (gate dict, last LLMResult or None)."""
    models = ctx.settings.models
    user = prompts.verifier_user(claim, _blocks(ctx, claim))
    has_numbers = bool(numbers.quantities(claim["text"]))
    escalated = None
    res = None
    try:
        async with sem:
            res = await ctx.llm.complete_json(model=models.fast, system=prompts.VERIFIER_SYSTEM, user=user,
                                              schema=Entailment)
        if res.data.verdict == "partial" and has_numbers and models.strong != models.fast:
            escalated = "fast model said partial on a claim with numbers"
    except Exception as exc:
        escalated = f"fast model failed: {describe_error(exc, 150)}"
    if escalated:
        first = res
        try:
            async with sem:
                res = await ctx.llm.complete_json(model=models.strong, system=prompts.VERIFIER_SYSTEM, user=user,
                                                  schema=Entailment)
        except Exception as exc:
            if first is None:
                return ({"status": "unsupported", "note": f"verification failed ({describe_error(exc, 150)}); "
                         "treated as unsupported", "model": None, "escalated": escalated}, None)
            res = first  # keep the fast verdict if the strong call fails
            escalated += f"; strong model failed: {describe_error(exc, 150)}"
    return ({"status": res.data.verdict, "note": res.data.note, "model": res.model, "escalated": escalated}, res)


async def verify_claim(ctx: RunContext, claim: dict, phase: str, sem: asyncio.Semaphore) -> str:
    label = f"Claim {claim['position'] + 1}"
    cits = claim["citations"]
    base = {"claim_id": claim["id"], "phase": phase}
    if claim.get("rejected"):
        why = "; ".join(f"{r['id'] or '-'} ({r['reason']})" for r in claim["rejected"])
        for c in cits:
            storage.set_gates(ctx.conn, claim["id"], c["evidence_id"], verdict="rejected",
                              note=f"citation check failed: {why}", number_gate=SKIPPED, id_gate=SKIPPED,
                              entail_gate=SKIPPED)
        storage.update_claim(ctx.conn, claim["id"], verdict="rejected")
        await ctx.emit("verify", f"{label}: rejected before the gates: {why}", level="warning",
                       detail={**base, "gate": "citation", "rejected": claim["rejected"]})
        return "rejected"

    num = numbers.check(claim["text"], _texts(claim))
    await ctx.emit("check", f"{label}: number gate {num.status}", level="warning" if not num.passed else "info",
                   detail={**base, "gate": "numbers", **num.as_dict()})
    if not num.passed:
        note = "number gate failed: " + "; ".join(num.problems)
        for c in cits:
            storage.set_gates(ctx.conn, claim["id"], c["evidence_id"], verdict="unsupported", note=note,
                              number_gate=num.as_dict(), id_gate=SKIPPED, entail_gate=SKIPPED)
        storage.update_claim(ctx.conn, claim["id"], verdict="unsupported")
        return "unsupported"

    id_status, id_gates = await id_gate(ctx, claim)
    await ctx.emit("check", f"{label}: identifier gate {id_status}",
                   level="warning" if id_status in ("fail", "error") else "info",
                   detail={**base, "gate": "identifiers", "status": id_status,
                           "checks": [e for g in id_gates.values() for e in g["checks"]]})
    if id_status == "fail":
        note = "identifier gate failed: " + "; ".join(_failed_ids(id_gates))
        for c in cits:
            storage.set_gates(ctx.conn, claim["id"], c["evidence_id"], verdict="unsupported", note=note,
                              number_gate=num.as_dict(), id_gate=id_gates.get(c["evidence_id"]), entail_gate=SKIPPED)
        storage.update_claim(ctx.conn, claim["id"], verdict="unsupported")
        return "unsupported"

    ent, res = await entailment(ctx, claim, sem)
    verdict = ent["status"]
    for c in cits:
        storage.set_gates(ctx.conn, claim["id"], c["evidence_id"], verdict=verdict, note=ent["note"],
                          number_gate=num.as_dict(), id_gate=id_gates.get(c["evidence_id"]), entail_gate=ent)
    storage.update_claim(ctx.conn, claim["id"], verdict=verdict)
    msg = f"{label}: entailment {verdict}" + (f" (escalated: {ent['escalated']})" if ent.get("escalated") else "")
    detail = {**base, "gate": "entailment", "verdict": verdict, "escalated": ent.get("escalated")}
    level = "info" if verdict == "supported" else "warning"
    if res is not None:
        await ctx.emit_llm("verify", msg, res, level=level, detail=detail)
    else:
        await ctx.emit("verify", msg, level="error", detail=detail)
    return verdict


async def verify_claims(ctx: RunContext, claim_ids: list[int] | None = None, phase: str = "initial") -> None:
    claims = storage.get_claims(ctx.conn, ctx.task_id, ids=claim_ids, include_deleted=False)
    sem = asyncio.Semaphore(ctx.settings.limits.max_concurrency)
    await asyncio.gather(*(verify_claim(ctx, c, phase, sem) for c in claims if c["citations"] or c.get("rejected")))
