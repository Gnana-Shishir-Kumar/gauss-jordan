"""Step 5 (STRONG): assess notes against the brief; propose follow-ups for thin/conflicting aspects.

Independence is computed in code: an aspect is covered only when its evidence spans >= 2 distinct domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import prompts, storage
from ..evidence import near_duplicate
from ..schemas import Coverage
from .context import RunContext, describe_error


@dataclass
class CoverageResult:
    complete: bool = False
    failed: bool = False
    follow_ups: list[str] = field(default_factory=list)
    aspects: list[dict] = field(default_factory=list)


async def assess_coverage(ctx: RunContext, brief: dict, previous: list[str]) -> CoverageResult:
    lim = ctx.settings.limits
    model = ctx.settings.models.strong
    notes = storage.get_notes(ctx.conn, ctx.task_id)
    kept = {e["id"]: e for e in storage.get_evidence(ctx.conn, ctx.task_id, kept_only=True)}
    note_ids = sorted({i for n in notes for i in n["evidence_ids"] if i in kept})
    try:
        res = await ctx.llm.complete_json(
            model=model, system=prompts.coverage_system(lim.max_queries_per_round, lim.max_follow_ups_per_aspect),
            user=prompts.coverage_user(brief, notes, [kept[i] for i in note_ids], previous), schema=Coverage)
    except Exception as exc:
        await ctx.emit("coverage", f"Coverage check failed: {describe_error(exc)}", level="error", model_id=model)
        return CoverageResult(failed=True)

    aspects = []
    for a in res.data.aspects:
        ids = [i for i in a.evidence_ids if i in kept]
        domains = sorted({kept[i]["domain"] for i in ids})
        aspects.append({"aspect": a.aspect, "status": a.status, "evidence_ids": ids, "domains": domains,
                        "independent": len(domains) >= lim.min_independent_sources})
    named = {a["aspect"].lower() for a in aspects}
    for want in brief.get("aspects", []):
        if want.lower() not in named:
            aspects.append({"aspect": want, "status": "thin", "evidence_ids": [], "domains": [], "independent": False})

    # Follow-ups: only for aspects that still lack independent sources, at most N per aspect, no near-duplicates.
    independent = {a["aspect"].lower() for a in aspects if a["independent"]}
    per_aspect: dict[str, int] = {}
    accepted, skipped, capped = [], [], []
    for f in res.data.follow_up_queries:
        key = f.aspect.lower()
        if key and key in independent:
            capped.append({"query": f.query, "aspect": f.aspect, "reason": "aspect already covered"})
        elif per_aspect.get(key, 0) >= lim.max_follow_ups_per_aspect:
            capped.append({"query": f.query, "aspect": f.aspect, "reason": "per-aspect cap"})
        elif near_duplicate(f.query, previous + accepted, lim.near_duplicate_threshold):
            skipped.append(f.query)
        elif len(accepted) < lim.max_queries_per_round:
            accepted.append(f.query)
            per_aspect[key] = per_aspect.get(key, 0) + 1
    complete = all(a["independent"] for a in aspects)
    n_ok = sum(a["independent"] for a in aspects)
    await ctx.emit_llm(
        "coverage", f"{n_ok}/{len(aspects)} aspects have >= {lim.min_independent_sources} independent sources; "
        f"{len(accepted)} follow-up queries", res,
        detail={"aspects": aspects, "follow_ups": accepted, "skipped_near_duplicates": skipped,
                "skipped_by_aspect_cap": capped})
    return CoverageResult(complete=complete, follow_ups=accepted, aspects=aspects)
