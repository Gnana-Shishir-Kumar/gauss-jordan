"""Step 1 (STRONG): question -> structured brief, or a clarifying question."""

from __future__ import annotations

from datetime import date, timedelta

from .. import prompts, storage
from ..schemas import Brief, parse_partial_date
from . import numbers
from .context import RunContext, describe_error


def _stated(value: str | None, question: str) -> bool:
    """A guideline field is kept only if the user wrote it (case/space-insensitive)."""
    norm = lambda s: " ".join(str(s or "").lower().split())  # noqa: E731
    return bool(value) and norm(value) in norm(question)


async def make_brief(ctx: RunContext, question: str, today: date | None = None,
                     guideline: dict | None = None) -> dict:
    """Returns the stored brief dict. If clarification is needed, brief['needs_clarification'] is True.

    guideline: the user's baseline {"name", "version", "date"} (e.g. from the CLI or UI). Overrides anything the
    model extracted; model-extracted guideline fields are kept only if they appear in the question. NCCN and other
    guideline sites are never fetched for this."""
    today = today or date.today()
    model = ctx.settings.models.strong
    try:
        res = await ctx.llm.complete_json(model=model, system=prompts.BRIEF_SYSTEM,
                                          user=prompts.brief_user(today.isoformat(), question), schema=Brief)
        brief = res.data
        await ctx.emit_llm("brief", "Clarification needed" if brief.needs_clarification else "Brief written", res,
                           level="warning" if brief.needs_clarification else "info")
    except Exception as exc:
        # Safe fallback: search the question as asked, with generic aspects.
        brief = Brief(cancer_type="not extracted", setting="not extracted", question=question,
                      aspects=["regulatory and guideline changes", "new trial results"])
        await ctx.emit("brief", f"Brief failed, using the question as the brief: {describe_error(exc)}",
                       level="error", model_id=model)

    data = brief.model_dump()
    if brief.needs_clarification:
        storage.update_task(ctx.conn, ctx.task_id, brief=data, status="needs_clarification")
        return data

    for f in ("guideline_name", "guideline_version"):
        if data.get(f) and not _stated(data[f], question):
            await ctx.emit("brief", f"Dropped {f.replace('_', ' ')} '{data[f]}': not stated in the question",
                           level="warning")
            data[f] = None
    gd = data.get("guideline_date")
    if gd and not any(q.kind == "date" and (gd == q.value or (len(q.value) >= 7 and gd.startswith(q.value)))
                      for q in numbers.quantities(question)):
        await ctx.emit("brief", f"Dropped guideline date {gd}: the question does not state it", level="warning")
        data["guideline_date"] = None
    for k, f in (("name", "guideline_name"), ("version", "guideline_version"), ("date", "guideline_date")):
        if (guideline or {}).get(k):
            data[f] = parse_partial_date(guideline[k]) if k == "date" else " ".join(str(guideline[k]).split())
    anchor, defaulted = brief.anchor_date, False
    if not anchor and data.get("guideline_date"):
        g = data["guideline_date"]
        anchor = g if len(g) == 10 else f"{g}-01" if len(g) == 7 else f"{g}-01-01"
    if not anchor or anchor > today.isoformat():
        anchor = (today - timedelta(days=ctx.settings.limits.default_anchor_days)).isoformat()
        defaulted = True
    data.update(anchor_date=anchor, anchor_defaulted=defaulted)
    storage.update_task(ctx.conn, ctx.task_id, brief=data, cancer_type=data["cancer_type"],
                        setting=data["setting"], anchor_date=anchor)
    if defaulted:
        await ctx.emit("brief", f"No anchor date given; defaulting to {anchor} (12 months ago)", level="warning")
    return data
