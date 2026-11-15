"""Step 1 (STRONG): question -> structured brief, or a clarifying question."""

from __future__ import annotations

from datetime import date, timedelta

from .. import prompts, storage
from ..schemas import Brief
from .context import RunContext, describe_error


async def make_brief(ctx: RunContext, question: str, today: date | None = None) -> dict:
    """Returns the stored brief dict. If clarification is needed, brief['needs_clarification'] is True."""
    today = today or date.today()
    model = ctx.settings.models.strong
    try:
        res = await ctx.llm.complete_json(model=model, system=prompts.brief_system(today.isoformat()),
                                          user=f"Question: {question}", schema=Brief)
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

    anchor, defaulted = brief.anchor_date, False
    if not anchor or anchor > today.isoformat():
        anchor = (today - timedelta(days=ctx.settings.limits.default_anchor_days)).isoformat()
        defaulted = True
    data.update(anchor_date=anchor, anchor_defaulted=defaulted)
    storage.update_task(ctx.conn, ctx.task_id, brief=data, cancer_type=data["cancer_type"],
                        setting=data["setting"], anchor_date=anchor)
    if defaulted:
        await ctx.emit("brief", f"No anchor date given; defaulting to {anchor} (12 months ago)", level="warning")
    return data
