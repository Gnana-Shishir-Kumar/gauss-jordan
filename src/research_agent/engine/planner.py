"""Step 2 (STRONG): brief -> 3-5 independent, fairly broad queries. Fallback: the question itself."""

from __future__ import annotations

from .. import prompts, storage
from ..schemas import Plan
from .context import RunContext, describe_error


async def plan_queries(ctx: RunContext, brief: dict) -> list[tuple[int, str]]:
    lim = ctx.settings.limits
    model = ctx.settings.models.strong
    try:
        res = await ctx.llm.complete_json(
            model=model, system=prompts.plan_system(lim.min_queries, lim.max_queries_per_round),
            user=prompts.plan_user(brief), schema=Plan,
            context={"min": lim.min_queries, "max": lim.max_queries_per_round})
        queries = res.data.queries[: lim.max_queries_per_round]
        await ctx.emit_llm("plan", f"Planned {len(queries)} queries", res, detail={"queries": queries})
    except Exception as exc:
        queries = [brief.get("question") or ""]
        await ctx.emit("plan", f"Planning failed, using the question as the only query: {describe_error(exc)}",
                       level="error", detail={"queries": queries}, model_id=model)
    return [(storage.add_sub_query(ctx.conn, ctx.task_id, q, 1, lim.round1_depth, "planned"), q) for q in queries]
