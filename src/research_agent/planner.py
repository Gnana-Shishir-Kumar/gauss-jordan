"""Plan step: question -> sub-queries; and the single follow-up round's replacement queries."""

from __future__ import annotations

from . import prompts, storage
from .context import RunContext
from .schemas import FollowUpPlan, Plan


async def plan(ctx: RunContext, question: str) -> list[tuple[int, str]]:
    lim = ctx.settings.limits
    res = await ctx.llm.complete_json(
        model=ctx.settings.models.planner,
        system=prompts.planner_system(lim.min_sub_queries, lim.max_sub_queries),
        user=prompts.planner_user(question),
        schema=Plan,
        context={"min": lim.min_sub_queries, "max": lim.max_sub_queries},
    )
    queries = res.data.sub_queries[: lim.max_sub_queries]
    out = [(storage.add_sub_query(ctx.conn, ctx.task_id, q, 1), q) for q in queries]
    await ctx.emit(
        "plan", f"Planned {len(out)} sub-queries",
        detail={"sub_queries": queries, "model": ctx.settings.models.planner},
        latency_ms=res.latency_ms, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
    )
    return out


async def plan_follow_up(
    ctx: RunContext, question: str, weak: list[tuple[int, str, str]], round_: int
) -> list[tuple[int, str]]:
    res = await ctx.llm.complete_json(
        model=ctx.settings.models.planner,
        system=prompts.follow_up_system(len(weak)),
        user=prompts.follow_up_user(question, [(t, why) for _, t, why in weak]),
        schema=FollowUpPlan,
        context={"max": len(weak)},
    )
    queries = res.data.queries[: len(weak)]
    out = [(storage.add_sub_query(ctx.conn, ctx.task_id, q, round_), q) for q in queries]
    await ctx.emit(
        "follow_up", f"Round {round_}: {len(out)} follow-up queries for {len(weak)} weak sub-queries",
        detail={"weak": [t for _, t, _ in weak], "queries": queries},
        latency_ms=res.latency_ms, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
    )
    return out
