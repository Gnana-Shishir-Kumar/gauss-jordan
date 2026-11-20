"""Baseline comparison for the demo: a plain "search + one LLM answer" vs this pipeline, on one question.

Baseline: ONE Tavily search (advanced, same date filter, no domain allowlist), then ONE LLM call that answers from
the search snippets and lists its claims with the URL each rests on. No verification.
Then every baseline claim is put through this project's verifier, exactly as our own claims are:
  - a claim citing a URL that was not in the search results is rejected (fabricated reference);
  - the cited page is fetched (one Tavily Extract call) and split into sentences; the claim is matched to the page's
    sentences that contain its numbers (or, if it has none, the most similar sentences);
  - number gate, identifier gate and entailment gate run unchanged.
The pipeline side is a stored run of the same question (--task, --db), or a fresh run with --run-pipeline.

Usage:
  uv run python evals/baseline_compare.py --db research.db.v4 --task 2
  uv run python evals/baseline_compare.py --run-pipeline "What has changed for ... since 2024-06-01?"
Writes evals/results/baseline-<stamp>.json and .md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from research_agent import storage
from research_agent.backends.identifiers import IdentifierResolver
from research_agent.config import get_settings
from research_agent.engine import numbers, run_all
from research_agent.engine.context import RunContext
from research_agent.engine.orchestrator import default_clients, tier_rules
from research_agent.engine.research import settle_sources
from research_agent.engine.verifier import verify_claims
from research_agent.evidence import content_hash, domain_of, normalize_url
from research_agent.prompts import PROMPT_VERSION, UNTRUSTED_RULE
from research_agent.render import claim_verdict, gate_summary
from research_agent.sentences import select, split_sentences
from research_agent.sources import classify, tier_of

HERE = Path(__file__).parent

BASELINE_SYSTEM = f"""You answer an oncologist's question from web search results. {UNTRUSTED_RULE}
Write a short answer, then list the specific factual claims you made, each with the URL of the search result it comes
from. Reply with only JSON: {{"answer": "...", "claims": [{{"text": "...", "url": "https://..."}}]}}"""


class BaselineClaim(BaseModel):
    text: str
    url: str = ""


class BaselineAnswer(BaseModel):
    answer: str = ""
    claims: list[BaselineClaim] = Field(default_factory=list)

    @field_validator("claims", mode="before")
    @classmethod
    def _claims(cls, v):
        return [c for c in (v or []) if isinstance(c, dict) and str(c.get("text") or "").strip()]


def _sentences_for(claim: str, sentences: list[str]) -> list[int]:
    """Page sentences the claim most plausibly rests on: those containing its numbers, else the most similar."""
    wanted = {q.value for q in numbers.quantities(claim)}
    if wanted:
        hits = [i for i, s in enumerate(sentences, 1) if wanted & {q.value for q in numbers.quantities(s)}]
        if hits:
            return hits[:4]
    return select(sentences, claim, k=3, max_chars=1500)


async def baseline(question: str, anchor: str | None, *, llm, search, settings, conn) -> dict:
    res = await search.search(question, depth="advanced", max_results=8, include_domains=[], start_date=anchor)
    results = [{"url": h.url, "title": h.title, "content": h.content} for h in res.hits]
    blocks = "\n\n".join(f'<untrusted_source url="{r["url"]}" title="{r["title"]}">\n{r["content"]}\n</untrusted_source>'
                         for r in results)
    out = await llm.complete_json(model=settings.models.strong, system=BASELINE_SYSTEM,
                                  user=f"Search results (untrusted web data):\n\n{blocks}\n\nQuestion: {question}",
                                  schema=BaselineAnswer)
    ans = out.data
    # Verify every baseline claim with this project's verifier.
    tid = storage.create_task(conn, f"[baseline] {question}", settings.model_ids(), PROMPT_VERSION)
    storage.update_task(conn, tid, anchor_date=anchor, status="verifying", brief={"question": question})
    sq = storage.add_sub_query(conn, tid, question)
    known = {normalize_url(r["url"]): r for r in results}
    cited = sorted({normalize_url(c.url) for c in ans.claims if normalize_url(c.url) in known})
    raw = {normalize_url(u): u for u in (r["url"] for r in results)}
    ext = await search.extract([raw[u] for u in cited]) if cited else None
    pages = {normalize_url(u): t for u, t in (ext.pages if ext else {}).items()}
    ev: dict[str, tuple[int, list[str]]] = {}
    for u in cited:
        r = known[u]
        sid = storage.upsert_source(conn, u, domain_of(u), r["title"], None, content_hash(pages.get(u, "")),
                                    tier_of(domain_of(u)) or 4, classify(u, domain_of(u)) or "news")
        sents = split_sentences(pages.get(u, ""))
        storage.set_source_text(conn, sid, pages.get(u, ""), sents, "tavily_extract_basic")
        ev[u] = (storage.add_evidence(conn, tid, sq, sid, "", 1.0, bool(sents), None if sents else
                                      "source_unavailable: extraction failed"), sents)
    for pos, c in enumerate(ans.claims):
        u = normalize_url(c.url)
        if u not in known:
            storage.add_claim(conn, tid, c.text, pos, "fact",
                              rejected=[{"id": c.url or "(none)", "reason": "URL not in the search results"}])
            continue
        eid, sents = ev[u]
        cid = storage.add_claim(conn, tid, c.text, pos, "fact",
                                rejected=[] if sents else [{"id": c.url, "reason": "page text could not be fetched"}])
        if sents:
            storage.add_citation(conn, cid, eid, _sentences_for(c.text, sents))
    ids = IdentifierResolver.from_settings(conn, settings)
    ctx = RunContext(conn, tid, settings, llm, search, None, ids=ids)
    await settle_sources(ctx, storage.get_evidence(conn, tid, kept_only=True))  # same tier + date rules as ours
    await verify_claims(ctx, phase="baseline")
    rows = []
    for c in storage.get_claims(conn, tid):
        labels, uncertain, why = tier_rules(c) if claim_verdict(c) == "supported" else ([], False, None)
        tiers = sorted({x.get("trust_tier") for x in c["citations"] if x.get("trust_tier")})
        outcome = ("flagged: " + claim_verdict(c)) if claim_verdict(c) != "supported" else             ("shown as uncertain: " + why) if uncertain else ("preliminary" if labels else "verified")
        rows.append({"claim": c["text"], "verdict": claim_verdict(c), "checks": gate_summary(c),
                     "tier": "/".join(map(str, tiers)) or "-", "outcome": outcome,
                     "url": (c["citations"][0]["url"] if c["citations"] else (c["rejected"] or [{}])[0].get("id"))})
    return {"answer": ans.answer, "search_results": results, "claims": rows,
            "credits": res.credits + (ext.credits if ext else 0), "llm_tokens": [out.input_tokens, out.output_tokens]}


def pipeline_summary(report: dict) -> dict:
    claims = [c for c in report["claims"] if c["section"] == "new"]
    ok = [c for c in claims if c["status"] == "active" and c.get("verdict") == "supported"]
    return {"task_id": report["task"]["id"], "status": report["task"]["status"],
            "claims_written": len(claims),
            "shown_verified": len(ok),
            "verified_tier12": sum(c["claim_type"] != "uncertain" for c in ok),
            "verified_but_uncertain_news_only": sum(c["claim_type"] == "uncertain" for c in ok),
            "removed_or_unverified": sum(c["status"] in ("deleted", "unverified") for c in claims),
            "credits": report["totals"]["credits"]}


def to_markdown(question: str, base: dict, pipe: dict | None) -> str:
    flagged = [r for r in base["claims"] if r["verdict"] != "supported"]
    weak = [r for r in base["claims"] if r["outcome"].startswith("shown as uncertain")]
    lines = [f"Question: {question}", "",
             f"**Baseline** (1 search + 1 LLM answer, no checks): {len(base['claims'])} claims. Through this "
             f"project's checks: **{len(flagged)} fail a verifier gate**, **{len(weak)} rest only on news or press "
             f"releases** (would be shown as uncertain), {len(base['claims']) - len(flagged) - len(weak)} pass. "
             f"{base['credits']:g} Tavily credits.", "",
             "| Baseline claim | Gates | Source tier | Outcome here | Cited URL |", "|---|---|---|---|---|"]
    for r in base["claims"]:
        lines.append(f"| {r['claim'].replace('|', '/')} | {r['checks']} | {r['tier']} | {r['outcome']} | "
                     f"{r['url'] or '-'} |")
    if pipe:
        lines += ["", f"**This pipeline** (task {pipe['task_id']}, {pipe['status']}): {pipe['claims_written']} claims "
                  f"written, {pipe['shown_verified']} pass every gate ({pipe['verified_tier12']} backed by tier 1-3 "
                  f"sources, {pipe['verified_but_uncertain_news_only']} news-only and shown as uncertain), "
                  f"{pipe['removed_or_unverified']} removed or marked unverified by repair; "
                  f"{pipe['credits']:g} Tavily credits."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="research.db.v4", help="DB with the stored pipeline run")
    p.add_argument("--task", type=int, help="stored pipeline task to compare against (its question is reused)")
    p.add_argument("--run-pipeline", metavar="QUESTION", help="run the pipeline fresh on this question instead")
    p.add_argument("--out-dir", default=str(HERE / "results"))
    args = p.parse_args(argv)
    settings = get_settings()
    try:
        llm, search = default_clients(settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    conn = storage.connect(args.db)
    if args.run_pipeline:
        from research_agent.engine.orchestrator import load_report
        report = asyncio.run(run_all(args.run_pipeline, llm=llm, search=search, conn=conn, settings=settings))
        llm, search = default_clients(settings)  # fresh clients for the next event loop
        report = load_report(conn, report["task"]["id"])
    elif args.task:
        from research_agent.engine.orchestrator import load_report
        report = load_report(conn, args.task)
    else:
        p.error("give --task ID or --run-pipeline QUESTION")
    task = report["task"]
    question = task["question"]
    settings = replace(settings, budget=replace(settings.budget, tavily_calls=0))
    base = asyncio.run(baseline(question, task.get("anchor_date"), llm=llm, search=search, settings=settings,
                                conn=conn))
    pipe = pipeline_summary(report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = to_markdown(question, base, pipe)
    (out_dir / f"baseline-{stamp}.json").write_text(json.dumps({"question": question, "baseline": base,
                                                                "pipeline": pipe}, indent=2), encoding="utf-8")
    (out_dir / f"baseline-{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
