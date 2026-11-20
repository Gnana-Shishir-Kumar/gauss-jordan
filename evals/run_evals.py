"""Run the eval questions end to end and save one timestamped results file.

Usage:
  uv run python evals/run_evals.py                  # all 15 questions (live: uses API keys and Tavily credits)
  uv run python evals/run_evals.py --limit 3        # first 3 only

Each run uses a fresh SQLite DB (evals/results/<stamp>.db) so cached results never hide credit costs.
Results: evals/results/<stamp>.json with model IDs, prompt version, per-question rows (including the
clinician_rating / clinician_notes fields for later expert review) and a pooled summary.
Then gate on drift: uv run python evals/compare.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research_agent import storage
from research_agent.config import get_settings
from research_agent.engine import run_all
from research_agent.engine.orchestrator import default_clients
from research_agent.metrics import aggregate, task_counts
from research_agent.prompts import PROMPT_VERSION

HERE = Path(__file__).parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_questions(questions: list[dict], *, llm, search, conn, settings) -> list[dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        report = await run_all(q["question"], llm=llm, search=search, conn=conn, settings=settings)
        latency = round(time.perf_counter() - t0, 1)
        t = report["task"]
        rows.append({
            "id": q["id"], "cancer_type": q.get("cancer_type"), "question": q["question"],
            "expect_insufficient": q.get("expect_insufficient", False), "task_id": t["id"], "status": t["status"],
            "stop_reason": t.get("stop_reason"), "latency_s": latency,
            "counts": task_counts(report, q.get("expect_insufficient", False)),
            "clinician_rating": q.get("clinician_rating"), "clinician_notes": q.get("clinician_notes"),
        })
        print(f"  {q['id']:<26} {t['status']:<22} {latency:>6}s  {report['totals']['credits']:g} credits",
              file=sys.stderr, flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0, help="skip the first N questions (to split a run into shards)")
    p.add_argument("--tavily-calls", type=int, help="per-brief Tavily call budget (default from config)")
    p.add_argument("--repair-reserve", type=int, help="of which reserved for repair")
    p.add_argument("--tag", default="", help="suffix for the result files (e.g. shard name)")
    p.add_argument("--out-dir", default=str(HERE / "results"))
    args = p.parse_args(argv)

    settings = get_settings()
    if args.tavily_calls is not None or args.repair_reserve is not None:
        from dataclasses import replace
        b = settings.budget
        settings = replace(settings, budget=replace(
            b, tavily_calls=args.tavily_calls if args.tavily_calls is not None else b.tavily_calls,
            tavily_repair_reserve=args.repair_reserve if args.repair_reserve is not None else b.tavily_repair_reserve))
    try:
        llm, search = default_clients(settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    questions = load_questions(Path(args.questions))[args.offset:]
    questions = questions[: args.limit] if args.limit else questions
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"-{args.tag}" if args.tag else "")
    conn = storage.connect(str(out_dir / f"{stamp}.db"))
    print(f"Running {len(questions)} questions (prompt {PROMPT_VERSION})", file=sys.stderr)
    rows = asyncio.run(run_questions(questions, llm=llm, search=search, conn=conn, settings=settings))
    result = {"run_at": stamp, "model_ids": settings.model_ids(), "prompt_version": PROMPT_VERSION,
              "limits": asdict(settings.limits), "budget": asdict(settings.budget),
              "summary": aggregate(rows), "questions": rows}
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "counts"}, indent=2))
    print(f"\nSaved {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
