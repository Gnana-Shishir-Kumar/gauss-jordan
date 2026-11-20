"""CLI: uv run research "question" [--yes]

Prints the brief and planned queries, asks for approval (unless --yes), then the update brief with verdicts,
sources, stop reason, credits used, and the trace.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import storage
from .config import get_settings
from .engine import prepare, resume, run
from .engine.orchestrator import default_clients, load_report
from .exports import to_csv, to_ris
from .privacy import IdentifierError, check_question
from .render import USE_LIMITS, format_report

EXIT = {"done": 0, "insufficient_evidence": 3, "needs_clarification": 4}


def _print_event(event: dict) -> None:
    mark = {"error": "!", "warning": "~"}.get(event["level"], "-")
    print(f"  {mark} [{event['step']}] {event['message']}", file=sys.stderr, flush=True)


def _print_plan(report: dict) -> None:
    t = report["task"]
    b = t.get("brief") or {}
    print(f"\nBrief: {b.get('cancer_type')} | {b.get('setting')} | since {t.get('anchor_date')}"
          + (" (defaulted to 12 months ago)" if b.get("anchor_defaulted") else ""))
    print(f"Question: {b.get('question')}")
    print("Aspects: " + "; ".join(b.get("aspects", [])))
    print("Planned queries:")
    for s in report["sub_queries"]:
        print(f"  - {s['text']}")


def _export(report: dict, fmt: str, out: str | None) -> None:
    text = to_ris(report) if fmt == "ris" else to_csv(report)
    path = Path(out or f"task{report['task']['id']}_sources.{fmt}")
    path.write_text(text, encoding="utf-8")
    print(f"\nExported sources to {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(prog="research", description="Dated oncology update briefs from live web evidence. "
                                + USE_LIMITS)
    p.add_argument("question", nargs="?", help="clinical topic question (no patient identifiers)")
    p.add_argument("--yes", "-y", action="store_true", help="skip the approval step")
    p.add_argument("--guideline", help='guideline baseline you are comparing against, e.g. "NCCN NSCLC" (never fetched)')
    p.add_argument("--guideline-version", help='e.g. "v3.2025"')
    p.add_argument("--guideline-date", help="YYYY-MM-DD; used as the anchor date if the question gives none")
    p.add_argument("--run", type=int, metavar="TASK_ID", help="approve and run a prepared task")
    p.add_argument("--resume", type=int, metavar="TASK_ID",
                   help="re-run writing and verification for a task that failed after research")
    p.add_argument("--show", type=int, metavar="TASK_ID", help="print a stored report")
    p.add_argument("--export", choices=["csv", "ris"], help="export the sources used")
    p.add_argument("--out", help="export path (default: task<ID>_sources.<fmt>)")
    p.add_argument("--no-trace", action="store_true", help="omit the trace")
    p.add_argument("--quiet", action="store_true", help="no live progress on stderr")
    p.add_argument("--db", help="SQLite path (default from config)")
    args = p.parse_args(argv)

    settings = get_settings()
    try:
        conn = storage.connect(args.db or settings.db_path)
    except storage.SchemaMismatch as exc:
        print(f"{exc}\nOr point RESEARCH_DB_PATH (or --db) at a new file.", file=sys.stderr)
        return 1

    def finish(report: dict) -> int:
        print(format_report(report, show_trace=not args.no_trace))
        if args.export:
            _export(report, args.export, args.out)
        return EXIT.get(report["task"]["status"], 1)

    if args.show is not None:
        if not storage.get_task(conn, args.show):
            print(f"No task {args.show}", file=sys.stderr)
            return 1
        return finish(load_report(conn, args.show))

    if not args.question and args.run is None and args.resume is None:
        p.error("a question is required (or --run / --resume / --show TASK_ID)")
    if args.question:
        try:  # privacy check first: needs no keys and stores nothing
            check_question(args.question)
        except IdentifierError as exc:
            print(str(exc), file=sys.stderr)
            return 6
    try:
        llm, search = default_clients(settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    on_event = None if args.quiet else _print_event
    kw = dict(llm=llm, search=search, conn=conn, settings=settings, on_event=on_event)

    if args.resume is not None:
        try:
            return finish(asyncio.run(resume(args.resume, **kw)))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.run is not None:
        task_id = args.run
    else:
        try:
            guideline = {"name": args.guideline, "version": args.guideline_version, "date": args.guideline_date}
            report = asyncio.run(prepare(args.question, guideline=guideline, **kw))
        except IdentifierError as exc:
            print(str(exc), file=sys.stderr)
            return 6
        task = report["task"]
        if task["status"] != "awaiting_approval":
            return finish(report)
        _print_plan(report)
        task_id = task["id"]
        if not args.yes:
            if not sys.stdin.isatty():
                print(f"\nNot approved. Rerun with --yes, or approve later: research --run {task_id}", file=sys.stderr)
                return 5
            if input("\nStart research? [y/N] ").strip().lower() not in ("y", "yes"):
                print(f"Not started. Approve later with: research --run {task_id}", file=sys.stderr)
                return 5
    if args.run is None:
        # prepare() ran in its own event loop; its clients' pooled connections are tied to that closed loop,
        # so the research run gets fresh clients (reusing them fails with "Event loop is closed").
        llm, search = default_clients(settings)
        kw.update(llm=llm, search=search)
    try:
        report = asyncio.run(run(task_id, **kw))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return finish(report)


if __name__ == "__main__":
    sys.exit(main())
