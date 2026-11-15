"""CLI: uv run research "question" """

from __future__ import annotations

import argparse
import asyncio
import sys

from . import storage
from .config import get_settings
from .pipeline import default_clients, load_report, run_research
from .render import format_report


def _print_event(event: dict) -> None:
    mark = {"error": "!", "warning": "~"}.get(event["level"], "-")
    print(f"  {mark} [{event['step']}] {event['message']}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(prog="research", description="Sourced, verified answers to a question.")
    p.add_argument("question", nargs="?", help="the research question")
    p.add_argument("--show", type=int, metavar="TASK_ID", help="print a stored report instead of running")
    p.add_argument("--no-trace", action="store_true", help="omit the trace section")
    p.add_argument("--quiet", action="store_true", help="no live progress on stderr")
    p.add_argument("--db", help="SQLite path (default from config)")
    args = p.parse_args(argv)

    settings = get_settings()
    conn = storage.connect(args.db or settings.db_path)

    if args.show is not None:
        if not storage.get_task(conn, args.show):
            print(f"No task {args.show}", file=sys.stderr)
            return 1
        print(format_report(load_report(conn, args.show), show_trace=not args.no_trace))
        return 0

    if not args.question:
        p.error("a question is required (or use --show TASK_ID)")
    try:
        llm, search = default_clients(settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report = asyncio.run(run_research(
        args.question, llm=llm, search=search, conn=conn, settings=settings,
        on_event=None if args.quiet else _print_event,
    ))
    print(format_report(report, show_trace=not args.no_trace))
    return {"done": 0, "insufficient_evidence": 3}.get(report["task"]["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
