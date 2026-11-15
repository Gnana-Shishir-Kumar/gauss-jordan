"""Compare a stored run's Tavily credits with "traditional" search strategies (no new API calls).

Usage:
  uv run python evals/credit_comparison.py                       # latest finished task in research.db
  uv run python evals/credit_comparison.py --db path.db --task 3
  uv run python evals/credit_comparison.py --markdown            # table for the README / slides
"""

from __future__ import annotations

import argparse
import json
import sys

from research_agent import storage
from research_agent.config import get_settings
from research_agent.credits import compare, to_markdown
from research_agent.engine.orchestrator import load_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="SQLite path (default from config)")
    p.add_argument("--task", type=int, help="task id (default: latest finished task)")
    p.add_argument("--markdown", action="store_true", help="print a markdown table")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    args = p.parse_args(argv)

    conn = storage.connect(args.db or get_settings().db_path)
    task_id = args.task or (conn.execute(
        "SELECT max(id) FROM tasks WHERE status IN ('done', 'insufficient_evidence')").fetchone()[0])
    if not task_id or not storage.get_task(conn, task_id):
        print("No finished task found.", file=sys.stderr)
        return 1
    result = compare(load_report(conn, task_id))
    if args.json:
        print(json.dumps(result, indent=2))
    elif args.markdown:
        print(to_markdown(result))
    else:
        s = result["stats"]
        print(f"Task {task_id}: {s['searches']} searches, {s['raw_results']} results "
              f"({s['unique_results']} unique), {s['extracted_urls']} pages extracted by the project\n")
        print(f"  {'Strategy':<40} {'search':>7} {'extract':>8} {'total':>7}  vs project")
        for r in result["rows"]:
            print(f"  {r['strategy']:<40} {r['search_credits']:>7g} {r['extract_credits']:>8g} {r['total']:>7g}  "
                  f"{r['vs_project']}x")
        rep = result["repeat_within_cache"]
        print(f"\n  Same question again within 7 days: project 0 (cache) vs "
              f"{', '.join(f'{t:g}' for t in rep['traditional'])}")
        print("\n  Project row = credits Tavily billed. Other rows = same queries and results at published rates "
              "(search basic 1 / advanced 2; extract 1 per 5 URLs, rounded up per call).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
