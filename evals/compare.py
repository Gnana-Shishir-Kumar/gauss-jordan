"""Drift gate: compare the latest eval run with a saved baseline.

Exits 1 if citation support rate or numeric check pass rate drops by more than --threshold percentage
points (default 3), or if either metric is missing from the latest run. Rerun evals and this gate before
changing any model ID or prompt (bump PROMPT_VERSION in prompts.py when a prompt changes).

Usage:
  uv run python evals/compare.py                      # latest results/*.json vs evals/baseline.json
  uv run python evals/compare.py --set-baseline       # accept the latest run as the new baseline
Exit codes: 0 pass, 1 regression, 2 missing input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
GATED = ("citation_support_rate", "numeric_check_pass_rate")
SHOWN = GATED + ("tier12_support_share", "preliminary_labeled_share", "insufficient_evidence_accuracy",
                 "avg_credits", "avg_latency_s")


def latest_result(results_dir: Path) -> Path | None:
    files = sorted(f for f in results_dir.glob("*.json") if not f.name.startswith(("seeded-", "baseline-")))
    return files[-1] if files else None


def compare(baseline: dict, latest: dict, threshold: float) -> list[str]:
    problems = []
    for m in GATED:
        b, l = baseline["summary"].get(m), latest["summary"].get(m)
        if b is None:
            continue
        if l is None:
            problems.append(f"{m}: missing from latest run (baseline {b})")
        elif b - l > threshold:
            problems.append(f"{m}: dropped {b - l:.1f} points ({b} -> {l}), threshold {threshold}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", default=str(HERE / "baseline.json"))
    p.add_argument("--latest", help="results file (default: newest in evals/results/)")
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--threshold", type=float, default=3.0, help="max allowed drop in percentage points")
    p.add_argument("--set-baseline", action="store_true", help="save the latest run as the baseline")
    args = p.parse_args(argv)

    latest_path = Path(args.latest) if args.latest else latest_result(Path(args.results_dir))
    if not latest_path or not latest_path.is_file():
        print("No eval results found. Run: uv run python evals/run_evals.py", file=sys.stderr)
        return 2
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    baseline_path = Path(args.baseline)

    if args.set_baseline:
        baseline_path.write_text(json.dumps({k: latest[k] for k in ("run_at", "model_ids", "prompt_version", "summary")},
                                            indent=2), encoding="utf-8")
        print(f"Baseline set from {latest_path.name}")
        return 0
    if not baseline_path.is_file():
        print(f"No baseline at {baseline_path}. Accept a run with --set-baseline.", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    print(f"Baseline: {baseline.get('run_at')}  prompt {baseline.get('prompt_version')}  {baseline.get('model_ids')}")
    print(f"Latest:   {latest.get('run_at')}  prompt {latest.get('prompt_version')}  {latest.get('model_ids')}\n")
    for m in SHOWN:
        b, l = baseline["summary"].get(m), latest["summary"].get(m)
        delta = f"{l - b:+.1f}" if isinstance(b, (int, float)) and isinstance(l, (int, float)) else "n/a"
        gate = " (gated)" if m in GATED else ""
        print(f"  {m:<32} {b!s:>8} -> {l!s:>8}  {delta:>7}{gate}")

    problems = compare(baseline, latest, args.threshold)
    if problems:
        print("\nDRIFT GATE FAILED:")
        for pr in problems:
            print(f"  - {pr}")
        return 1
    print("\nDrift gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
