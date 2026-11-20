"""Seeded-error eval: does the verifier catch errors we inject on purpose?

Takes briefs that already ran (stored in SQLite: their sources, sentences, notes and verified claims) and, for every
verified claim, builds a clean control copy plus corrupted copies:
  wrong_number        one number in the claim changed (e.g. HR 0.70 -> 0.77)
  swapped_citation    the claim cites another claim's sentences from a different source
  fabricated_id       a PMID or NCT number that does not exist is added to the claim
  stale_date          an approval date before the anchor is put in the claim
  tier4_only_efficacy an efficacy claim whose support is re-labelled as a press release (tier 4)
  number_from_other_arm (harder, beyond the five required types) a number in the claim replaced by ANOTHER number
                      that does appear in the cited sentences (e.g. the comparator arm's 16.6 months instead of the
                      drug's 23.7): the number gate passes by design, so only entailment can catch it
Each copy goes through the real pipeline steps: the three verifier gates (live Token Factory model for entailment,
live PubMed/ClinicalTrials.gov/Crossref for identifiers), then the bounded repair loop (re-cite, delete or mark
unverified; targeted searches are disabled so no Tavily credits are spent), then the code tier rules.

Reported: precision and recall per error type and per gate, citation support rate before and after repair, how
many corrupted claims would still be shown as verified (leaks), and the original briefs' credits, tokens and seconds.
Results are saved per run (evals/results/seeded-<stamp>.json) so runs can be compared over time.

Usage:
  uv run python evals/seeded_eval.py evals/results/*shard*.db research.db.v4 [--no-repair] [--max-briefs 12]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from research_agent import storage
from research_agent.backends.identifiers import IdentifierResolver
from research_agent.config import get_settings
from research_agent.engine import numbers
from research_agent.engine.context import RunContext
from research_agent.engine.orchestrator import default_clients, is_efficacy_number, tier_rules
from research_agent.engine.repair import repair_loop
from research_agent.engine.research import ResearchState
from research_agent.engine.verifier import verify_claims
from research_agent.prompts import PROMPT_VERSION
from research_agent.render import COMPANY_REPORTED

HERE = Path(__file__).parent
ERROR_TYPES = ("wrong_number", "swapped_citation", "fabricated_id", "stale_date", "tier4_only_efficacy",
               "number_from_other_arm")
FAKE_PMID, FAKE_NCT = "PMID 99999999", "NCT09999999"


# --- building cases ------------------------------------------------------------------

def _alter(q: numbers.Quantity) -> str:
    """A plausible but wrong value for a number as written: 0.70 -> 0.77, 23.7 -> 26.1, 61% -> 67%."""
    raw = q.raw
    digits = raw.rstrip("%").split("mg")[0]
    if "." in digits:
        decimals = len(digits.split(".")[1])
        v = round(float(("0" if digits.startswith(".") else "") + digits) * 1.1 + 10 ** -decimals, decimals)
        new = f"{v:.{decimals}f}"
    else:
        new = str(int(digits.replace(",", "")) + max(3, int(int(digits.replace(",", "")) * 0.1)))
    return raw.replace(digits, new, 1)


def _swap(text: str, raw: str, new: str) -> str:
    """Replace the number as written, never a piece of a longer number ('.1' inside '2025.1')."""
    # not inside a longer number, and not part of a name like HER2 or PD-1
    return re.sub(rf"(?<![\w.]){re.escape(raw)}(?!\d)", lambda m: new, text, count=1)


def make_cases(brief: dict, claims: list[dict], anchor: str | None, rng: random.Random) -> list[dict]:
    """Cases for one brief: {kind, expect_error, text, citations (evidence_id -> sentence idxs), tier_override}."""
    cases = []
    for c in claims:
        cites = {x["evidence_id"]: x["sentence_ids"] for x in c["citations"] if x["sentence_ids"]}
        if not cites:
            continue
        base = {"source_claim": c["id"], "citations": cites, "item_date": c.get("item_date"),
                "evidence_level": c.get("evidence_level"), "population": c.get("population")}
        cases.append({**base, "kind": "control", "text": c["text"]})
        nums = [q for q in numbers.quantities(c["text"]) if q.kind == "number"]
        if nums:
            q = rng.choice(nums)
            cases.append({**base, "kind": "wrong_number", "text": _swap(c["text"], q.raw, _alter(q)),
                          "detail": f"{q.raw} -> {_alter(q)}"})
        if nums:
            sent_nums = [q for x in c["citations"] for st in x.get("sentences", []) for q in numbers.quantities(st["text"])
                         if q.kind == "number" and q.value not in {n.value for n in nums}]
            if sent_nums:
                q, alt = rng.choice(nums), rng.choice(sent_nums)
                cases.append({**base, "kind": "number_from_other_arm", "text": _swap(c["text"], q.raw, alt.raw),
                              "detail": f"{q.raw} -> {alt.raw} (also in the cited sentences)"})
        others = [o for o in claims if o["id"] != c["id"]
                  and not ({x["evidence_id"] for x in o["citations"]} & set(cites))
                  and any(x["sentence_ids"] for x in o["citations"])]
        if others:
            o = rng.choice(others)
            cases.append({**base, "kind": "swapped_citation", "text": c["text"],
                          "citations": {x["evidence_id"]: x["sentence_ids"] for x in o["citations"] if x["sentence_ids"]},
                          "detail": f"cites claim {o['id']}'s sentences"})
        fake = FAKE_PMID if rng.random() < 0.5 else FAKE_NCT
        cases.append({**base, "kind": "fabricated_id", "text": c["text"].rstrip(".") + f" ({fake}).", "detail": fake})
        stale = f"{int((anchor or '2025')[:4]) - 2}-03-03"
        cases.append({**base, "kind": "stale_date", "text": c["text"].rstrip(".") + f", approved on March 3, {stale[:4]}.",
                      "item_date": stale, "detail": f"approved on {stale}"})
        if is_efficacy_number(c["text"]):
            cases.append({**base, "kind": "tier4_only_efficacy", "text": c["text"], "tier_override": 4,
                          "detail": "support re-labelled as a press release"})
    return cases


def copy_brief(src, dst, task: dict) -> tuple[int, dict[int, int]]:
    """Copy a task's kept evidence (sources, raw text, sentences) and notes into the eval DB. Returns
    (new task id, old evidence id -> new evidence id)."""
    tid = storage.create_task(dst, task["question"], task.get("model_ids"), PROMPT_VERSION)
    storage.update_task(dst, tid, brief=task["brief"], anchor_date=task.get("anchor_date"), status="verifying",
                        cancer_type=task.get("cancer_type"), setting=task.get("setting"))
    sq = storage.add_sub_query(dst, tid, "seeded eval")
    emap: dict[int, int] = {}
    for e in storage.get_evidence(src, task["id"], kept_only=True):
        s = storage.get_source(src, e["source_id"])
        sid = storage.upsert_source(dst, s["url"], s["domain"], s["title"], s["published_date"], s["content_hash"],
                                    s["trust_tier"], s["source_type"])
        sents = storage.get_sentences(src, e["source_id"])
        storage.set_source_text(dst, sid, s["raw_text"] or "", [sents[i] for i in sorted(sents)], s["fetched_via"] or "")
        dst.execute("UPDATE sources SET trust_tier = ?, source_type = ?, source_date = ?, date_source = ? WHERE id = ?",
                    (s["trust_tier"], s["source_type"], s.get("source_date"), s.get("date_source"), sid))
        emap[e["id"]] = storage.add_evidence(dst, tid, sq, sid, e["excerpt"], e["relevance_score"] or 0, True, None,
                                             e.get("evidence_date"), e["shown"])
    dst.commit()
    for n in storage.get_notes(src, task["id"]):
        refs = []
        for r in n["sentence_ids"]:
            e, i = r[1:].split("-")
            if int(e) in emap:
                refs.append(f"E{emap[int(e)]}-{i}")
        if refs:
            storage.add_note(dst, tid, sq, n["text"], sorted({int(r[1:].split("-")[0]) for r in refs}), {}, refs)
    return tid, emap


def _relabelled_copy(dst, tid: int, eid: int, tier: int, n: int) -> int:
    """A copy of one evidence row whose source is re-labelled (e.g. as a tier 4 press release), under its own URL so
    the original source keeps its tier for every other case."""
    e = storage.get_evidence(dst, tid, ids=[eid])[0]
    s = storage.get_source(dst, e["source_id"])
    sid = storage.upsert_source(dst, f"{s['url']}#seeded-tier{tier}-{n}", s["domain"], s["title"], None, None, tier,
                                "press_release" if tier == 4 else s["source_type"])
    sents = storage.get_sentences(dst, e["source_id"])
    storage.set_source_text(dst, sid, s["raw_text"] or "", [sents[i] for i in sorted(sents)], "seeded copy")
    return storage.add_evidence(dst, tid, e["sub_query_id"], sid, e["excerpt"], e["relevance_score"] or 0, True, None,
                                e.get("evidence_date"), e["shown"])


# --- running -----------------------------------------------------------------------

async def run_brief(src, dst, task: dict, rng: random.Random, settings, llm, ids, repair: bool,
                    max_claims: int | None = None) -> dict:
    t0 = time.perf_counter()
    claims = [c for c in storage.get_claims(src, task["id"], include_deleted=False)
              if c["status"] == "active" and c.get("verdict") == "supported" and c["section"] == "new"]
    if max_claims and len(claims) > max_claims:
        claims = sorted(rng.sample(claims, max_claims), key=lambda c: c["position"])
    cases = make_cases(task["brief"], claims, task.get("anchor_date"), rng)
    tid, emap = copy_brief(src, dst, task)
    for pos, case in enumerate(cases):
        cid = storage.add_claim(dst, tid, case["text"], pos, "fact", evidence_level=case.get("evidence_level"),
                                population=case.get("population"), item_date=case.get("item_date"))
        case["claim_id"] = cid
        for e, idxs in case["citations"].items():
            if e not in emap:
                continue
            eid = emap[e]
            if case.get("tier_override"):
                eid = _relabelled_copy(dst, tid, eid, case["tier_override"], pos)
            storage.add_citation(dst, cid, eid, idxs)
    ctx = RunContext(dst, tid, settings, llm, None, None, ids=ids)
    await verify_claims(ctx, phase="initial")
    before = {c["id"]: c for c in storage.get_claims(dst, tid)}
    if repair:
        await repair_loop(ctx, task["brief"], ResearchState())
    after = {c["id"]: c for c in storage.get_claims(dst, tid)}
    rows = []
    for case in cases:
        b, a = before[case["claim_id"]], after[case["claim_id"]]
        labels, _, _ = tier_rules(a) if a.get("verdict") == "supported" else ([], False, None)
        gate = first_failed_gate(b)
        if case["kind"] == "tier4_only_efficacy":
            flagged = COMPANY_REPORTED in labels or gate is not None
            gate = "tier_rule" if COMPANY_REPORTED in labels else gate
        else:
            flagged = gate is not None
        row = {"kind": case["kind"], "detail": case.get("detail"), "text": case["text"], "flagged": flagged,
               "repaired": repair, "gate": gate, "verdict_before": b.get("verdict")}
        rows.append(score_after(row, a, labels))
    totals = storage.task_totals(src, task["id"])
    return {"task_id": task["id"], "question": task["question"], "status": task["status"], "cases": rows,
            "repaired": repair,
            "brief_credits": totals["credits"], "brief_input_tokens": totals["input_tokens"],
            "brief_output_tokens": totals["output_tokens"], "brief_seconds": _seconds(task),
            "eval_tokens": storage.task_totals(dst, tid), "eval_seconds": round(time.perf_counter() - t0, 1)}


def error_still_present(row: dict, final_text: str, revised: bool) -> bool:
    """Is the injected error still in what the reader would see? (Repair may legitimately fix it: dropping a fake
    PMID, restoring the right number, re-citing the right sentences.)"""
    kind, detail = row["kind"], row.get("detail") or ""
    if kind == "wrong_number":
        bad = detail.split(" -> ", 1)[1]
        return bool(re.search(rf"(?<![\d.]){re.escape(bad)}(?!\d)", final_text))
    if kind == "number_from_other_arm":
        return final_text == row["text"]
    if kind == "fabricated_id":
        return detail in final_text
    if kind == "stale_date":
        return f"March 3, {detail.split()[-1][:4]}" in final_text
    if kind == "swapped_citation":
        return not revised  # a revised claim was re-cited and passed every gate again with its new citations
    return True  # tier4_only_efficacy: the error is the support itself


def score_after(row: dict, a: dict, labels: list[str]) -> dict:
    """Outcome after repair. shown_as_verified ignores tier labels (a company-reported efficacy claim is still shown,
    labelled); a seeded error 'leaks' only if it is still present AND would be presented as verified without the
    label that flags it."""
    shown = a["status"] == "active" and a.get("verdict") == "supported"
    leaked = row["kind"] != "control" and shown and error_still_present(row, a["text"], bool(a.get("revised"))) \
        and not (row["kind"] == "tier4_only_efficacy" and COMPANY_REPORTED in labels)
    return {**row, "verdict_after": a.get("verdict"), "status_after": a["status"], "final_text": a["text"],
            "shown_as_verified_after_repair": shown, "error_shown_after_repair": leaked,
            "repair": [e.get("action") for e in a.get("repair_log") or []]}


def rescore(json_path: Path) -> dict:
    """Recompute after-repair outcomes from a saved run's eval DB (no LLM calls)."""
    result = json.loads(json_path.read_text(encoding="utf-8"))
    db = storage.connect(str(json_path.with_suffix(".db")))
    task_ids = [r[0] for r in db.execute("SELECT id FROM tasks ORDER BY id")]
    for b, tid in zip(result["briefs"], task_ids):
        claims = storage.get_claims(db, tid)
        for row, a in zip(b["cases"], claims):
            assert a["original_text"] == row["text"] or a["revised"], (tid, row["text"][:60])
            labels, _, _ = tier_rules(a) if a.get("verdict") == "supported" else ([], False, None)
            row.update(score_after(row, a, labels))
    result["summary"] = summarize(result["briefs"])
    result["rescored"] = True
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def first_failed_gate(claim: dict) -> str | None:
    if claim.get("verdict") == "rejected" or claim.get("rejected"):
        return "citation"
    for key, name in (("number_gate", "numbers"), ("id_gate", "identifiers"), ("entail_gate", "entailment")):
        for x in claim["citations"]:
            st = (x.get(key) or {}).get("status")
            if st in ("fail", "partial", "unsupported"):
                return name
    return None if claim.get("verdict") == "supported" else "entailment"


def _seconds(task: dict) -> float | None:
    try:
        a = datetime.fromisoformat(task["created_at"])
        b = datetime.fromisoformat(task["finished_at"])
        return round((b - a).total_seconds(), 1)
    except (TypeError, ValueError, KeyError):
        return None


def summarize(briefs: list[dict]) -> dict:
    rows = [r for b in briefs for r in b["cases"]]
    controls = [r for r in rows if r["kind"] == "control"]
    bad = [r for r in rows if r["kind"] != "control"]
    fp = sum(r["flagged"] for r in controls)
    tp = sum(r["flagged"] for r in bad)

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    per_type = {}
    for k in ERROR_TYPES:
        rs = [r for r in bad if r["kind"] == k]
        per_type[k] = {"cases": len(rs), "caught": sum(r["flagged"] for r in rs), "recall": pct(sum(r["flagged"] for r in rs), len(rs)),
                       "leaked_after_repair": sum(r.get("error_shown_after_repair", False) for r in rs
                                                  if r.get("repaired", True)),
                       "in_repaired_briefs": sum(bool(r.get("repaired", True)) for r in rs),
                       "caught_by": dict(Counter(r["gate"] for r in rs if r["flagged"]))}
    per_gate = {}
    for g in ("citation", "numbers", "identifiers", "entailment", "tier_rule"):
        tpg = sum(r["gate"] == g for r in bad)
        fpg = sum(r["gate"] == g for r in controls)
        per_gate[g] = {"flags_on_errors": tpg, "flags_on_controls": fpg, "precision": pct(tpg, tpg + fpg)}
    support = lambda key, rs: pct(sum(r[key] == "supported" for r in rs), len(rs))  # noqa: E731
    rep_rows = [r for b in briefs if b.get("repaired", True) for r in b["cases"]]
    rep_controls = [r for r in rep_rows if r["kind"] == "control"]
    rep_bad = [r for r in rep_rows if r["kind"] != "control"]
    return {
        "briefs": len(briefs), "cases": len(rows), "controls": len(controls), "seeded_errors": len(bad),
        "precision": pct(tp, tp + fp), "recall": pct(tp, len(bad)), "false_positive_rate": pct(fp, len(controls)),
        "per_error_type": per_type, "per_gate": per_gate,
        "repaired_briefs": sum(bool(b.get("repaired", True)) for b in briefs), "repaired_cases": len(rep_rows),
        "citation_support_before_repair": support("verdict_before", rep_rows),
        "citation_support_after_repair": pct(sum(r["shown_as_verified_after_repair"] for r in rep_rows), len(rep_rows)),
        "controls_supported_after_repair": pct(sum(r["shown_as_verified_after_repair"] for r in rep_controls),
                                               len(rep_controls)),
        "errors_leaked_after_repair": sum(r.get("error_shown_after_repair", False) for r in rep_bad),
        "seeded_errors_in_repaired_briefs": len(rep_bad),
        "errors_fixed_or_removed_by_repair": sum(not r.get("error_shown_after_repair", False) for r in rep_bad),
        "avg_brief_credits": _avg(b["brief_credits"] for b in briefs),
        "avg_brief_input_tokens": _avg(b["brief_input_tokens"] for b in briefs),
        "avg_brief_output_tokens": _avg(b["brief_output_tokens"] for b in briefs),
        "avg_brief_seconds": _avg(b["brief_seconds"] for b in briefs if b["brief_seconds"] is not None),
        "avg_eval_seconds_per_brief": _avg(b["eval_seconds"] for b in briefs),
    }


def _avg(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


async def run(db_paths: list[str], *, max_briefs: int, repair: bool, seed: int, llm, settings, out: Path,
              ids=None, max_claims: int | None = None, repair_briefs: int | None = None) -> dict:
    settings = replace(settings, budget=replace(settings.budget, tavily_calls=0, tavily_repair_reserve=0))
    dst = storage.connect(str(out.with_suffix(".db")))
    ids = ids or IdentifierResolver.from_settings(dst, settings)
    rng = random.Random(seed)
    briefs = []
    for path in db_paths:
        src = storage.connect(path)
        tasks = [storage.get_task(src, r[0]) for r in src.execute(
            "SELECT id FROM tasks WHERE status IN ('done', 'insufficient_evidence') ORDER BY id")]
        for t in tasks:
            if len(briefs) >= max_briefs:
                break
            if not any(c.get("verdict") == "supported" and c["status"] == "active"
                       for c in storage.get_claims(src, t["id"], include_deleted=False)):
                continue
            print(f"  brief {len(briefs) + 1}: {t['question'][:80]}", file=sys.stderr, flush=True)
            do_repair = repair and (repair_briefs is None or len(briefs) < repair_briefs)
            briefs.append(await run_brief(src, dst, t, rng, settings, llm, ids, do_repair, max_claims))
    result = {"run_at": out.stem, "model_ids": settings.model_ids(), "prompt_version": PROMPT_VERSION,
              "repair": repair, "seed": seed, "sources": db_paths, "summary": summarize(briefs), "briefs": briefs}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def to_markdown(summary: dict) -> str:
    lines = [f"{summary['briefs']} briefs, {summary['controls']} clean claims, {summary['seeded_errors']} seeded errors. "
             f"Overall precision {summary['precision']}%, recall {summary['recall']}%, "
             f"false-positive rate on clean claims {summary['false_positive_rate']}%.", "",
             "| Error type | Cases | Caught | Recall | Caught by | Error still shown as verified after repair* |",
             "|---|---|---|---|---|---|"]
    for k, v in summary["per_error_type"].items():
        by = ", ".join(f"{g} {n}" for g, n in sorted(v["caught_by"].items(), key=lambda x: -x[1])) or "-"
        lines.append(f"| {k} | {v['cases']} | {v['caught']} | {v['recall']}% | {by} | "
                     f"{v['leaked_after_repair']} of {v.get('in_repaired_briefs', '?')} |")
    lines += ["", "| Gate | Flags on seeded errors | Flags on clean claims | Precision |", "|---|---|---|---|"]
    for g, v in summary["per_gate"].items():
        lines.append(f"| {g} | {v['flags_on_errors']} | {v['flags_on_controls']} | {v['precision']}% |")
    lines += ["", f"Repair ran on {summary['repaired_briefs']} briefs ({summary['repaired_cases']} cases). "
              f"Citation support rate on those cases: {summary['citation_support_before_repair']}% before repair, "
              f"{summary['citation_support_after_repair']}% shown as verified after repair; clean claims still "
              f"verified after repair: {summary['controls_supported_after_repair']}%.",
              f"*Counted on the briefs where repair ran: a seeded error 'leaks' only if it is still in the claim and the "
              f"claim is shown as verified without the label that flags it. Leaks: "
              f"{summary.get('errors_leaked_after_repair')} of {summary.get('seeded_errors_in_repaired_briefs')}; "
              f"the rest were corrected, removed or marked unverified by repair.",
              f"Per brief (original run): {summary['avg_brief_credits']} Tavily credits, "
              f"{summary['avg_brief_input_tokens']} input / {summary['avg_brief_output_tokens']} output tokens, "
              f"{summary['avg_brief_seconds']} s."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dbs", nargs="*", help="SQLite files with finished briefs (globs allowed)")
    p.add_argument("--rescore", metavar="JSON", help="recompute after-repair outcomes of a saved run (no LLM calls)")
    p.add_argument("--max-briefs", type=int, default=15)
    p.add_argument("--no-repair", action="store_true", help="verify only; skip the repair loop")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-claims", type=int, default=5, help="verified claims sampled per brief")
    p.add_argument("--repair-briefs", type=int, help="run the repair loop on only the first N briefs (slow STRONG "
                   "calls); precision and recall always use every brief")
    p.add_argument("--out-dir", default=str(HERE / "results"))
    args = p.parse_args(argv)
    if args.rescore:
        print(to_markdown(rescore(Path(args.rescore))["summary"]))
        return 0
    paths = sorted({x for pat in args.dbs for x in (glob.glob(pat) or [pat]) if Path(x).is_file()})
    settings = get_settings()
    try:
        llm, _ = default_clients(settings)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"seeded-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    result = asyncio.run(run(paths, max_briefs=args.max_briefs, repair=not args.no_repair, seed=args.seed, llm=llm,
                             settings=settings, out=out, max_claims=args.max_claims,
                             repair_briefs=args.repair_briefs))
    print(to_markdown(result["summary"]))
    print(f"\nSaved {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
