"""Credit comparison: what a stored run cost vs. what "traditional" search strategies would have cost.

Uses the run's own queries and search results (from the trace), so no new API calls are made.
The project row is the credits Tavily actually billed. The traditional rows price the same queries and
results at Tavily's published rates:
  search: basic 1 credit, advanced 2 credits
  extract (basic): 1 credit per 5 successful URLs, rounded up per call
"""

from __future__ import annotations

import math

SEARCH_COST = {"basic": 1, "fast": 1, "ultra-fast": 1, "advanced": 2}
URLS_PER_EXTRACT_CREDIT = 5


def extract_cost(n_urls: int, batch_size: int) -> tuple[int, int]:
    """(calls, credits) to extract n_urls in batches of batch_size, rounding up per call."""
    if n_urls <= 0:
        return 0, 0
    calls = math.ceil(n_urls / batch_size)
    sizes = [batch_size] * (n_urls // batch_size) + ([n_urls % batch_size] if n_urls % batch_size else [])
    return calls, sum(math.ceil(s / URLS_PER_EXTRACT_CREDIT) for s in sizes)


def run_stats(report: dict) -> dict:
    searches = [t for t in report["trace"] if t["step"] == "search" and "results" in t["detail"]]
    depths = [t["detail"].get("depth", "basic") for t in searches]
    urls = [r["url"] for t in searches for r in t["detail"].get("results", [])]
    extracts = [t for t in report["trace"] if t["step"] == "extract" and t["detail"].get("urls")]
    return {
        "searches": len(searches),
        "depths": depths,
        "raw_results": len(urls),
        "unique_results": len(set(urls)),
        "extracted_urls": sum(t["detail"]["urls"] for t in extracts),
        "extract_calls": len(extracts),
        "actual_search_credits": sum(t.get("tavily_credits") or 0 for t in report["trace"] if t["step"] == "search"),
        "actual_extract_credits": sum(t.get("tavily_credits") or 0 for t in report["trace"] if t["step"] == "extract"),
    }


def compare(report: dict, batch_size: int = 20) -> dict:
    s = run_stats(report)
    same_depth = sum(SEARCH_COST.get(d, 1) for d in s["depths"])
    all_advanced = SEARCH_COST["advanced"] * s["searches"]
    n = s["unique_results"]
    batched_calls, batched = extract_cost(n, batch_size)
    rows = [
        {"strategy": "This project (billed)",
         "how": "snippet scoring, top results only, batched extract, tiered depth",
         "search_credits": s["actual_search_credits"], "extract_calls": s["extract_calls"],
         "pages": s["extracted_urls"], "extract_credits": s["actual_extract_credits"]},
        {"strategy": "Fetch every result, batched",
         "how": "same searches; extract all results 20 per call",
         "search_credits": same_depth, "extract_calls": batched_calls, "pages": n, "extract_credits": batched},
        {"strategy": "Fetch every result, one call per page",
         "how": "same searches; extract each result in its own call",
         "search_credits": same_depth, "extract_calls": n, "pages": n, "extract_credits": n},
        {"strategy": "Advanced search + fetch every page",
         "how": "every search at advanced depth; one extract call per page",
         "search_credits": all_advanced, "extract_calls": n, "pages": n, "extract_credits": n},
    ]
    for r in rows:
        r["total"] = r["search_credits"] + r["extract_credits"]
    base = rows[0]["total"]
    for r in rows:
        r["vs_project"] = round(r["total"] / base, 1) if base else None
    return {"stats": s, "rows": rows,
            "repeat_within_cache": {"project": 0, "traditional": [r["total"] for r in rows[1:]]}}


def to_markdown(result: dict) -> str:
    s = result["stats"]
    lines = [
        f"Same run: {s['searches']} searches returned {s['raw_results']} results ({s['unique_results']} unique URLs); "
        f"the project extracted {s['extracted_urls']} of them.",
        "",
        "| Strategy | How | Search credits | Extract calls | Pages extracted | Extract credits | **Total** | vs project |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        lines.append(f"| {r['strategy']} | {r['how']} | {r['search_credits']:g} | {r['extract_calls']} | {r['pages']} | "
                     f"{r['extract_credits']:g} | **{r['total']:g}** | {r['vs_project']}x |")
    rep = result["repeat_within_cache"]
    lines += ["", f"Asking the same question again within 7 days: project **0** credits (cache) vs "
                  f"{', '.join(f'{t:g}' for t in rep['traditional'])} credits for the traditional strategies above."]
    return "\n".join(lines)
