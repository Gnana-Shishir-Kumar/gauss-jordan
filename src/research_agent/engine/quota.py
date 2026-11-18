"""Per-brief call quotas.

Tavily has one pooled budget (default 40 calls: every search and every extract call counts as one), of which a reserve
(default 8) can only be spent by the repair loop's targeted searches. PubMed, ClinicalTrials.gov, openFDA and Crossref
each have their own cap. A refused take() never raises: the caller records why it skipped the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Budget

APIS = ("tavily", "pubmed", "ctgov", "openfda", "crossref")


@dataclass
class Quota:
    budget: Budget
    used: dict[str, int] = field(default_factory=lambda: {a: 0 for a in APIS})
    refused: dict[str, int] = field(default_factory=lambda: {a: 0 for a in APIS})
    repair_used: int = 0  # Tavily calls spent by the repair loop

    def cap(self, api: str) -> int:
        b = self.budget
        return {"tavily": b.tavily_calls, "pubmed": b.pubmed_calls, "ctgov": b.ctgov_calls,
                "openfda": b.openfda_calls, "crossref": b.crossref_calls}[api]

    def research_cap(self) -> int:
        return max(0, self.budget.tavily_calls - self.budget.tavily_repair_reserve)

    def left(self, api: str, phase: str = "research") -> int:
        if api == "tavily" and phase == "research":
            return max(0, self.research_cap() - self.used["tavily"])
        return max(0, self.cap(api) - self.used[api])

    def take(self, api: str, phase: str = "research", n: int = 1) -> bool:
        if self.left(api, phase) < n:
            self.refused[api] += 1
            return False
        self.used[api] += n
        if api == "tavily" and phase == "repair":
            self.repair_used += n
        return True

    def research_exhausted(self) -> bool:
        return self.left("tavily", "research") <= 0

    def snapshot(self) -> dict:
        return {"used": dict(self.used), "caps": {a: self.cap(a) for a in APIS},
                "tavily_repair_reserve": self.budget.tavily_repair_reserve, "tavily_repair_used": self.repair_used,
                "refused": {a: n for a, n in self.refused.items() if n}}
