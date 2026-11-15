"""Source trust tiers and source-type classification (deterministic rules first)."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from .config import TIER_DOMAINS, TIER_LABELS

PRELIMINARY_TYPES = {"conference_abstract", "preprint", "press_release"}
TYPE_LABELS = {
    "regulatory_action": "Regulatory action",
    "guideline": "Guideline",
    "registry_entry": "Trial registry entry",
    "peer_reviewed_article": "Peer-reviewed article",
    "conference_abstract": "Conference abstract",
    "preprint": "Preprint",
    "press_release": "Press release",
    "news": "News report",
}
_WIRES = {"prnewswire.com", "businesswire.com", "globenewswire.com"}
_JOURNALS = set(TIER_DOMAINS[2]) | {"ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov"}
_NEWS = {"ascopost.com", "onclive.com", "targetedonc.com", "cancernetwork.com"}


def _base_domain(domain: str, candidates) -> str | None:
    d = domain.lower().removeprefix("www.")
    for c in sorted(candidates, key=len, reverse=True):
        if d == c or d.endswith("." + c):
            return c
    return None


def tier_of(domain: str) -> int | None:
    for tier in sorted(TIER_DOMAINS):
        if _base_domain(domain, TIER_DOMAINS[tier]):
            return tier
    return None


def domains_for(tiers) -> list[str]:
    return [d for t in sorted(tiers) for d in TIER_DOMAINS[t]]


def tier_label(tier: int | None) -> str:
    return TIER_LABELS.get(tier or 0, "unlisted source")


def classify(url: str, domain: str) -> str | None:
    """Rule-based source type. None means ambiguous: the worker model may classify it."""
    u = url.lower()
    path = urlsplit(u).path
    base = _base_domain(domain, [d for t in TIER_DOMAINS.values() for d in t])
    if base is None:
        return None
    if base in ("fda.gov", "ema.europa.eu"):
        return "regulatory_action"
    if base == "clinicaltrials.gov":
        return "registry_entry"
    if base == "nccn.org":
        return "guideline"
    if "_suppl" in u or "meetings.asco.org" in u or re.search(r"/(abstract|abstracts|meeting|congress)s?/", path):
        if base in ("asco.org", "esmo.org", "ascopubs.org", "aacrjournals.org", "annalsofoncology.org"):
            return "conference_abstract"
    if base in ("esmo.org", "asco.org", "cancer.gov") and "guideline" in path:
        return "guideline"
    if base in _WIRES:
        return "press_release"
    if base in _NEWS:
        return "news"
    if base in _JOURNALS:
        return "peer_reviewed_article"
    return None


def is_preliminary(source_type: str | None) -> bool:
    return source_type in PRELIMINARY_TYPES


def is_peer_reviewed(source_type: str | None) -> bool:
    return source_type == "peer_reviewed_article"


def type_label(source_type: str | None) -> str:
    return TYPE_LABELS.get(source_type or "", "Unclassified source")
