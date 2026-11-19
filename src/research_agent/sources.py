"""Source types (deterministic rules first) and trust tiers (assigned in code from type + identifier check)."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from .config import BLOCKED_DOMAINS, TIER_DOMAINS, TIER_LABELS

PRELIMINARY_TYPES = {"conference_abstract", "preprint"}
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
_JOURNALS = set(TIER_DOMAINS[2])
_NEWS = {"ascopost.com", "onclive.com", "targetedonc.com", "cancernetwork.com"}


def _base_domain(domain: str, candidates) -> str | None:
    d = domain.lower().removeprefix("www.")
    for c in sorted(candidates, key=len, reverse=True):
        if d == c or d.endswith("." + c):
            return c
    return None


def tier_of(domain: str) -> int | None:
    """Search group of a domain (longest listed match wins); None if unlisted or blocked."""
    d = (domain or "").lower().removeprefix("www.")
    if _base_domain(d, BLOCKED_DOMAINS):
        return None
    best: tuple[int, int] | None = None
    for tier, domains in TIER_DOMAINS.items():
        base = _base_domain(d, domains)
        if base and (best is None or len(base) > best[0]):
            best = (len(base), tier)
    return best[1] if best else None


def domains_for(tiers) -> list[str]:
    return [d for t in sorted(tiers) for d in TIER_DOMAINS[t]]


def tier_label(tier: int | None) -> str:
    return TIER_LABELS.get(tier or 0, "unlisted source")


def classify(url: str, domain: str) -> str | None:
    """Rule-based source type. None means ambiguous: the worker model may classify it."""
    u = url.lower()
    path = urlsplit(u).path
    d = (domain or "").lower().removeprefix("www.")
    base = _base_domain(d, [x for t in TIER_DOMAINS.values() for x in t])
    if base is None:
        return None
    if base in ("fda.gov", "ema.europa.eu"):
        return "regulatory_action"
    if base == "clinicaltrials.gov":
        return "registry_entry"
    if base in TIER_DOMAINS[3]:
        return "conference_abstract"
    if d.startswith("dailynews.") or d.startswith("news."):
        return "news"  # e.g. ASCO Daily News on ascopubs.org is reporting, not a journal article
    if "_suppl" in u or re.search(r"/(abstract|abstracts|meeting|congress)s?/", path):
        if base in ("asco.org", "esmo.org", "ascopubs.org", "aacrjournals.org", "annalsofoncology.org"):
            return "conference_abstract"
    if base in ("esmo.org", "asco.org") and "guideline" in path:
        return "guideline"
    if base == "cancer.gov":
        return "guideline" if ("pdq" in path or "guideline" in path) else "news"
    if base in _WIRES:
        return "press_release"
    if base in _NEWS:
        return "news"
    if base in _JOURNALS:
        return "peer_reviewed_article"
    return None


def assign_tier(source_type: str | None, id_status: str | None) -> int:
    """Trust tier, enforced in code:
      1  regulatory action, registry entry, guideline publication;
      2  peer-reviewed article whose PMID/DOI resolved to the same title (id_status "confirmed");
      3  conference abstract or preprint (always labelled preliminary);
      4  press release, news, and any journal page whose identifier could not be confirmed."""
    if source_type in ("regulatory_action", "registry_entry", "guideline"):
        return 1
    if source_type == "peer_reviewed_article":
        return 2 if id_status == "confirmed" else 4
    if source_type in ("conference_abstract", "preprint"):
        return 3
    return 4


def _type_tier(x) -> tuple[str | None, int | None]:
    if isinstance(x, dict):
        return x.get("source_type"), x.get("trust_tier")
    return x, None


def is_preliminary(x) -> bool:
    """Conference abstracts and preprints (tier 3). Accepts a source/citation dict or a source type."""
    st, tier = _type_tier(x)
    return tier == 3 or st in PRELIMINARY_TYPES


def is_peer_reviewed(x) -> bool:
    """Peer reviewed only when the article's identifier was confirmed (tier 2)."""
    st, tier = _type_tier(x)
    return st == "peer_reviewed_article" and (tier == 2 if isinstance(x, dict) else True)


def is_company_or_news(x) -> bool:
    st, tier = _type_tier(x)
    return tier == 4 if tier is not None else st in ("press_release", "news")


def type_label(source_type: str | None) -> str:
    return TYPE_LABELS.get(source_type or "", "Unclassified source")
