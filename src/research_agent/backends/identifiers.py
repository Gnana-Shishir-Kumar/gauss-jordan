"""Identifier resolution for verifier gate 2: PMIDs (PubMed E-utilities), NCT numbers (ClinicalTrials.gov API v2)
and DOIs (Crossref). Results are cached in SQLite (id_cache); service errors are never cached.

Rate limits: NCBI allows 3 requests/s without an API key and 10/s with one (NCBI_API_KEY); calls to each service are
spaced accordingly. A contact address is sent to NCBI/Crossref only if RESEARCH_CONTACT_EMAIL is set.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import unquote

from .. import storage
from ..config import Settings
from ..evidence import terms
from ..retry import with_retry
from .http import HTTP

KINDS = ("pmid", "nct", "doi")
API_OF = {"pmid": "pubmed", "nct": "ctgov", "doi": "crossref"}

_PMID_RE = re.compile(r"\bPMID:?\s*(\d{5,9})\b", re.I)
_NCT_RE = re.compile(r"\b(NCT\d{8})\b", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>\]\[]+)", re.I)
_URL_PMID = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})", re.I)
_URL_NCT = re.compile(r"clinicaltrials\.gov/(?:study|ct2/show(?:/record)?)/(NCT\d{8})", re.I)
_URL_DOI = re.compile(r"(?:doi\.org/|/doi/(?:full/|abs/|pdf/|epdf/|epub/)?)(10\.\d{4,9}/[^?#\s]+)", re.I)
_URL_NATURE = re.compile(r"nature\.com/articles/([a-z0-9][a-z0-9.\-]+)", re.I)
_TITLE_SUFFIX = re.compile(r"\s*[|\-–—]\s*(pubmed|pmc|nejm|the lancet|jama.*|clinicaltrials\.gov|nature|"
                           r"journal of clinical oncology|annals of oncology|ascopubs|aacr.*|nih)\s*$", re.I)


def _clean_doi(doi: str) -> str:
    return unquote(doi).rstrip(".,;:)").lower()


@dataclass(frozen=True)
class Identifier:
    kind: str  # pmid | nct | doi
    value: str

    def __str__(self) -> str:
        return {"pmid": f"PMID {self.value}", "nct": self.value, "doi": f"doi:{self.value}"}[self.kind]


def ids_in_text(text: str) -> list[Identifier]:
    out: list[Identifier] = []
    for m in _PMID_RE.finditer(text or ""):
        out.append(Identifier("pmid", m.group(1)))
    for m in _NCT_RE.finditer(text or ""):
        out.append(Identifier("nct", m.group(1).upper()))
    for m in _DOI_RE.finditer(text or ""):
        out.append(Identifier("doi", _clean_doi(m.group(1))))
    return list(dict.fromkeys(out))


def ids_in_url(url: str) -> list[Identifier]:
    """The source's own identifier, when its URL carries one."""
    u = url or ""
    if m := _URL_PMID.search(u):
        return [Identifier("pmid", m.group(1))]
    if m := _URL_NCT.search(u):
        return [Identifier("nct", m.group(1).upper())]
    if m := _URL_DOI.search(u):
        return [Identifier("doi", _clean_doi(m.group(1)))]
    if m := _URL_NATURE.search(u):
        return [Identifier("doi", f"10.1038/{m.group(1).lower()}")]
    return []


def norm_title(title: str | None) -> str:
    t = re.sub(r"<[^>]+>", "", title or "")
    prev = None
    while prev != t:
        prev, t = t, _TITLE_SUFFIX.sub("", t)
    return " ".join(t.split())


def title_matches(resolved: str | None, stored: str | None, text_head: str = "") -> bool:
    """The resolved title must match the source we stored: >= 60% of its terms in the stored title, or the resolved
    title appearing in the start of the page text (page titles are often truncated or decorated)."""
    r = norm_title(resolved)
    if not r:
        return False
    rt = terms(r)
    if rt and len(rt & terms(norm_title(stored))) / len(rt) >= 0.6:
        return True
    head = " ".join((text_head or "").lower().split())
    return bool(head) and " ".join(r.lower().split()) in head


@dataclass
class Resolution:
    ident: Identifier
    status: str  # found | not_found | error | skipped
    title: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    cached: bool = False


class Resolver(Protocol):
    async def resolve(self, ident: Identifier, take=None) -> Resolution: ...


class IdentifierResolver:
    def __init__(self, conn, http: HTTP, *, ncbi_api_key: str = "", contact_email: str = "", attempts: int = 3,
                 base_delay: float = 1.0, max_delay: float = 10.0, ttl_days: int = 30):
        self.conn = conn
        self.http = http
        self._ncbi_key = ncbi_api_key
        self.contact_email = contact_email
        self.retry = dict(attempts=attempts, base_delay=base_delay, max_delay=max_delay)
        self.ttl_days = ttl_days
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    @classmethod
    def from_settings(cls, conn, settings: Settings) -> "IdentifierResolver":
        net = settings.network
        return cls(conn, HTTP(timeout_s=net.id_timeout_s), ncbi_api_key=settings.ncbi_api_key,
                   contact_email=settings.contact_email, attempts=net.max_retries, base_delay=net.backoff_base_s,
                   max_delay=net.backoff_max_s)

    def _interval(self, api: str) -> float:
        if api == "pubmed":
            return 0.11 if self._ncbi_key else 0.34
        return 0.2

    async def _spaced(self, api: str, fn):
        lock = self._locks.setdefault(api, asyncio.Lock())
        async with lock:
            wait = self._last.get(api, 0.0) + self._interval(api) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[api] = time.monotonic()
        self.calls[api] = self.calls.get(api, 0) + 1
        return await with_retry(fn, **self.retry)

    async def resolve(self, ident: Identifier, take=None) -> Resolution:
        hit = storage.get_id_cache(self.conn, ident.kind, ident.value, self.ttl_days)
        if hit:
            return Resolution(ident, hit["status"], hit["title"], hit["data"], cached=True)
        api = API_OF[ident.kind]
        if take is not None and not take(api):
            return Resolution(ident, "skipped", detail=f"{api} call quota reached")
        try:
            res = await {"pmid": self._pmid, "nct": self._nct, "doi": self._doi}[ident.kind](ident)
        except Exception as exc:  # noqa: BLE001 - a lookup failure is a gate result, not a crash
            return Resolution(ident, "error", detail=f"{type(exc).__name__}: {str(exc)[:200]}")
        storage.put_id_cache(self.conn, ident.kind, ident.value, res.status, res.title, res.data)
        return res

    async def _pmid(self, ident: Identifier) -> Resolution:
        params = {"db": "pubmed", "id": ident.value, "retmode": "json", "tool": "oncology-update-briefs"}
        if self._ncbi_key:
            params["api_key"] = self._ncbi_key
        if self.contact_email:
            params["email"] = self.contact_email
        data = await self._spaced("pubmed", lambda: self.http.get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params, label="PubMed esummary"))
        rec = ((data or {}).get("result") or {}).get(ident.value) or {}
        if not rec or rec.get("error") or not rec.get("title"):
            return Resolution(ident, "not_found", detail=str(rec.get("error") or "no PubMed record"))
        return Resolution(ident, "found", rec.get("title"),
                          {"pubdate": rec.get("pubdate"), "epubdate": rec.get("epubdate"),
                           "sortpubdate": rec.get("sortpubdate"), "journal": rec.get("fulljournalname"),
                           "pubtype": rec.get("pubtype") or [],
                           "doi": next((a.get("value") for a in rec.get("articleids") or []
                                        if a.get("idtype") == "doi"), None)})

    async def _nct(self, ident: Identifier) -> Resolution:
        data = await self._spaced("ctgov", lambda: self.http.get_json(
            f"https://clinicaltrials.gov/api/v2/studies/{ident.value}", {"format": "json"},
            label="ClinicalTrials.gov"))
        proto = (data or {}).get("protocolSection") or {}
        idm = proto.get("identificationModule") or {}
        if not idm:
            return Resolution(ident, "not_found", detail="no ClinicalTrials.gov record")
        status = proto.get("statusModule") or {}
        design = proto.get("designModule") or {}
        return Resolution(ident, "found", idm.get("officialTitle") or idm.get("briefTitle"),
                          {"brief_title": idm.get("briefTitle"), "official_title": idm.get("officialTitle"),
                           "overall_status": status.get("overallStatus"),
                           "last_update_posted": (status.get("lastUpdatePostDateStruct") or {}).get("date"),
                           "results_first_posted": (status.get("resultsFirstPostDateStruct") or {}).get("date"),
                           "start_date": (status.get("startDateStruct") or {}).get("date"),
                           "primary_completion": (status.get("primaryCompletionDateStruct") or {}).get("date"),
                           "phases": design.get("phases") or [], "has_results": bool(data.get("hasResults"))})

    async def _doi(self, ident: Identifier) -> Resolution:
        params = {"mailto": self.contact_email} if self.contact_email else None
        data = await self._spaced("crossref", lambda: self.http.get_json(
            f"https://api.crossref.org/works/{ident.value}", params, label="Crossref"))
        msg = (data or {}).get("message") or {}
        titles = msg.get("title") or []
        if not titles:
            return Resolution(ident, "not_found", detail="no Crossref record")
        issued = ((msg.get("published") or msg.get("issued") or {}).get("date-parts") or [[None]])[0]
        return Resolution(ident, "found", titles[0],
                          {"container": (msg.get("container-title") or [None])[0], "type": msg.get("type"),
                           "issued": "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(issued) if p)
                           if issued and issued[0] else None})
