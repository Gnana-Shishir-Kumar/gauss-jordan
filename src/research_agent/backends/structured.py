"""Structured ground-truth sources behind the SearchBackend interface: PubMed (E-utilities, plus PubMed Central
full text), ClinicalTrials.gov API v2, openFDA (Drugs@FDA approvals and drug labels) and Crossref.

Tavily discovers pages; these APIs provide ground truth for what it found:
  - a PubMed URL is fetched as the PubMed record (title, journal, date, abstract), a PubMed Central URL (or an
    article whose web page could not be extracted) as PMC full text, a ClinicalTrials.gov URL as the registry record
    (status, phase, dates, last update posted, whether results are posted), and an FDA approval page is paired with
    the Drugs@FDA approval history of the drugs it names;
  - search() exists on each backend so any of them can be used through the same interface (e.g. a targeted
    repair search), but discovery stays with Tavily.
Each "page" these backends return is plain text written as sentences, so it goes through the same sentence store,
sentence IDs and verifier gates as any web page. All calls go through retry + rate spacing; quotas are charged by the
engine. No dependencies beyond httpx.
"""

from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

from ..config import Settings
from ..retry import with_retry
from .http import HTTP
from .search_base import ExtractResult, SearchHit, SearchResult

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
OPENFDA = "https://api.fda.gov/drug"
CROSSREF = "https://api.crossref.org/works"

_PUBMED_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})", re.I)
_PMC_URL = re.compile(r"ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)|pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)", re.I)
_CTGOV_URL = re.compile(r"clinicaltrials\.gov/(?:study|ct2/show(?:/record)?)/(NCT\d{8})", re.I)
# International nonproprietary name stems common in oncology (small molecules, antibodies, conjugates, others).
DRUG_RE = re.compile(r"\b([a-z]{3,}(?:tinib|rafenib|ciclib|parib|lisib|zomib|tuzumab|zumab|ximab|umab|mab|vedotin|"
                     r"deruxtecan|govitecan|mertansine|tecan|platin|taxel|rubicin|stat|degib|toclax|sertib|metinib|"
                     r"lukast|leucel|cel)(?:-[a-z]{4})?)\b", re.I)
_STOP = {"tinib", "mab", "stat"}


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _sentence(s: str) -> str:
    s = " ".join((s or "").split()).strip()
    return s if not s or s.endswith((".", "!", "?")) else s + "."


class _Spaced:
    """Rate spacing per service plus the shared retry policy."""

    def __init__(self, http: HTTP, interval: float, attempts: int = 3, base_delay: float = 1.0, max_delay: float = 10.0,
                 retry_after_max: float = 60.0):
        self.http = http
        self.interval = interval
        self.retry = dict(attempts=attempts, base_delay=base_delay, max_delay=max_delay,
                          retry_after_max=retry_after_max)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def call(self, fn):
        async with self._lock:
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
        return await with_retry(fn, **self.retry)


def _retry_args(settings: Settings) -> dict:
    n = settings.network
    return dict(attempts=n.max_retries, base_delay=n.backoff_base_s, max_delay=n.backoff_max_s,
                retry_after_max=n.retry_after_max_s)


# --- PubMed / PubMed Central -----------------------------------------------------------

class PubMedBackend:
    name = "pubmed"
    api = "pubmed"

    def __init__(self, http: HTTP, *, api_key: str = "", email: str = "", **retry: Any):
        self.http = http
        self._key = api_key
        self.email = email
        self.spaced = _Spaced(http, 0.11 if api_key else 0.34, **retry)

    @classmethod
    def from_settings(cls, settings: Settings) -> "PubMedBackend":
        return cls(HTTP(timeout_s=settings.network.id_timeout_s), api_key=settings.ncbi_api_key,
                   email=settings.contact_email, **_retry_args(settings))

    def _params(self, **p: Any) -> dict:
        p["tool"] = "oncology-update-briefs"
        if self._key:
            p["api_key"] = self._key
        if self.email:
            p["email"] = self.email
        return p

    def handles(self, url: str) -> bool:
        return bool(_PUBMED_URL.search(url or "") or _PMC_URL.search(url or ""))

    async def search(self, query: str, *, depth: str = "basic", max_results: int = 5,
                     include_domains: list[str] | None = None, start_date: str | None = None) -> SearchResult:
        params = self._params(db="pubmed", term=query, retmode="json", retmax=max_results, sort="relevance")
        if start_date:
            params.update(datetype="pdat", mindate=start_date.replace("-", "/"), maxdate="3000")
        data = await self.spaced.call(lambda: self.http.get_json(f"{EUTILS}/esearch.fcgi", params, "PubMed esearch"))
        ids = ((data or {}).get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return SearchResult()
        summ = await self.spaced.call(lambda: self.http.get_json(
            f"{EUTILS}/esummary.fcgi", self._params(db="pubmed", id=",".join(ids), retmode="json"), "PubMed esummary"))
        res = (summ or {}).get("result") or {}
        hits = [SearchHit(url=f"https://pubmed.ncbi.nlm.nih.gov/{i}/", title=(res.get(i) or {}).get("title") or "",
                          content=(res.get(i) or {}).get("title") or "", score=0.9 - 0.05 * n,
                          published_date=(res.get(i) or {}).get("sortpubdate", "")[:10].replace("/", "-") or None)
                for n, i in enumerate(ids) if res.get(i)]
        return SearchResult(hits=hits)

    async def extract(self, urls: list[str], *, depth: str = "basic") -> ExtractResult:
        out = ExtractResult()
        for u in urls:
            try:
                text = None
                if m := _PMC_URL.search(u):
                    text = await self.pmc_text(m.group(1) or m.group(2))
                elif m := _PUBMED_URL.search(u):
                    text = await self.record_text(m.group(1))
                if text:
                    out.pages[u] = text
                else:
                    out.failed[u] = "no PubMed record"
            except Exception as exc:  # noqa: BLE001 - per-URL failure is reported, not raised
                out.failed[u] = f"{type(exc).__name__}: {str(exc)[:150]}"
        return out

    async def record_text(self, pmid: str) -> str | None:
        """Title, journal, publication date, identifiers and the abstract, as sentences. Uses PMC full text instead
        when the article is open access in PubMed Central."""
        xml = await self.spaced.call(lambda: self.http.get_text(
            f"{EUTILS}/efetch.fcgi", self._params(db="pubmed", id=pmid, retmode="xml"), "PubMed efetch"))
        if not xml:
            return None
        root = ET.fromstring(xml)
        art = root.find(".//PubmedArticle")
        if art is None:
            return None
        title = _text(art.find(".//ArticleTitle"))
        journal = _text(art.find(".//Journal/Title"))
        date = _pubmed_xml_date(art)
        types = [_text(t) for t in art.findall(".//PublicationType")]
        ids = {i.get("IdType"): _text(i) for i in art.findall(".//ArticleIdList/ArticleId")}
        lines = [_sentence(title), f"Published in {journal} on {date or 'an unstated date'} (PMID {pmid}"
                 + (f"; doi:{ids['doi']}" if ids.get("doi") else "") + ")."]
        if types:
            lines.append(f"Publication type: {', '.join(types)}.")
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            body = _text(ab)
            if body:
                lines.append(f"{label.title()}: {body}" if label else body)
        if ids.get("pmc"):
            full = await self.pmc_text(ids["pmc"])
            if full:
                return "\n".join(lines) + "\n" + full
        return "\n".join(lines)

    async def pmc_text(self, pmcid: str) -> str | None:
        """Open-access full text from PubMed Central (article body paragraphs; figures and tables excluded)."""
        xml = await self.spaced.call(lambda: self.http.get_text(
            f"{EUTILS}/efetch.fcgi", self._params(db="pmc", id=pmcid.removeprefix("PMC"), retmode="xml"),
            "PMC efetch"))
        if not xml:
            return None
        root = ET.fromstring(xml)
        body = root.find(".//body")
        if body is None:
            return None
        title = _text(root.find(".//article-title"))
        paras = [_text(p) for p in body.iter("p") if _text(p)]
        return "\n".join([_sentence(title), f"Full text from PubMed Central {pmcid}."] + paras) if paras else None

    async def pmc_for(self, ident_kind: str, value: str) -> str | None:
        """PMCID for a PMID or DOI, if the article is in PubMed Central."""
        term = f"{value}[doi]" if ident_kind == "doi" else f"{value}[pmid]"
        data = await self.spaced.call(lambda: self.http.get_json(
            f"{EUTILS}/esearch.fcgi", self._params(db="pmc", term=term, retmode="json"), "PMC esearch"))
        ids = ((data or {}).get("esearchresult") or {}).get("idlist") or []
        return f"PMC{ids[0]}" if ids else None


def _pubmed_xml_date(art: ET.Element) -> str | None:
    months = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                          "Nov", "Dec"], 1)}
    for path in (".//ArticleDate", ".//JournalIssue/PubDate"):
        d = art.find(path)
        if d is None or d.find("Year") is None:
            continue
        y = _text(d.find("Year"))
        mo = _text(d.find("Month"))
        day = _text(d.find("Day"))
        mo_n = int(mo) if mo.isdigit() else months.get(mo[:3])
        if mo_n and day.isdigit():
            return f"{y}-{mo_n:02d}-{int(day):02d}"
        return f"{y}-{mo_n:02d}" if mo_n else y
    return None


# --- ClinicalTrials.gov -------------------------------------------------------------

class CTGovBackend:
    name = "ctgov"
    api = "ctgov"

    def __init__(self, http: HTTP, **retry: Any):
        self.http = http
        self.spaced = _Spaced(http, 0.2, **retry)

    @classmethod
    def from_settings(cls, settings: Settings) -> "CTGovBackend":
        return cls(HTTP(timeout_s=settings.network.id_timeout_s), **_retry_args(settings))

    def handles(self, url: str) -> bool:
        return bool(_CTGOV_URL.search(url or ""))

    async def search(self, query: str, *, depth: str = "basic", max_results: int = 5,
                     include_domains: list[str] | None = None, start_date: str | None = None) -> SearchResult:
        params: dict[str, Any] = {"query.term": query, "pageSize": max_results, "format": "json",
                                  "fields": "NCTId,BriefTitle,LastUpdatePostDate"}
        if start_date:
            params["filter.advanced"] = f"AREA[LastUpdatePostDate]RANGE[{start_date},MAX]"
        data = await self.spaced.call(lambda: self.http.get_json(CTGOV, params, "ClinicalTrials.gov search"))
        hits = []
        for n, st in enumerate((data or {}).get("studies") or []):
            ps = st.get("protocolSection") or {}
            nct = (ps.get("identificationModule") or {}).get("nctId")
            if nct:
                hits.append(SearchHit(url=f"https://clinicaltrials.gov/study/{nct}",
                                      title=(ps.get("identificationModule") or {}).get("briefTitle") or nct,
                                      content="", score=0.9 - 0.05 * n,
                                      published_date=((ps.get("statusModule") or {}).get("lastUpdatePostDateStruct")
                                                      or {}).get("date")))
        return SearchResult(hits=hits)

    async def extract(self, urls: list[str], *, depth: str = "basic") -> ExtractResult:
        out = ExtractResult()
        for u in urls:
            m = _CTGOV_URL.search(u or "")
            if not m:
                out.failed[u] = "not a ClinicalTrials.gov study URL"
                continue
            try:
                data = await self.spaced.call(lambda: self.http.get_json(
                    f"{CTGOV}/{m.group(1).upper()}", {"format": "json"}, "ClinicalTrials.gov"))
                text = study_text(data) if data else None
                if text:
                    out.pages[u] = text
                else:
                    out.failed[u] = "no ClinicalTrials.gov record"
            except Exception as exc:  # noqa: BLE001
                out.failed[u] = f"{type(exc).__name__}: {str(exc)[:150]}"
        return out


def study_text(data: dict) -> str | None:
    """A ClinicalTrials.gov v2 record written as checkable sentences (status, phase, dates, outcomes, results)."""
    ps = data.get("protocolSection") or {}
    idm, stm, dm = ps.get("identificationModule") or {}, ps.get("statusModule") or {}, ps.get("designModule") or {}
    if not idm.get("nctId"):
        return None
    nct = idm["nctId"]

    def d(key: str) -> str | None:
        return (stm.get(key) or {}).get("date")

    lines = [_sentence(f"{idm.get('officialTitle') or idm.get('briefTitle')} ({nct})")]
    if idm.get("briefTitle") and idm.get("officialTitle"):
        lines.append(_sentence(f"Brief title: {idm['briefTitle']}"))
    phases = ", ".join(p.replace("PHASE", "phase ") for p in dm.get("phases") or []) or "not stated"
    lines.append(f"{nct} overall status is {str(stm.get('overallStatus') or 'unknown').replace('_', ' ').lower()}; "
                 f"phase: {phases}.")
    if stm.get("whyStopped"):
        lines.append(_sentence(f"{nct} why stopped: {stm['whyStopped']}"))
    for key, label in (("startDateStruct", "Start date"), ("primaryCompletionDateStruct", "Primary completion date"),
                       ("completionDateStruct", "Completion date"), ("studyFirstPostDateStruct", "First posted"),
                       ("resultsFirstPostDateStruct", "Results first posted"),
                       ("lastUpdatePostDateStruct", "Last update posted")):
        if d(key):
            lines.append(f"{nct} {label.lower()}: {d(key)}.")
    lines.append(f"{nct} has results posted: {'yes' if data.get('hasResults') else 'no'}.")
    enroll = (dm.get("enrollmentInfo") or {}).get("count")
    if enroll:
        lines.append(f"{nct} enrollment: {enroll} participants.")
    conds = ", ".join((ps.get("conditionsModule") or {}).get("conditions") or [])
    if conds:
        lines.append(_sentence(f"{nct} conditions: {conds}"))
    arms = (ps.get("armsInterventionsModule") or {}).get("interventions") or []
    if arms:
        lines.append(_sentence(f"{nct} interventions: " + "; ".join(i.get("name", "") for i in arms[:6])))
    for o in ((ps.get("outcomesModule") or {}).get("primaryOutcomes") or [])[:3]:
        lines.append(_sentence(f"{nct} primary outcome: {o.get('measure')} (time frame: {o.get('timeFrame')})"))
    summary = (ps.get("descriptionModule") or {}).get("briefSummary")
    if summary:
        lines.append(" ".join(summary.split()))
    return "\n".join(lines)


# --- openFDA (Drugs@FDA approvals, drug labels) ---------------------------------------

class OpenFDABackend:
    name = "openfda"
    api = "openfda"

    def __init__(self, http: HTTP, **retry: Any):
        self.http = http
        self.spaced = _Spaced(http, 0.3, **retry)

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenFDABackend":
        return cls(HTTP(timeout_s=settings.network.id_timeout_s), **_retry_args(settings))

    def handles(self, url: str) -> bool:
        return (url or "").startswith(f"{OPENFDA}/drugsfda.json")

    @staticmethod
    def approvals_url(generic: str) -> str:
        return f'{OPENFDA}/drugsfda.json?search=openfda.generic_name:"{quote(generic.lower())}"&limit=5'

    async def search(self, query: str, *, depth: str = "basic", max_results: int = 3,
                     include_domains: list[str] | None = None, start_date: str | None = None) -> SearchResult:
        """Drug names in the query -> their Drugs@FDA approval records."""
        names = drug_names(query)[:max_results]
        return SearchResult(hits=[SearchHit(url=self.approvals_url(n), title=f"Drugs@FDA approvals: {n}",
                                            content=n, score=0.9) for n in names])

    async def extract(self, urls: list[str], *, depth: str = "basic") -> ExtractResult:
        out = ExtractResult()
        for u in urls:
            m = re.search(r'openfda\.generic_name:"([^"]+)"', u)
            if not m:
                out.failed[u] = "not an openFDA Drugs@FDA URL"
                continue
            try:
                text = await self.approvals_text(m.group(1))
                if text:
                    out.pages[u] = text
                else:
                    out.failed[u] = "no Drugs@FDA record"
            except Exception as exc:  # noqa: BLE001
                out.failed[u] = f"{type(exc).__name__}: {str(exc)[:150]}"
        return out

    async def approvals_text(self, generic: str) -> str | None:
        data = await self.spaced.call(lambda: self.http.get_json(
            f"{OPENFDA}/drugsfda.json", {"search": f'openfda.generic_name:"{generic}"', "limit": 5},
            "openFDA drugsfda"))
        results = (data or {}).get("results") or []
        if not results:
            return None
        lines = [f"Drugs@FDA (openFDA) approval history for {generic}."]
        for r in results:
            brand = ", ".join((r.get("openfda") or {}).get("brand_name") or []) or r.get("sponsor_name") or "?"
            app = r.get("application_number", "")
            subs = sorted(r.get("submissions") or [], key=lambda s: s.get("submission_status_date") or "",
                          reverse=True)
            for s in subs[:8]:
                if s.get("submission_status") != "AP":
                    continue
                dt = s.get("submission_status_date") or ""
                date = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}" if len(dt) == 8 else dt
                kind = {"ORIG": "original approval", "SUPPL": "supplement"}.get(s.get("submission_type"), "submission")
                cls = (s.get("submission_class_code_description") or s.get("submission_class_code") or "").lower()
                lines.append(f"{brand} ({app}) {kind} {s.get('submission_type', '')}-{s.get('submission_number', '')}"
                             f"{' (' + cls + ')' if cls else ''} was approved on {date}.")
        return "\n".join(lines) if len(lines) > 1 else None


def drug_names(text: str) -> list[str]:
    out: list[str] = []
    for m in DRUG_RE.finditer(text or ""):
        n = m.group(1).lower()
        n = re.sub(r"-[a-z]{4}$", "", n)  # biologic suffix (amivantamab-vmjw -> amivantamab)
        if n not in _STOP and n not in out:
            out.append(n)
    return out


# --- Crossref ----------------------------------------------------------------------------

class CrossrefBackend:
    """Bibliographic search over Crossref; DOI resolution itself lives in identifiers.IdentifierResolver."""
    name = "crossref"
    api = "crossref"

    def __init__(self, http: HTTP, *, email: str = "", **retry: Any):
        self.http = http
        self.email = email
        self.spaced = _Spaced(http, 0.2, **retry)

    @classmethod
    def from_settings(cls, settings: Settings) -> "CrossrefBackend":
        return cls(HTTP(timeout_s=settings.network.id_timeout_s), email=settings.contact_email,
                   **_retry_args(settings))

    def handles(self, url: str) -> bool:
        return False

    async def search(self, query: str, *, depth: str = "basic", max_results: int = 5,
                     include_domains: list[str] | None = None, start_date: str | None = None) -> SearchResult:
        params: dict[str, Any] = {"query.bibliographic": query, "rows": max_results}
        if start_date:
            params["filter"] = f"from-pub-date:{start_date}"
        if self.email:
            params["mailto"] = self.email
        data = await self.spaced.call(lambda: self.http.get_json(CROSSREF, params, "Crossref search"))
        hits = []
        for n, it in enumerate(((data or {}).get("message") or {}).get("items") or []):
            if it.get("DOI"):
                hits.append(SearchHit(url=f"https://doi.org/{it['DOI']}", title=(it.get("title") or [""])[0],
                                      content="", score=0.9 - 0.05 * n))
        return SearchResult(hits=hits)

    async def extract(self, urls: list[str], *, depth: str = "basic") -> ExtractResult:
        return ExtractResult(failed={u: "Crossref has metadata only, no page text" for u in urls})


class Structured:
    """The structured backends a run can use, by API name (for quota accounting)."""

    def __init__(self, pubmed: PubMedBackend, ctgov: CTGovBackend, openfda: OpenFDABackend, crossref: CrossrefBackend):
        self.pubmed, self.ctgov, self.openfda, self.crossref = pubmed, ctgov, openfda, crossref

    @classmethod
    def from_settings(cls, settings: Settings) -> "Structured":
        return cls(PubMedBackend.from_settings(settings), CTGovBackend.from_settings(settings),
                   OpenFDABackend.from_settings(settings), CrossrefBackend.from_settings(settings))

    def for_url(self, url: str):
        for b in (self.pubmed, self.ctgov, self.openfda):
            if b.handles(url):
                return b
        return None
