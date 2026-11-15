"""Export the sources used in a brief as CSV or RIS (Zotero / EndNote). Standard library only."""

from __future__ import annotations

import csv
import io

from .sources import tier_label, type_label

_RIS_TYPE = {
    "peer_reviewed_article": "JOUR",
    "conference_abstract": "ABST",
    "preprint": "UNPB",
    "guideline": "GOVDOC",
    "regulatory_action": "GOVDOC",
    "registry_entry": "DBASE",
    "press_release": "PRESS",
    "news": "NEWS",
}
FIELDS = ["evidence_ids", "title", "url", "domain", "source_type", "date", "trust_tier", "tier_label", "verdicts"]


def sources_used(report: dict) -> list[dict]:
    """Sources cited by claims (one row per URL); falls back to kept evidence if nothing was cited."""
    rows: dict[str, dict] = {}
    for c in report.get("claims", []):
        for x in c.get("citations", []):
            if not x.get("url"):
                continue
            r = rows.setdefault(x["url"], {"evidence_ids": [], "title": x.get("title") or "", "url": x["url"],
                                           "domain": x.get("domain") or "", "source_type": x.get("source_type") or "",
                                           "date": x.get("evidence_date") or "", "trust_tier": x.get("trust_tier"),
                                           "verdicts": []})
            if x["evidence_id"] not in r["evidence_ids"]:
                r["evidence_ids"].append(x["evidence_id"])
            if x.get("verdict"):
                r["verdicts"].append(f"E{x['evidence_id']}:{x['verdict']}")
    if not rows:
        for e in report.get("evidence", []):
            if e.get("kept"):
                rows.setdefault(e["url"], {"evidence_ids": [e["id"]], "title": e.get("title") or "", "url": e["url"],
                                           "domain": e.get("domain") or "", "source_type": e.get("source_type") or "",
                                           "date": e.get("evidence_date") or "", "trust_tier": e.get("trust_tier"),
                                           "verdicts": []})
    return list(rows.values())


def to_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for r in sources_used(report):
        w.writerow({**r, "evidence_ids": " ".join(f"E{i}" for i in r["evidence_ids"]),
                    "tier_label": tier_label(r["trust_tier"]), "verdicts": " ".join(r["verdicts"])})
    return buf.getvalue()


def _ris_line(tag: str, value: object) -> str:
    return f"{tag}  - {' '.join(str(value).split())}"


def to_ris(report: dict) -> str:
    task = report.get("task") or {}
    out: list[str] = []
    for r in sources_used(report):
        out.append(_ris_line("TY", _RIS_TYPE.get(r["source_type"], "ELEC")))
        out.append(_ris_line("TI", r["title"] or r["url"]))
        out.append(_ris_line("UR", r["url"]))
        if r["date"]:
            parts = r["date"].split("-")
            out.append(_ris_line("PY", parts[0]))
            out.append(_ris_line("DA", "/".join(parts + [""] * (3 - len(parts))) + "/"))
        out.append(_ris_line("DB", r["domain"]))
        out.append(_ris_line("KW", type_label(r["source_type"])))
        out.append(_ris_line("KW", tier_label(r["trust_tier"])))
        out.append(_ris_line("N1", f"Source type: {r['source_type'] or 'unclassified'}; trust tier: "
                             f"{r['trust_tier'] or 'n/a'}; cited as {' '.join(f'E{i}' for i in r['evidence_ids'])}; "
                             f"research task {task.get('id', '')}"))
        out.append("ER  - ")
        out.append("")
    return "\n".join(out)
