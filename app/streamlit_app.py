"""Streamlit UI. Run: uv run streamlit run app/streamlit_app.py

UI only. Research runs in a background thread with its own event loop and DB connection; the page polls
SQLite for progress, so a page refresh does not interrupt a run.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import threading
import time

import streamlit as st

from research_agent import storage
from research_agent.config import Settings, get_settings
from research_agent.engine import prepare, run
from research_agent.engine.checks import before_anchor, number_tokens, numbers_in
from research_agent.engine.orchestrator import default_clients, load_report
from research_agent.exports import sources_used, to_csv, to_ris
from research_agent.privacy import IdentifierError, find_identifiers
from research_agent.prompts import PROMPT_VERSION
from research_agent.render import (DISCLAIMER, NO_IDENTIFIERS, USE_LIMITS, claim_verdict, evidence_label,
                                   is_flagged, key_numbers, sort_items)
from research_agent.sources import is_preliminary, tier_label, type_label

VERDICT = {"supported": ("green", ":material/check_circle:"), "partial": ("orange", ":material/rule:"),
           "unsupported": ("red", ":material/cancel:"), "source_unavailable": ("gray", ":material/link_off:"),
           "uncited": ("gray", ":material/help:"), "unverified": ("gray", ":material/help:")}
STATUS = {"done": "green", "insufficient_evidence": "orange", "failed": "red", "needs_clarification": "blue"}
CONF = {"high": "green", "moderate": "orange", "low": "red"}
TIER_COLOR = {1: "blue", 2: "violet", 3: "gray"}
RUNNING = {"researching", "writing", "verifying"}
LEVEL_ICON = {"info": ":material/check:", "warning": ":material/warning:", "error": ":material/error:"}


def md(text) -> str:
    return str(text or "").replace("$", "\\$")


def password_gate() -> bool:
    """If APP_PASSWORD is set (required in the container), ask for it once per browser session."""
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return True
    st.title("Oncology Update Briefs")
    st.caption(USE_LIMITS)
    with st.form("login"):
        given = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(given.encode(), expected.encode()):
                st.session_state.authed = True
                st.rerun()
            time.sleep(1)  # slow down guessing
            st.error("Wrong password.")
    return False


@st.cache_resource
def get_conn(path: str):
    return storage.connect(path)


def _run_in_background(task_id: int, settings: Settings) -> None:
    def target():
        conn = storage.connect(settings.db_path)
        try:
            llm, search = default_clients(settings)
            asyncio.run(run(task_id, llm=llm, search=search, conn=conn, settings=settings))
        except ValueError:
            pass  # already started elsewhere (double click / second tab): leave that run alone
        except Exception as exc:  # recorded so the polling UI can show it
            storage.finish_task(conn, task_id, "failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        finally:
            conn.close()
    threading.Thread(target=target, daemon=True, name=f"research-{task_id}").start()


# --- views -----------------------------------------------------------------------

def show_brief(report: dict) -> None:
    t, b = report["task"], report["task"].get("brief") or {}
    st.markdown(f"**Cancer type:** {md(b.get('cancer_type'))}  \n**Setting:** {md(b.get('setting'))}  \n"
                f"**Question:** {md(b.get('question'))}  \n**Since:** {t.get('anchor_date')}"
                + (" *(no date given: defaulted to 12 months ago)*" if b.get("anchor_defaulted") else ""))
    st.markdown("**A complete answer must cover:** " + "; ".join(md(a) for a in b.get("aspects", [])))


def show_progress_log(report: dict) -> None:
    for t in report["trace"][-14:]:
        st.markdown(f"{LEVEL_ICON.get(t['level'], '')} **{t['step']}** · {md(t['message'])}")


@st.fragment(run_every="1s")
def live_progress(task_id: int) -> None:
    conn = get_conn(get_settings().db_path)
    report = load_report(conn, task_id)
    status = report["task"]["status"]
    if status not in RUNNING and status != "awaiting_approval":
        st.rerun()
    totals = report["totals"]
    with st.status(f"{status.capitalize()}… · {totals['credits']:g} Tavily credits so far", expanded=True):
        show_progress_log(report)


def item_card(c: dict, anchor: str | None = None) -> None:
    v = claim_verdict(c)
    color, icon = VERDICT.get(v, ("gray", ""))
    cits = c["citations"]
    tiers = sorted({x["trust_tier"] for x in cits if x.get("trust_tier")})
    types = sorted({type_label(x.get("source_type")) for x in cits})
    badges = [f":{color}-badge[{icon} {v.replace('_', ' ')}]", f":gray-badge[{c['claim_type']}]"]
    badges += [f":{TIER_COLOR[t]}-badge[{tier_label(t).split(' · ')[0]}]" for t in tiers]
    if any(is_preliminary(x.get("source_type")) for x in cits):
        badges.append(":orange-badge[:material/science: PRELIMINARY]")
    if is_flagged(c):
        badges.append(":red-badge[FLAGGED: not verified]")
    if before_anchor(c.get("item_date"), anchor):
        badges.append(f":red-badge[DATED BEFORE {anchor}: not new]")
    with st.container(border=True):
        st.markdown(" ".join(badges))
        st.markdown(f"**{c.get('item_date') or 'date not stated'} · {', '.join(types)}**  \n{md(c['text'])}")
        st.caption(f"Evidence level: {md(c.get('evidence_level') or 'not stated')} · {md(evidence_label(c))}  \n"
                   f"Population: {md(c.get('population') or 'population not stated')}")
        nums = key_numbers(c)
        if nums:
            st.markdown("**Key numbers** (each beside its verbatim source sentence)")
            claim_nums = number_tokens(c["text"])
            for sentence, eid in nums:
                in_sentence = numbers_in(sentence)
                found = ", ".join(dict.fromkeys(raw for raw, norm in claim_nums if norm in in_sentence)) or "-"
                left, right = st.columns([1, 5])
                left.markdown(f"`{found}`")
                right.markdown(f"> {md(sentence)}  \n:gray[E{eid}]")
        for x in cits:
            vc, _ = VERDICT.get(x.get("verdict") or "unverified", ("gray", ""))
            title = md(x.get("title") or x.get("domain") or x.get("url"))
            st.markdown(f"E{x['evidence_id']} · [{title}]({x.get('url')}) · {md(type_label(x.get('source_type')))} · "
                        f":{vc}[{(x.get('verdict') or 'unverified').replace('_', ' ')}]")
            if x.get("verifier_note"):
                st.caption(md(x["verifier_note"]))


def show_report(report: dict) -> None:
    t = report["task"]
    ans = t.get("answer_json") or {}
    st.subheader(md(t["question"]))
    st.badge(t["status"].replace("_", " "), color=STATUS.get(t["status"], "gray"))
    if t.get("error"):
        st.error(t["error"])
    totals = report["totals"]
    a, b, c = st.columns(3)
    a.metric("Tavily credits", f"{totals['credits']:g}")
    b.metric("Input tokens", totals["input_tokens"])
    c.metric("Output tokens", totals["output_tokens"])
    st.caption(f"Research stopped: **{(t.get('stop_reason') or '-').replace('_', ' ')}**")
    with st.expander("Brief", icon=":material/description:"):
        show_brief(report)
    if t["status"] == "failed":
        st.caption("The run failed before an answer was produced; nothing below is a finding.")
        show_trace(report)
        return

    if (t.get("brief") or {}).get("anchor_defaulted"):
        st.warning(f"No date was given, so this covers the 12 months since {t.get('anchor_date')}.")
    if ans.get("insufficiency_note"):
        st.warning(f"Insufficient evidence: {md(ans['insufficiency_note'])}")

    st.markdown("#### 1. Bottom line")
    bl = ans.get("bottom_line")
    if bl:
        st.markdown(f":{CONF.get(bl['confidence'], 'gray')}-badge[confidence: {bl['confidence']}] {md(bl['text'])}")
        st.caption(md(bl.get("reason")))
    else:
        st.markdown("No bottom line: the evidence found was insufficient.")

    claims = report["claims"]
    st.markdown(f"#### 2. What's new since {t.get('anchor_date')}")
    new = sort_items([c for c in claims if c["section"] == "new"])
    if not new:
        st.markdown("No verified developments found.")
    for c in new:
        item_card(c, t.get("anchor_date"))
    st.markdown("#### 3. Sources disagree")
    conflicts = [c for c in claims if c["section"] == "conflict"]
    for c in conflicts:
        item_card(c, t.get("anchor_date"))
    if not conflicts:
        st.markdown("No conflicts found.")
    st.markdown("#### 4. Not yet established")
    st.markdown("\n".join(f"- {md(x)}" for x in ans.get("not_established") or ["(none listed)"]))
    st.markdown("#### 5. Check before acting")
    st.markdown("\n".join(f"- {md(x)}" for x in ans.get("check_before_acting") or ["(none listed)"]))
    st.markdown("#### 6. Clinician-support statement")
    st.info(DISCLAIMER + "  \n" + USE_LIMITS)

    used = sources_used(report)
    if used:
        st.markdown("#### Sources")
        for s in used:
            st.markdown(f"- [{md(s['title'] or s['url'])}]({s['url']}) :gray[{s['domain']} · "
                        f"{md(type_label(s['source_type']))} · {s['date'] or 'undated'} · {tier_label(s['trust_tier'])}]")
        x, y = st.columns(2)
        x.download_button("Download sources (CSV)", to_csv(report), f"task{t['id']}_sources.csv", "text/csv",
                          icon=":material/table:")
        y.download_button("Download sources (RIS for Zotero/EndNote)", to_ris(report), f"task{t['id']}_sources.ris",
                          "application/x-research-info-systems", icon=":material/menu_book:")
    show_trace(report)


def show_trace(report: dict) -> None:
    t = report["task"]
    with st.expander("Trace", icon=":material/timeline:"):
        st.caption(f"Models: {t.get('model_ids')} · prompt version: {t.get('prompt_version')}")
        st.dataframe([{"step": e["step"], "level": e["level"], "message": e["message"], "model": e["model_id"],
                       "prompt_version": e["prompt_version"], "latency_ms": e["latency_ms"],
                       "in_tok": e["input_tokens"], "out_tok": e["output_tokens"], "credits": e["tavily_credits"],
                       "at": e["created_at"]} for e in report["trace"]], use_container_width=True, hide_index=True)
        st.markdown("**Sub-queries**")
        st.dataframe([{k: s[k] for k in ("round", "depth", "text", "status", "result_count")}
                      for s in report["sub_queries"]], use_container_width=True, hide_index=True)
        st.markdown("**Evidence** (dropped items are kept with a reason)")
        st.dataframe([{"id": f"E{e['id']}", "kept": bool(e["kept"]), "drop_reason": e["drop_reason"],
                       "tier": e["trust_tier"], "type": e["source_type"], "date": e["evidence_date"],
                       "relevance": e["relevance_score"], "url": e["url"]} for e in report["evidence"]],
                     use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("url")})


def main() -> None:
    st.set_page_config(page_title="Oncology Update Briefs", page_icon=":material/biotech:", layout="wide")
    settings = get_settings()
    if not password_gate():
        return
    conn = get_conn(settings.db_path)

    with st.sidebar:
        st.markdown("### Oncology Update Briefs")
        st.caption("Nebius Token Factory + Tavily")
        st.markdown(f"Strong: `{settings.models.strong}`  \nFast: `{settings.models.fast}`  \n"
                    f"Prompt version: `{PROMPT_VERSION}`")
        st.caption(USE_LIMITS)
        recent = storage.recent_tasks(conn)
        if recent:
            st.markdown("#### Recent briefs")
            for r in recent:
                if st.button(f"#{r['id']} {r['question'][:48]}", key=f"t{r['id']}", help=r["status"],
                             use_container_width=True):
                    st.session_state.task_id = r["id"]

    st.title("What has changed since…?")
    st.caption("Dated update briefs for oncologists: regulatory actions, guideline updates, trial readouts and "
               "conference data, safety communications and recent papers, each claim cited and checked.")
    st.info(DISCLAIMER + "  \n" + USE_LIMITS, icon=":material/medical_information:")

    with st.form("ask"):
        question = st.text_area("Clinical question", help=NO_IDENTIFIERS,
                                placeholder="e.g. What has changed for first-line EGFR-mutant metastatic NSCLC "
                                            "since the 2024 ESMO guideline update?")
        st.caption(f":material/lock: {NO_IDENTIFIERS}")
        submitted = st.form_submit_button("Prepare brief", type="primary", icon=":material/edit_note:")

    if submitted and question.strip():
        findings = find_identifiers(question)
        if findings:
            st.error("This looks like it contains patient identifiers (" + ", ".join(findings) + "). "
                     "Remove them and ask about the clinical topic only.")
            return
        missing = settings.missing_keys()
        if missing:
            st.error(f"Missing environment variables: {', '.join(missing)}. Copy .env.example to .env.")
            return
        llm, search = default_clients(settings)
        with st.spinner("Writing the research brief and search plan…"):
            try:
                report = asyncio.run(prepare(question, llm=llm, search=search, conn=conn, settings=settings))
            except IdentifierError as exc:
                st.error(str(exc))
                return
        st.session_state.task_id = report["task"]["id"]

    task_id = st.session_state.get("task_id")
    task = storage.get_task(conn, task_id) if task_id else None
    if not task:
        return
    report = load_report(conn, task_id)
    status = task["status"]
    if status == "needs_clarification":
        st.warning(f"**Clarification needed:** {md(task['brief'].get('clarifying_question'))}  \n"
                   "Rephrase the question above with the cancer type and setting (stage, line, biomarker).")
    elif status in RUNNING or (status == "awaiting_approval" and st.session_state.get("started") == task_id):
        live_progress(task_id)
    elif status == "awaiting_approval":
        st.markdown("### Review the plan")
        show_brief(report)
        st.markdown("**Planned searches** (round 1: regulatory, registry, guideline and journal sources only)")
        st.markdown("\n".join(f"{i}. {md(s['text'])}" for i, s in enumerate(report["sub_queries"], 1)))
        st.caption(f"Budget: up to {settings.limits.max_rounds} rounds, {settings.budget.credits:g} Tavily credits, "
                   f"{settings.budget.time_s:g}s.")
        if st.button("Start research", type="primary", icon=":material/play_arrow:"):
            st.session_state.started = task_id
            _run_in_background(task_id, settings)
            st.rerun()
    else:
        show_report(report)


main()
