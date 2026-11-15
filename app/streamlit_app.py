"""Streamlit UI. Run: uv run streamlit run app/streamlit_app.py

UI only: all research logic lives in research_agent.pipeline.
"""

from __future__ import annotations

import asyncio

import streamlit as st

from research_agent import storage
from research_agent.config import get_settings
from research_agent.pipeline import default_clients, load_report, run_research
from research_agent.render import claim_verdict

VERDICT_STYLE = {
    "supported": ("green", ":material/check_circle:"),
    "partial": ("orange", ":material/rule:"),
    "unsupported": ("red", ":material/cancel:"),
    "source_unavailable": ("gray", ":material/link_off:"),
    "uncited": ("gray", ":material/help:"),
    "unverified": ("gray", ":material/help:"),
}
STATUS_STYLE = {"done": "green", "insufficient_evidence": "orange", "failed": "red"}
LEVEL_ICON = {"info": ":material/check:", "warning": ":material/warning:", "error": ":material/error:"}
STEP_LABEL = {
    "start": "Starting", "plan": "Planning", "search": "Searching", "extract": "Extracting",
    "evidence": "Scoring evidence", "follow_up": "Follow-up search", "write": "Writing",
    "verify": "Verifying", "done": "Done", "error": "Failed",
}


def md(text: str) -> str:
    """Escape characters Streamlit markdown would interpret (LaTeX $)."""
    return (text or "").replace("$", "\\$")


@st.cache_resource
def get_conn(path: str):
    return storage.connect(path)


def render_report(report: dict) -> None:
    task = report["task"]
    ev = {e["id"]: e for e in report["evidence"]}

    st.subheader(task["question"])
    st.badge(task["status"].replace("_", " "), color=STATUS_STYLE.get(task["status"], "gray"))
    if task.get("error"):
        st.error(task["error"])
    answer = task.get("final_answer") or ""
    if answer.startswith("Note:"):
        st.warning(md(answer.split("\n", 1)[0][len("Note: "):]))

    if report["claims"]:
        st.markdown("#### Answer")
    for c in report["claims"]:
        verdict = claim_verdict(c)
        with st.container(border=True):
            color, icon = VERDICT_STYLE.get(verdict, ("gray", ""))
            st.markdown(
                f":{color}-badge[{icon} {verdict.replace('_', ' ')}] :gray-badge[{c['claim_type']}]  \n"
                f"**{c['position'] + 1}.** {md(c['text'])}"
            )
            for x in c["citations"]:
                e = ev.get(x["evidence_id"], {})
                title = md(e.get("title") or e.get("domain") or e.get("url", "source"))
                v = x.get("verdict") or "unverified"
                color = VERDICT_STYLE.get(v, ("gray",))[0]
                link = f"[{title}]({e['url']})" if e.get("url") else title
                st.markdown(f"E{x['evidence_id']} · {link} · :{color}[{v.replace('_', ' ')}]")
                if x.get("verifier_note"):
                    st.caption(md(x["verifier_note"]))
                if e.get("excerpt"):
                    with st.expander(f"Excerpt E{x['evidence_id']}"):
                        st.text(e["excerpt"])

    kept = [e for e in report["evidence"] if e["kept"]]
    if kept:
        st.markdown("#### Sources")
        seen = set()
        for e in sorted(kept, key=lambda e: -(e["relevance_score"] or 0)):
            if e["url"] in seen:
                continue
            seen.add(e["url"])
            date = f" · {e['published_date']}" if e.get("published_date") else ""
            st.markdown(f"- E{e['id']} [{md(e.get('title') or e['url'])}]({e['url']}) "
                        f":gray[{e.get('domain', '')}{date} · relevance {e['relevance_score']:.2f}]")

    with st.expander("Trace", icon=":material/timeline:"):
        tin = sum(t.get("input_tokens") or 0 for t in report["trace"])
        tout = sum(t.get("output_tokens") or 0 for t in report["trace"])
        a, b, c3, d = st.columns(4)
        a.metric("Sub-queries", len(report["sub_queries"]))
        b.metric("Evidence kept", f"{len(kept)}/{len(report['evidence'])}")
        c3.metric("Input tokens", tin)
        d.metric("Output tokens", tout)
        st.dataframe(
            [{"step": t["step"], "level": t["level"], "message": t["message"],
              "latency_ms": t["latency_ms"], "in_tok": t["input_tokens"], "out_tok": t["output_tokens"],
              "at": t["created_at"]} for t in report["trace"]],
            use_container_width=True, hide_index=True,
        )
        st.markdown("**Sub-queries**")
        st.dataframe(
            [{k: s[k] for k in ("id", "round", "text", "status", "result_count")} for s in report["sub_queries"]],
            use_container_width=True, hide_index=True,
        )
        st.markdown("**Evidence** (dropped items kept with a reason)")
        st.dataframe(
            [{"id": e["id"], "kept": bool(e["kept"]), "drop_reason": e["drop_reason"],
              "relevance": e["relevance_score"], "url": e["url"]} for e in report["evidence"]],
            use_container_width=True, hide_index=True,
            column_config={"url": st.column_config.LinkColumn("url")},
        )


def main() -> None:
    st.set_page_config(page_title="Research Agent", page_icon=":material/travel_explore:", layout="wide")
    settings = get_settings()
    conn = get_conn(settings.db_path)

    with st.sidebar:
        st.markdown("### Research Agent")
        st.caption("Nebius Token Factory + Tavily")
        st.markdown(f"Planner: `{settings.models.planner}`  \nWriter: `{settings.models.writer}`  \n"
                    f"Verifier: `{settings.models.verifier}`")
        recent = conn.execute("SELECT id, question, status FROM tasks ORDER BY id DESC LIMIT 20").fetchall()
        if recent:
            st.markdown("#### Recent runs")
            for r in recent:
                if st.button(f"#{r['id']} {r['question'][:50]}", key=f"t{r['id']}", help=r["status"],
                             use_container_width=True):
                    st.session_state.task_id = r["id"]

    st.title("Research Agent")
    st.caption("Ask a question. Every claim is cited, then checked against the cited excerpt.")
    with st.form("ask"):
        question = st.text_area("Question", placeholder="e.g. What changed in the EU AI Act timeline in 2025?")
        submitted = st.form_submit_button("Research", type="primary", icon=":material/search:")

    if submitted and question.strip():
        missing = settings.missing_keys()
        if missing:
            st.error(f"Missing environment variables: {', '.join(missing)}. Copy .env.example to .env.")
            return
        llm, search = default_clients(settings)
        status = st.status("Starting…", expanded=True)

        def on_event(e: dict) -> None:
            status.update(label=f"{STEP_LABEL.get(e['step'], e['step'])}…")
            status.markdown(f"{LEVEL_ICON.get(e['level'], '')} **{e['step']}** · {md(e['message'])}")

        report = asyncio.run(run_research(question, llm=llm, search=search, conn=conn,
                                          settings=settings, on_event=on_event))
        s = report["task"]["status"]
        status.update(label=f"Finished: {s.replace('_', ' ')}", expanded=False,
                      state="error" if s == "failed" else "complete")
        st.session_state.task_id = report["task"]["id"]

    task_id = st.session_state.get("task_id")
    if task_id and storage.get_task(conn, task_id):
        render_report(load_report(conn, task_id))


main()
