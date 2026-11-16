# Oncology Update Briefs

**What has changed for this clinical question since a given date?** This research agent turns an oncologist's question into a dated update brief they can read in under five minutes and verify in seconds. Every claim cites a source excerpt, every number is shown beside the verbatim sentence it came from, and a verifier model checks each claim against the original text.

Built for the Nebius hackathon. It uses [Nebius Token Factory](https://tokenfactory.nebius.com) for all LLM inference and [Tavily](https://tavily.com) `search` + `extract` for live web evidence.

> **Clinician-support tool.** This brief summarizes published and public information to support a clinician's professional judgment and independent review. It is not for patients and does not give patient-specific recommendations.
>
> **Use limits:**
> - clinician-facing only
> - not for patients
> - not for urgent or time-critical decisions
> - does not use patient data
> - summarizes published and public information for independent review

## Who it's for and the task

The user is a practicing oncologist. A guideline was last updated, or a major paper came out, on some date, and they need to know what has changed since then for one clinical question. For example: first-line EGFR-mutant metastatic NSCLC since the 2024 ESMO update, or HER2-low metastatic breast cancer since June 2025.

The output is a dated brief:

1. **Bottom line:** one sentence, with confidence (high, moderate or low) and the reason for it.
2. **What's new since [anchor date]:** one item per development, newest first. Each item shows:
   - the date and source type
   - what changed
   - the evidence level in plain words (e.g. "phase 3 randomized trial")
   - whether it is peer-reviewed or **PRELIMINARY**, labeled in the item itself
   - the study population, or "population not stated"
   - key numbers, each beside its verbatim source sentence
   - the source links with their verdicts
3. **Sources disagree:** conflicts, and how the sources differ.
4. **Not yet established:** aspects with insufficient evidence.
5. **Check before acting:** e.g. full publication pending, label wording, guideline version.
6. **The clinician-support statement.**

### Why live web evidence complements PubMed-based tools

PubMed-based tools are strong on indexed, peer-reviewed literature. Many decision-relevant changes in oncology appear on the web first, or never appear in PubMed at all:

- regulatory approvals, label changes and safety communications (FDA, EMA)
- guideline updates (NCCN, ESMO, ASCO)
- conference abstracts and trial readouts
- trial status changes (ClinicalTrials.gov)
- company announcements

This agent searches those sources first, filtered to on or after the anchor date. It labels anything preliminary as preliminary, so it adds to a PubMed workflow rather than replacing it.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # fill in NEBIUS_API_KEY and TAVILY_API_KEY
uv run python scripts/list_models.py   # confirm the STRONG and FAST model IDs exist
```

`list_models.py` prints every Token Factory model ID and marks the configured STRONG and FAST models `OK` or `NOT FOUND`, with similar IDs suggested. Set `RESEARCH_STRONG_MODEL` and `RESEARCH_FAST_MODEL` in `.env`, or edit `config.py`, before relying on them. **Keys are read only from the environment and are never logged or printed.**

## Run

```bash
# CLI: shows the brief and planned queries, asks to start (or pass --yes)
uv run research "What has changed for first-line EGFR-mutant metastatic NSCLC since 2024-06-01?"
uv run research "..." --yes --export ris --out sources.ris
uv run research --run 12                      # approve a prepared task later
uv run research --show 12                     # reprint a stored brief

# Web UI
uv run streamlit run app/streamlit_app.py
```

**CLI exit codes:**

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | failed |
| 2 | missing keys |
| 3 | insufficient evidence |
| 4 | needs clarification |
| 5 | not approved |
| 6 | identifiers detected |

**UI:**
- a question box with the no-identifiers notice
- the brief and planned queries, with a **Start research** button
- live progress: the run happens in a background thread and the page polls SQLite, so a refresh doesn't interrupt it
- the brief with verdict, claim-type, tier and PRELIMINARY badges
- numbers beside their source sentences
- clickable sources, CSV and RIS downloads, credit and token totals
- an expandable trace showing the model ID and prompt version for every step

**Tests and evals:**

```bash
uv run pytest                    # mocked unit tests, no network
uv run pytest -m live -s         # optional live smoke test (skipped without keys)
uv run python evals/run_evals.py # 15 questions, live
uv run python evals/compare.py   # drift gate against evals/baseline.json
```

## Deploy to the cloud (container on a VM, e.g. Nebius AI Cloud)

One image runs the Streamlit UI. The same image has the `research` CLI, and both share one SQLite database on a persistent volume, so run history survives restarts and redeploys.

**1. Create a VM.**
- An Ubuntu CPU instance with a public IP and your SSH key is enough, because all inference runs on Token Factory.
- Open ports 22, 80 and 443.
- Install Docker Engine with the Compose plugin.

**2. Get the code and configure it.**
```bash
git clone https://github.com/Gnana-Shishir-Kumar/gauss-jordan.git && cd gauss-jordan
cp .env.example .env    # set NEBIUS_API_KEY, TAVILY_API_KEY, APP_PASSWORD (+ DOMAIN for HTTPS)
```
`APP_PASSWORD` is required: compose refuses to start the app without it. Keys are read from `.env` at runtime and are never put into the image.

**3. Start it.**
```bash
docker compose up -d --build                   # UI on the VM at 127.0.0.1:8501 only
docker compose --profile https up -d --build   # plus Caddy: https://$DOMAIN with an automatic certificate
```
- **Without a domain:** reach the UI through an SSH tunnel. Run `ssh -L 8501:localhost:8501 user@VM_IP`, then open http://localhost:8501.
- **With a domain:** point its DNS A record at the VM and use the `https` profile. Caddy terminates TLS and proxies to the app.
- **Either way,** the app asks for `APP_PASSWORD` once per browser session.

**4. Use the CLI on the VM** (same database as the UI, so `--show` works on runs started in the browser):
```bash
docker compose exec app python scripts/list_models.py
docker compose exec app research "What has changed for first-line EGFR-mutant metastatic NSCLC since 2024-06-01?"
docker compose exec app research --show 3 --export ris --out /data/task3.ris
docker compose cp app:/data/task3.ris .
```

**Operations:**
| Task | Command |
|---|---|
| Update | `git pull && docker compose up -d --build` |
| Logs | `docker compose logs -f app` |
| Back up history | `docker compose cp app:/data/research.db ./backup.db` |

The image runs as a non-root user and has a health check on `/_stcore/health`.

## Architecture

```mermaid
flowchart TD
    Q[Question] --> P{Identifier check}
    P -- identifiers --> X[Blocked, nothing stored]
    P --> B["1. Brief (STRONG)<br/>cancer type, setting, question, anchor date"]
    B -- too vague --> C[Clarifying question]
    B --> PL["2. Plan (STRONG)<br/>3-5 broad queries"]
    PL --> A[3. Approval<br/>UI button / CLI prompt or --yes]
    A --> R["4. Research workers (parallel)<br/>Tavily search: tiers, start_date, cache<br/>snippet scoring, batched extract, cache<br/>FAST notes with verified numbers/design/population"]
    R --> CV["5. Coverage (STRONG)<br/>aspects: covered / thin / conflict<br/>independence = distinct domains, in code"]
    CV -- thin or conflict, no stop rule --> R
    CV -- stop rule --> W["6. Write (STRONG)<br/>from notes only"]
    W --> V["7. Verify<br/>number check in code, then FAST verifier vs original excerpt"]
    V --> RP["8. Repair once (STRONG)<br/>re-verify revised claims only"]
    RP --> F[Finalize: tier-3-only -> uncertain,<br/>confidence caps, status]
    F --> OUT[Update brief + trace in SQLite]
```

- **Engine** (`src/research_agent/engine/`): plain async functions, one per step (`brief`, `planner`, `research`, `coverage`, `writer`, `verifier`), plus `checks` (the deterministic guards) and the `orchestrator`. The orchestrator exposes `prepare()` (brief and plan, then it stops for approval), `run()` (everything after approval) and `run_all()`. There is no UI code in the engine, so a FastAPI wrapper could call these functions as they are.
- **Backends** (`backends/`): `llm.py` wraps Token Factory through `AsyncOpenAI`. `tavily.py` implements the `SearchBackend` protocol in `search_base.py`. `cache.py` checks before every Tavily call. To add another source, implement `search()` and `extract()`; the engine doesn't change.
- **Storage** (`storage.py`): SQLite, written after every step. A crash leaves a full trace, and the UI polls it for progress.
- **Everything else:** `prompts.py` (all prompts and `PROMPT_VERSION`), `sources.py` (tiers and source types), `privacy.py`, `exports.py`, `metrics.py`, `render.py`.

### How it decides what to search, and when to stop

1. **Brief.** The STRONG model extracts cancer type, setting (stage, line, biomarker), the question, the anchor date, and 2–6 aspects a complete answer must cover.
   - With no anchor date, it defaults to 12 months ago and says so in the brief.
   - If cancer type or setting is too vague to search well, it returns a clarifying question instead of searching.
2. **Plan.** 3–5 independent, fairly broad queries.
3. **Round 1** searches **tiers 1–2 only**, at basic depth, with up to 10 results per query and `start_date` set to the anchor date.
   - Each worker scores its hits on the **snippets** (Tavily score blended with term overlap) and sends only its top 2 URLs to extraction.
   - The round's URLs are extracted together, at most 20 per call, after checking the cache.
4. **Coverage.** The STRONG model labels each aspect covered, thin or conflicting, and cites evidence IDs.
   - Code, not the model, decides whether an aspect has **≥2 independent sources**, meaning two distinct domains.
   - Follow-up queries are generated only for thin or conflicting aspects. They run at advanced depth over tiers 1–3.
   - Each follow-up names its aspect. Code allows **at most one follow-up per aspect per round**, and none for an aspect that already has independent sources.
   - A query is also skipped if it overlaps an earlier one by ≥0.7 (term Jaccard). Every skipped query is logged with its reason.
5. **Stop rules**, checked in this order. The first that fires is saved to `tasks.stop_reason`:

   | Stop reason | When |
   |---|---|
   | `coverage_complete` | every aspect has ≥2 independent sources |
   | `few_new_evidence` | the round added fewer than 2 new kept evidence items |
   | `max_rounds` | 2 rounds have run (round 1 plus one follow-up round) |
   | `time_budget` / `credit_budget` | the time or credit budget is used up |
   | `no_new_queries` | there are no non-duplicate follow-ups |
   | `coverage_check_failed` | the coverage step failed |

### Source tiers and how trust is shown

| Tier | Domains | Use |
|---|---|---|
| 1: regulatory, registries, guidelines | fda.gov, ema.europa.eu, clinicaltrials.gov, cancer.gov, nccn.org, esmo.org, asco.org, ncbi.nlm.nih.gov, pubmed.ncbi.nlm.nih.gov | all rounds |
| 2: journals | nejm.org, thelancet.com, jamanetwork.com, ascopubs.org, annalsofoncology.org, aacrjournals.org, nature.com | all rounds |
| 3: oncology news, press releases | ascopost.com, onclive.com, targetedonc.com, cancernetwork.com, prnewswire.com, businesswire.com, globenewswire.com | follow-up rounds only, as leads; **always labeled lower trust** |

- **Filtering:** domains are restricted with Tavily `include_domains`, and dates with `start_date` (publish date). Any result outside the allowlist is recorded as `off_allowlist` and not used.
- **Tier 3:** a claim whose only supporting sources are tier 3 is retagged **uncertain**.
- **Press releases:** company newsroom domains can't be put on an allowlist, so they are only found through the three wire services listed.
- **Source types:** `regulatory_action`, `guideline`, `registry_entry`, `peer_reviewed_article`, `conference_abstract`, `preprint`, `press_release`, `news`. Deterministic URL rules run first (e.g. `fda.gov` is a regulatory action, `_suppl` DOIs and `meetings.asco.org` are conference abstracts, journal domains are peer-reviewed articles). The FAST worker classifies only what the rules leave ambiguous, and anything left over falls back to the conservative `news`.
- **Preliminary evidence:** conference abstracts, preprints and press releases are **PRELIMINARY**. The label is added by code inside each item, not left to the model or a legend.
- **Peer review:** `peer_reviewed` is true only when a supporting source is a peer-reviewed article.
- **Login walls:** some guideline pages need a login. When extraction fails, the evidence is stored with `drop_reason = source_unavailable: …`, any citation to it gets the verdict `source_unavailable`, and the run continues.
- **Confidence caps:** the bottom line is capped at **low** when no finding is supported by a tier 1–2, non-preliminary source. It is capped at **moderate** when items are flagged or sources disagree.

### How evidence is deduplicated and checked

- **Dedupe.** URLs are normalized (lowercase host, no `www.`, no fragment, no tracking parameters, sorted query). Evidence is deduped by normalized URL and by a content hash, within and across rounds. Dropped items are kept with a reason:
  - `duplicate_url`, `duplicate_content`
  - `low_relevance`, `not_selected_for_extraction`
  - `off_allowlist`, `source_unavailable`
- **Untrusted data.** Retrieved text and notes derived from it always sit inside `<untrusted_source>` or `<untrusted_note>` blocks. Any such tags inside page text are removed first, and every prompt containing them says never to follow instructions found there. The agent only reads the web and takes no actions based on page content.
- **Worker notes (FAST)** are checked against the excerpts in code:
  - Each number must come with a sentence that appears verbatim in the excerpt and contains that number.
  - A design phrase (e.g. "phase 3 randomized") must appear in the excerpt.
  - A population needs a verbatim population sentence that states it.
  - Anything that fails is dropped and logged.
- **The writer** sees notes and trusted metadata only, never raw page text.
- **Numbers.** Before the LLM verifier runs, code checks that every number in a claim (HR, %, months, p-values, doses, dates) appears in that claim's stored verbatim source sentence, and that the sentence appears in the cited excerpt. A failure marks the citation **unsupported** without calling the LLM and sends the claim to repair. Numbers inside names such as HER2, CDK4/6, PD-1, G12C or "phase 3" are not checked as results.
- **Evidence level** may only name phase or design words stated in a cited excerpt; otherwise it is "not stated".
- **Population** must come from a verified note for a cited source; otherwise it is "population not stated". It is never inferred.
- **Verifier (FAST).** Each claim–citation pair is checked against the original excerpt and gets `supported`, `partial`, `unsupported` or `source_unavailable`.
- **Repair (once).** Unsupported or partial claims go back to the STRONG writer with the verifier's note. The writer may revise the claim, cite other evidence, or mark it uncertain. Only revised claims are re-verified. Anything still unsupported is shown **FLAGGED: not verified** and never as verified. If nothing is supported, the task ends `insufficient_evidence`.

## Safety and governance

| Concern | What addresses it |
|---|---|
| **Hallucination detection** | Number checks in code against verbatim source sentences; per-citation LLM verification against the original excerpt; one repair pass; unverified claims shown as FLAGGED; `insufficient_evidence` when nothing holds up; writer limited to verified notes. |
| **Explainability** | Every claim cites evidence IDs with links; numbers are shown beside their source sentences; claim type, evidence level, peer-review or PRELIMINARY label, tier and verifier note appear on each item; the stop reason is shown. |
| **Clinical safety** | Clinician-facing statement and use limits in the UI, CLI and README; no patient-specific recommendations; not for urgent decisions; tier-3-only claims tagged uncertain; confidence caps; a "check before acting" section; preliminary evidence labeled in the item. |
| **Privacy** | No patient data is used. An input check blocks MRN-like numbers, dates of birth and names with "patient", Mr or Mrs before anything is stored. Keys come only from the environment and are never logged. |
| **Auditability** | SQLite trace written after every step: sub-queries, every evidence item including dropped ones and why, notes, claims, verdicts, latency, tokens and Tavily credits. Model ID and prompt version are on every task and trace event. CSV and RIS exports. |
| **Drift** | `PROMPT_VERSION` is recorded on every task and event. `evals/compare.py` fails if citation support rate or number-check pass rate drops by more than 3 points against the baseline. **Rerun evals and the drift gate before changing any model or prompt.** |
| **Evidence-population bias** | The population is shown per item when a source states it (age range, region, key exclusions), and is never inferred; otherwise "population not stated", so clinicians can judge applicability to their patients. |

**Why the design follows FDA's non-device CDS principles.** The FDA's criteria for clinical decision support software that is not a medical device (21st Century Cures Act §520(o)(1)(E), FDA CDS guidance, 2022) include:

- **No image or signal analysis.** The tool does not acquire, process or analyze medical images or signals.
- **Published information only.** It displays and summarizes medical information such as guidelines, regulatory documents and studies, not patient data.
- **Clinician-facing.** It supports a healthcare professional, not patients or caregivers, and is not intended for time-critical decisions.
- **Transparent, independently reviewable basis.** Every claim links to its source, numbers sit beside their verbatim sentences, and verification verdicts are shown, so the clinician need not rely mainly on the tool.

**This is a design rationale, not a regulatory determination.** Whether a specific deployment is a device depends on its intended use and labeling, and needs a regulatory review.

## Exports

From the UI download buttons or `--export csv|ris`, the sources used in a brief export as:
- **CSV:** evidence IDs, title, URL, domain, source type, date, trust tier, verdicts.
- **RIS** for Zotero or EndNote: `TY` by source type (JOUR, ABST, GOVDOC, PRESS…), `TI`, `UR`, `PY` and `DA`, `DB`, `KW` with source type and tier, and `N1` with type, tier and evidence IDs.

Both use the standard library only.

## Evals and drift gate

- **Questions:** `evals/questions.jsonl` holds 15 oncology "what's changed" questions across 12 cancer types. Three use fictional investigational codes and should end in insufficient evidence. Each has `clinician_rating` and `clinician_notes` fields, left `null` for later expert review.
- **Running:** `evals/run_evals.py` runs them live with a fresh database (so the cache doesn't hide credit costs). It saves `evals/results/<stamp>.json` with the model IDs, the prompt version and these pooled metrics:

| Metric | Definition |
|---|---|
| citation support rate | supported citations / checked citations |
| numeric check pass rate | first-pass number checks passed / numeric claims |
| tier 1–2 support share | supported claims backed by a tier 1–2 source / supported claims |
| preliminary labeled share | claims citing preliminary evidence that carry the PRELIMINARY label |
| insufficient-evidence accuracy | runs that ended `insufficient_evidence` exactly when expected |
| cost and speed | average credits, latency and tokens per question |

- **Drift gate:** `evals/compare.py` compares the newest result with `evals/baseline.json` and exits 1 if a gated metric drops by more than 3 points or is missing. `--set-baseline` saves the latest run as the new baseline.

## Credit and latency numbers

**Measured live** on 2026-11-15, with the live smoke-test question (first-line EGFR-mutant metastatic NSCLC since 2024-06-01), default settings and an empty cache:

| | Value |
|---|---|
| Result | `done`, stop reason `max_rounds`; 5 items and 1 conflict, every claim verified or repaired |
| Tavily credits | **11**: round 1 had 5 basic searches (5) and one extract batch (1); round 2 had 2 advanced searches (4) and one extract batch (1) |
| Wall time | about **8 minutes** (3.5 to 6 s per search, under 5 s per extract batch) |
| Tokens | 31,193 input / 69,510 output |
| Slowest LLM calls | writer 165 s (about 16,000 output tokens), notes 19 to 103 s each, coverage 51 to 55 s each |
| Fastest LLM calls | verifier 2 to 5 s per claim |

The first live attempt ran with the original settings: 3 rounds, a 4,096-token output limit and no per-aspect cap. It used **23 credits** and then failed at the writer, because the token limit cut off its JSON. That run is why the output limit, timeout, round count and follow-up cap are what they are now.

- **Worst case under the defaults:**

  | Item | Credits |
  |---|---|
  | Round 1: 5 basic searches | 5 |
  | Round 1: extract 10 URLs | 2 |
  | Round 2: up to 5 advanced searches (one per thin aspect) | 10 |
  | Round 2: extract 10 URLs | 2 |
  | **Total** | **19** |

  That is below the 60-credit budget, which also stops the loop. Extract is billed at 1 credit per 5 successful URLs at basic depth.
- **Cache:** repeat questions within 7 days reuse cached searches and pages at 0 credits.
- **Latency:** dominated by LLM calls. These models write long reasoning before their JSON, so each call is allowed up to 16,384 output tokens (`llm_max_tokens`) and 300 s. Other defaults: 30 s per search, 60 s per extract, and a 420 s research budget.
- **Across questions:** these numbers come from one question. Run `evals/run_evals.py` for averages over all 15.

### Credits vs traditional search

`evals/credit_comparison.py` replays a stored run's own queries and search results under traditional strategies, so it makes no new API calls. For the live run above:

| Strategy | Search credits | Extract calls | Pages extracted | Extract credits | **Total** | vs project |
|---|---|---|---|---|---|---|
| **This project** (billed): snippet scoring, top results only, batched extract, basic depth in round 1 | 9 | 2 | 14 | 2 | **11** | 1.0x |
| Fetch every result, batched 20 per call | 9 | 4 | 67 | 14 | **23** | 2.1x |
| Fetch every result, one call per page | 9 | 67 | 67 | 67 | **76** | 6.9x |
| Advanced search + fetch every page | 14 | 67 | 67 | 67 | **81** | 7.4x |

Asking the same question again within 7 days costs the project **0** credits (cache), against 23–81 for the traditional strategies.

Where the savings come from:
- **Reading snippets first.** The 7 searches returned 67 unique pages. Scoring the snippets let the project extract only the 14 most relevant, instead of fetching all 67.
- **Batching extraction.** Pages from a round are extracted together, up to 20 per call, instead of one call per page.
- **Depth by need.** Round 1 uses basic search (1 credit). Only targeted follow-ups use advanced (2 credits), and only one per thin aspect.
- **The cache.** Searches and pages are reused for 7 days.

How the numbers are computed:
- The project row is what Tavily actually billed.
- The other rows price the same queries and results at Tavily's published rates: search 1 credit (basic) or 2 (advanced); extract 1 credit per 5 successful URLs, rounded up per call.
- Rounding up per call is what makes one-page-per-call expensive. If Tavily instead pooled fractions across calls, that row would fall to about 23, the same as the batched row, and the snippet-scoring and cache savings would remain.
- In this run Tavily billed slightly less for extraction than the published rate (2 credits for 14 pages, against 3).

```bash
uv run python evals/credit_comparison.py --markdown    # latest finished task in research.db
```

## Example brief (live run, redacted trace)

> A real run from 2026-11-15, shown to illustrate the output. It is a snapshot of web sources from that day, **not clinical guidance**: check the sources before acting.
>
> **Redactions:** long claim text and queries are trimmed with `[…]` and the per-call token counts are removed. Traces never contain patient data, identifiers or API keys.

```
Question: What has changed for first-line treatment of metastatic EGFR-mutant non-small cell lung cancer since 2024-06-01?
Status:   done

BRIEF
Cancer type: non-small cell lung cancer
Setting: metastatic, first line, EGFR-mutant
Since: 2024-06-01
Aspects: regulatory approvals and label changes; guideline updates; trial readouts and conference data;
         safety communications; trial status changes

1. BOTTOM LINE
First-line treatment of metastatic EGFR-mutant NSCLC has shifted toward combination regimens, with FDA approval
of amivantamab plus lazertinib and ASCO living guideline endorsement of combination therapy as preferred, while
osimertinib monotherapy remains an option for patients prioritizing tolerability. (confidence: moderate;
multiple tier 1 and 2 sources support the shift, but some dates are unknown and OS/comparative data remain preliminary.)

2. WHAT'S NEW SINCE 2024-06-01 (newest first)
1. 2025-11-19 · Peer-reviewed article · Tier 1
   What changed: An individual patient data comparison of FLAURA2 and MARIPOSA found a statistically significant
   PFS difference favoring FLAURA2 in the ITT population (HR 0.79), not confirmed for OS; […]  [fact; supported]
   Evidence level: phase 3 clinical trials · Peer-reviewed
   Population: population not stated
   Key numbers (source sentences):
     E69: "In progression free survival (PFS), there was a statistically significant difference favoring the FLAURA2
           in the intent-to-treat (ITT) population [Hazard ratio (HR) 0.79], not confirmed in the overall survival
           (OS) analysis where we did not find any significant difference."
     […]
   Source E69: https://pubmed.ncbi.nlm.nih.gov/41270639 (supported)
2. 2024-08-19 · Regulatory action · Tier 1
   What changed: FDA approved lazertinib in combination with amivantamab-vmjw for first-line treatment of locally
   advanced or metastatic NSCLC with EGFR exon 19 deletions or exon 21 L858R substitution mutations; lazertinib with
   amivantamab improved PFS versus osimertinib (HR 0.70), median PFS 23.7 versus 16.6 months.  [fact; supported]
   Evidence level: regulatory decision · Not peer reviewed (regulatory action)
   Population: first-line treatment of locally advanced or metastatic NSCLC with EGFR exon 19 deletions or exon 21
               L858R substitution mutations
   Key numbers (source sentences):
     E67: "Lazertinib with amivantamab demonstrated a statistically significant improvement in PFS compared to
           osimertinib with a hazard ratio of 0.70 (95% confidence interval [CI]: 0.58, 0.85; p-value=0.0002)."
     E67: "The median PFS was 23.7 months (95% CI: 19.1, 27.7) in the lazertinib with amivantamab arm and 16.6
           months (95% CI: 14.8, 18.5) in the osimertinib arm."
   Source E67: https://fda.gov/drugs/resources-information-approved-drugs/fda-approves-lazertinib-amivantamab-vmjw-non-small-lung-cancer (supported)
3. date not stated · Peer-reviewed article · Tier 2
   What changed: ASCO living guideline: FLAURA2 and MARIPOSA support superior efficacy of combination treatment versus
   osimertinib alone; clinicians should consider combination regimens the preferred option, with osimertinib
   monotherapy remaining an option for patients prioritizing tolerability; […]  [fact; supported]
   Evidence level: expert guideline · Peer-reviewed
   Source E45: https://ascopubs.org/doi/10.1200/JCO-25-02822 (supported)
4. date not stated · Peer-reviewed article · Tier 2
   What changed: An exploratory overall survival analysis of FLAURA2 in patients with poorer prognostic factors was
   published in Annals of Oncology; the publication year is not established.  [uncertain; partial]
   (failed the number check on the first draft; repaired, then re-verified as partial)
5. date not stated · Conference abstract · Tier 1
   What changed: In descriptive risk analyses, any-grade VTE incidence was numerically higher with amivantamab plus
   lazertinib (21%) than with amivantamab (11%) or lazertinib (11%) monotherapy.  [fact; supported]
   Evidence level: descriptive univariate risk analyses · PRELIMINARY (conference abstract), not peer reviewed
   Key numbers (source sentences):
     E66: "The incidence of VTE events of any grade was numerically higher in pts who received ami+laz (21%) than
           those who received ami (11%) or laz (11%) monotherapy."
   Source E66: https://asco.org/abstracts-presentations/218907 (supported)

3. SOURCES DISAGREE
1. One source describes osimertinib as the preferred first-line option; the ASCO living guideline now states
   combination regimens should be considered preferred.  [uncertain; supported]
   Sources E45 (ascopubs.org) and E47 (ncbi.nlm.nih.gov/pmc) (both supported)

4. NOT YET ESTABLISHED
- No direct head-to-head FLAURA2 versus MARIPOSA trial; the indirect comparison suggests a PFS difference but OS was not confirmed.
- VTE risk with amivantamab plus lazertinib is based on descriptive univariate analyses, not a definitive safety analysis.
- Exact dates of the ASCO living guideline update and the VTE analysis are not stated in the evidence.
- […]

5. CHECK BEFORE ACTING
- Confirm current FDA label wording for lazertinib/amivantamab, including exact mutation indications.
- Confirm ASCO living guideline version and date.
- Confirm VTE risk data in a peer-reviewed publication or full safety analysis.
- […]

6. Clinician-support tool. […] Use limits: […]

Stop reason: max_rounds
Tavily credits: 11 · tokens: 31193 in / 69510 out
Models: strong=deepseek-ai/DeepSeek-V4-Pro-0813, fast=deepseek-ai/DeepSeek-V4.1-Flash-1017 · prompt version: onc-2026.11.2

TRACE
  [brief   ] Brief written                                              (4 s, STRONG)
  [plan    ] Planned 5 queries                                          (6 s, STRONG)
  [approval] Waiting for approval / Approved; research started
  [round   ] Round 1: 5 queries, basic depth, tiers [1, 2]
  [search  ] 10 results for 'first-line EGFR-mutant NSCLC safety communication FDA EMA 2024 2025'  (1 cr)
  [search  ] 10 results for '[…] clinical trial status change 2024 2025'                         (1 cr)
  [search  ] 10 results for '[…] trial results ASCO ESMO WCLC 2024 2025'                         (1 cr)
  [search  ] 10 results for 'NCCN ESMO guideline update first-line EGFR-mutant metastatic NSCLC […]' (1 cr)
  [search  ] 10 results for 'FDA EMA regulatory approval label change […]'                        (1 cr)
  [extract ] Extracted 8/10 URLs in one batch     (2 failed: annalsofoncology.org, oncologypro.esmo.org -> source_unavailable)  (1 cr)
  [evidence] Kept 2/2 for each of 4 queries
  [notes   ] 4 + 2 + 4 + 3 notes                                        (19-103 s each, FAST)
  [coverage] 2/5 aspects have >= 2 independent sources; 2 follow-up queries
             (2 more skipped: "aspect already covered")                 (55 s, STRONG)
  [round   ] Round 2: 2 queries, advanced depth, tiers [1, 2, 3]
  [search  ] 10 results for 'MARIPOSA FLAURA2 clinical trial status update ClinicalTrials.gov 2025'  (2 cr)
  [search  ] 10 results for 'amivantamab lazertinib safety warning venous thromboembolism FDA 2025'  (2 cr)
  [extract ] Extracted 4/4 URLs in one batch                            (1 cr)
  [notes   ] 3 + 7 notes                                                (54-55 s each, FAST)
  [coverage] 2/5 aspects have >= 2 independent sources                  (51 s, STRONG)
  [stop    ] Research stopped after round 2: max_rounds
  [write   ] Drafted 5 items, 1 conflicts (sufficient=True)             (165 s, STRONG)
  [check   ] Claims 1, 4, 5: numeric check passed · Claim 3: numeric check failed
  [verify  ] Claims 1, 2, 4, 5: supported · Claim 6: 2 supported        (2-5 s each, FAST)
  [repair  ] Revised 1/1 failed claims                                  (23 s, STRONG)
  [verify  ] Claim 3: 1 partial
  [done    ] Finished: done
```

## Models and integration setup

| Tier | Default | Used for |
|---|---|---|
| STRONG | `deepseek-ai/DeepSeek-V4-Pro-0813` | brief, planner, coverage check, writer, repair |
| FAST | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | worker notes, citation verification |

These IDs were confirmed available on Token Factory with `scripts/list_models.py` on 2026-11-15. Token Factory model IDs carry version suffixes and change over time, so rerun that script if a call fails with "model not found". Both tiers are configurable in `config.py` or `.env` (`RESEARCH_STRONG_MODEL`, `RESEARCH_FAST_MODEL`). Rerun the evals and the drift gate after changing either one.

**Token Factory**
- **Client:** `AsyncOpenAI(base_url="https://api.tokenfactory.nebius.com/v1")`.
- **JSON output:** JSON mode, with a prompt-only fallback for models that reject `response_format`.
- **Validation:** every output is validated with Pydantic. On failure the model gets one retry with the error, then a safe fallback:

  | Step | Fallback |
  |---|---|
  | Brief | the question itself |
  | Plan | the question as the only query |
  | Notes | that worker is skipped |
  | Coverage | stop with `coverage_check_failed` |
  | Writer | `failed` |
  | Verifier | `unsupported`, with a note |
- **Retries:** the SDK's own retries are off. `retry.py` applies exponential backoff with jitter to transient errors (timeouts, connection errors, 429, 5xx) for both services.

**Tavily (`search` and `extract` only; the `/research` endpoint is never used)**
- **Search:** `include_domains` by tier, `start_date` set to the anchor date, `include_usage` for exact credits, basic depth in round 1 and advanced in follow-ups.
- **Extract:** batched up to 20 URLs per call, with `query` and `chunks_per_source=3` so it returns the most relevant chunks.
- **Credits:** recorded in `trace_events.tavily_credits` for every call and totaled per task.

**Storage:** the specified schema plus:
- oncology columns: `cancer_type`, `setting`, `anchor_date`, `trust_tier`, `source_type`, `evidence_date`, `evidence_level`, `peer_reviewed`
- governance columns: `model_ids`, `prompt_version`, `model_id`
- answer structure: `answer_json`, `section`, `population`, `item_date`, `notes.details`, `claim_citations.source_sentence`
- two cache tables: `search_cache` and `extract_cache`

`research.db` is versioned (`PRAGMA user_version = 2`). An older database is refused with a clear message rather than silently migrated.
