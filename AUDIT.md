# Repository audit: gauss-jordan (Oncology Update Briefs)

Audit date: 2026-11-16. Audited commit: `603956c` (the tip of `feat/oncology-v2`, merged into `main` as PR #4, `24de252`).
Purpose: a self-contained handoff for an engineer or AI assistant picking this repo up with no prior context.

---

## 1. What this is

A research agent for the Nebius hackathon. A **practicing oncologist** asks "what has changed for this clinical question since a given date?" The agent produces a **dated update brief**: every claim cites a web source, every number sits beside its verbatim source sentence, and an LLM verifier checks each claim against the original excerpt.

- **LLM inference:** Nebius Token Factory through the `openai` SDK (`AsyncOpenAI`, base URL `https://api.tokenfactory.nebius.com/v1`).
  - STRONG model `deepseek-ai/DeepSeek-V4-Pro-0813`: brief, plan, coverage, writer, repair.
  - FAST model `deepseek-ai/DeepSeek-V4.1-Flash-1017`: worker notes, verifier.
  - Both IDs were confirmed live on 2026-11-15.
- **Web evidence:** Tavily `search` and `extract` only, never `/research`. Searches are restricted to a three-tier domain allowlist and to dates on or after the anchor date.
- **Storage:** SQLite (stdlib), written after every step. **UI:** Streamlit. **CLI:** `research`. **Tests:** pytest.
- **Judging weights from the brief:**

  | Criterion | Weight |
  |---|---|
  | Sourced and verifiable answers | 30% |
  | Works end to end | 25% |
  | Usefulness | 20% |
  | Agent design | 15% |
  | Meaningful use of both services | 10% |

  Clarity and reliability beat complexity.
- **Positioning:** clinician-support only. Not for patients, not for urgent decisions, no patient data.

## 2. Repository state

| Item | State |
|---|---|
| `main` | Has `feat/oncology-v2` merged (PR #4). Same tree as `603956c`. |
| `feat/oncology-v2` | Local and remote in sync at `603956c`. No uncommitted changes. |
| `origin/fix/verification-checks` | **Unmerged and diverged.** One commit `4228417` by `pjsk02`, branched from `e096149` (before the last four `main` commits). See section 6. |
| Tests | 102 mocked tests pass in about 4 s (`uv run pytest`). 1 live test, excluded by default (`uv run pytest -m live`). |
| Live runs | EGFR NSCLC question ran end to end twice with real APIs (details in section 5). |
| Evals | **Never run live.** No `evals/baseline.json`, so the drift gate (`evals/compare.py`) exits 2 until a baseline is set. |
| CI | **None.** No `.github/workflows`. Tests run only locally. |
| Secrets | `.env` exists locally and is gitignored. Not tracked. `research.db*` is gitignored. |
| Deployment | Docker image builds and runs locally (health, non-root, password gate, CLI, volume persistence verified). **Never deployed to a VM**, and the Caddy HTTPS profile is untested. |

## 3. Architecture (where things live)

```
src/research_agent/
  config.py          models, tier allowlist, limits, budget, network (timeouts, max_tokens), .env loader
  prompts.py         every prompt + PROMPT_VERSION (currently onc-2026.11.3); untrusted-data wrapping
  schemas.py         Pydantic models for every LLM output (Brief, Plan, WorkerNotes, Coverage, Draft, Repair, ClaimVerification)
  storage.py         SQLite schema (user_version 2) + queries; cache tables
  sources.py         tier lookup, rule-based source_type, preliminary/peer-reviewed labels
  evidence.py        URL normalization, content hash, excerpt choice, relevance, dedupe, near-duplicate queries
  privacy.py         blocks questions containing patient identifiers (before anything is stored)
  render.py          text rendering of the six-section brief, disclaimer, use limits, claim_verdict
  exports.py         CSV and RIS export of cited sources
  metrics.py         eval metrics; credits.py: credit comparison vs traditional search
  cli.py             `research` entry point
  backends/          llm.py (Token Factory wrapper), tavily.py, search_base.py (SearchBackend protocol), cache.py
  engine/            brief, planner, research (parallel workers), coverage, writer (+repair), verifier,
                     checks (deterministic guards), context (trace/budget), orchestrator (prepare/run/run_all)
app/streamlit_app.py UI only; runs research in a background thread, polls SQLite
evals/               questions.jsonl (15), run_evals.py, compare.py (drift gate), credit_comparison.py
Dockerfile, compose.yaml, deploy/Caddyfile   container deployment
```

**Pipeline:**
1. `prepare()`: identifier check, then brief (or a clarifying question), then plan, then `awaiting_approval`.
2. `run()`:
   - **Research rounds.** Round 1 searches tiers 1–2 at basic depth. Each worker scores snippets and takes its top 2 URLs, the round extracts in batches of up to 20, then FAST notes are taken.
   - **Coverage check.** Code counts independent sources by domain. Follow-ups are capped at one per thin aspect per round.
   - **Stop rule.** `max_rounds=2`, budgets, or no new queries.
   - **Write** from notes only. Writer citations are validated, and note IDs are mapped to evidence IDs.
   - **Deterministic checks:** numbers, design, population.
   - **FAST verifier** per claim.
   - **One repair pass,** then finalize: tier-3-only claims become uncertain, the bottom line's confidence is capped, and the task status is set.

## 4. Conventions to keep

- **Commit messages:** no "Co-Authored-By: Claude" trailers (owner's request).
- **Dependencies:** ask the owner before adding any dependency. Current runtime deps are `openai`, `tavily-python`, `pydantic` and `streamlit`; dev dep is `pytest`.
- **Prompt changes:** bump `PROMPT_VERSION` in `prompts.py` on any prompt change, then rerun evals and the drift gate.
- **Keys:** never log or print keys. They come from env/`.env` only.
- **Deterministic checks:** the guards in `engine/checks.py`, `writer.apply_item` and `orchestrator._finalize` are the product's main safety claim. Don't weaken them to make a run pass.
- **Live tests spend credits** (about 11–13 Tavily credits and around 8 minutes per question). Mocked tests use `FakeLLM` and `FakeSearch` in `tests/conftest.py`.

## 5. Live evidence so far (EGFR-mutant NSCLC, first line, since 2024-06-01)

| Run | Result | What it taught |
|---|---|---|
| 1 (3 rounds, max_tokens 4096) | **failed** at the writer | JSON truncated by the output-token limit. Raised it to 16,384 and the timeout to 300 s; cut rounds to 2; added the per-aspect cap. |
| 2 | **done**, 11 credits, 8 min | Worked end to end. Writer took 165 s. Became the README example and credit numbers. |
| 3 (CLI) | **insufficient_evidence** | Writer cited **note IDs as evidence IDs**, so every claim pointed at dropped pages. Separately, the CLI reused clients across two event loops ("Event loop is closed"). Both fixed in `603956c`. |
| 4 (CLI, after fixes) | **done**, 13 credits | 10/10 citations on read evidence: 6 supported, 3 partial, 1 unsupported. Repair then failed because the model omitted a number's `value`. Fixed in `603956c`, but **not yet confirmed live**. |

## 6. The unmerged `fix/verification-checks` branch

Commit `4228417` (`pjsk02`, 2026-11-15) branches from `e096149`, before the live-run fixes, and touches 18 files with 19 new tests. Its message lists:
- **Verification:** worst verdict wins, `partial` counts as flagged, a stricter bottom line, and items dated before the anchor are marked "not new".
- **Number checks:** range and CI handling, whole-word matching, a minimum sentence length.
- **Negation handling:** in design words and populations.
- **Cache keys:** exact queries instead of sorted terms; the extract cache keyed by URL plus question (schema v3).
- **Independence:** counted per site, so subdomains of one site count once.
- **Eval scoring:** failed or clarification runs no longer count as correct.

Merging it into `main` **conflicts** in `README.md`, `src/research_agent/engine/writer.py` and `src/research_agent/prompts.py`. Both lines of work used `PROMPT_VERSION = "onc-2026.11.2"` for different changes. It also bumps the DB schema to v3 while `main` is v2.

Recommendation: **rebase or re-apply it onto `main`**, not the other way round.
- Keep `main`'s note-to-evidence mapping (`writer.apply_item`), the optional number `value`, and the CLI client fix.
- Pick a fresh `PROMPT_VERSION` (e.g. `onc-2026.11.4`).
- Run the full suite plus one live run.

Many of the defects in section 7 are what this branch fixes, so merging it is the cheapest way to close them.

## 7. Findings in `main` (verified by running the code)

Severity ratings: **High** means it can make the brief wrong or misleading. **Medium** means it weakens a guarantee or wastes credits. **Low** is polish.

| # | Sev | Finding | Where | Fixed on the fix branch? |
|---|---|---|---|---|
| 1 | High | **Claim verdict is best-of.** One supporting citation makes the claim "verified" even if another citation is unsupported, and `partial` counts as verified. | `render.py:22` `claim_verdict`; `orchestrator.py` `_finalize` | Yes (worst verdict wins) |
| 2 | High | **Search cache keys collide.** `normalize_query` sorts terms and drops tokens of 2 characters or fewer, so "stage II" and "stage IV" collide, and so do "PD-1" and "PD-L1". A cached result for one is served for the other. | `evidence.py:60`, `backends/cache.py:15` | Yes |
| 3 | High | **"What's new" items are not checked against the anchor date.** A development dated before the anchor can appear as new. | `engine/writer.py` `apply_item`; `orchestrator.py` | Yes |
| 4 | Medium | **Range and CI upper bounds aren't checked.** In `0.58-0.85` and `30-40%` the number after the hyphen is never extracted, so it passes unchecked. | `engine/checks.py:21` `_NUM_RE` | Yes |
| 5 | Medium | **Very short "source sentences" are accepted.** A sentence like `"0.7"` passes the verbatim check. | `engine/checks.py:65` `check_numbers` | Yes (minimum 4 words) |
| 6 | Medium | **Negation is ignored.** The level "randomized trial" is kept when the excerpt says "non-randomized"; the population "small cell lung cancer" matches "non-small cell lung cancer". | `engine/checks.py:94,110` | Yes |
| 7 | Medium | **Subdomains count as independent sources.** `accessdata.fda.gov` and `fda.gov` count as 2 domains, which inflates "≥2 independent sources" and can stop research early. | `engine/coverage.py:41` | Yes |
| 8 | Medium | **Source type mislabels.** `dailynews.ascopubs.org` (ASCO Daily News) is classified `peer_reviewed_article` at tier 2. Unclassified tier-1 pages (e.g. cancer.gov) fall back to `news`. | `sources.py:22,71`; `engine/research.py:205` | No |
| 9 | Medium | **Stuck tasks.** If the CLI process or the Streamlit server dies mid-run, the task stays `researching`, `writing` or `verifying` forever. There is no stale-task detection, and the UI polls indefinitely. | `app/streamlit_app.py:66`; no recovery code | No |
| 10 | Medium | **The time budget doesn't bound the whole run.** `budget.time_s=420` is only checked between research steps, and LLM calls (writer up to 300 s) run after it. Live runs took about 8 minutes against a 7-minute budget. | `config.py:90`, `engine/context.py:45` | No |
| 11 | Medium | **Many items lack dates.** `evidence_date` only comes from Tavily's `published_date` (often missing) or a worker date whose year appears in the excerpt. Items show "date not stated", which weakens the newest-first ordering. | `engine/research.py:201` | Partially (dates must match a cited source) |
| 12 | Low | **Near-duplicate query check is word-overlap only**, so reworded duplicate queries can pass. Mitigated by the per-aspect cap. | `evidence.py:71` | Improved |
| 13 | Low | **Evals have never run;** there is no baseline, so the drift gate can't pass yet. Eval scoring counts failed runs as correct insufficient-evidence. | `evals/`, `metrics.py` | Scoring fixed |
| 14 | Low | **No CI.** A GitHub Actions job running `uv run pytest` would protect `main`. | — | No |
| 15 | Low | **Credit comparison relies on an assumption.** It prices extraction "per 5 URLs, rounded up per call". Tavily billed less than that in live runs; the README says so. | `credits.py` | n/a |
| 16 | Low | **Latency.** About 8 minutes per question; notes and coverage calls take 20–160 s each because the models emit long reasoning (5–16k output tokens). This matters for live demos. | — | No |

**What's solid:**
- the identifier check (blocks before storing)
- untrusted-data wrapping and tag stripping
- batched, cached Tavily usage with per-call credit tracking
- the note-to-evidence citation mapping
- `source_unavailable` handling for login walls
- the trace with model ID and prompt version on every event
- Pydantic validation with one repair attempt and safe fallbacks
- a non-root Docker image with no baked-in keys

## 8. External feedback: a biologist's review (2026-11-16)

A research biologist reviewed the project. Their points, paraphrased:
- Scientists go to the source and read the experiment design, details and graphs closely, so an AI agent checking an AI summary may not add much. Physicians might like it; they are not convinced scientists need it.
- For news, people can already write code to get a newsfeed. What makes this agent better?

How each point maps to this audit:

| Objection | What answers it | What currently undercuts the answer |
|---|---|---|
| "We read the source, design and graphs ourselves" | The target user is a clinician, not a bench scientist (section 1). The brief is triage: it says which sources changed since the anchor date and quotes the exact sentence behind every number, so the reader reaches the paper faster. | Tavily `extract` is text only, so figures, image tables and supplements are never read, and this limit isn't documented. Items record evidence level and population, but not sample size, comparator arm, primary endpoint or follow-up. |
| "AI checking AI isn't useful" | The main safety claim is the **deterministic** checks (section 4): numbers must appear verbatim in the cited sentence, and design and population words are matched against the excerpt. The LLM verifier is a second layer. | Findings **1** (best-of verdict), **4** (range and CI upper bounds unchecked), **5** (very short source sentences accepted) and **6** (negation ignored) are holes in exactly these guarantees. All four are fixed on `fix/verification-checks`. |
| "Why not a newsfeed?" | Anchor-date filtering, the tier allowlist, counting independent sources, PRELIMINARY vs peer-reviewed labels, the "Sources disagree" section, and a verbatim sentence beside every number. | Finding **3** (items dated before the anchor can appear as new) breaks the main thing that separates this from a feed, and **11** (many items undated) weakens it further. The evals have never run (**13**), so there is no measured comparison against a simple baseline. |

What to do about it (folded into section 9):
- Merge `fix/verification-checks` first. It closes 1, 3, 4, 5 and 6, which are the specific weak points this reviewer raised.
- Add a quality baseline next to the credit comparison. Run the 15 eval questions through a plain date-filtered search (news or PubMed) and compare known developments found, items wrongly labelled new, and number errors.
- State in the README use limits that figures, image tables and supplements are not read.
- Consider extracting design details per item: sample size, comparator, primary endpoint and follow-up. This is a prompt and schema change, so bump `PROMPT_VERSION`.
- In the pitch, lead with the deterministic checks and present the LLM verifier as secondary.

## 9. Recommended next steps (in order)

1. **Bring in `fix/verification-checks`:** rebase it onto `main`, resolve the 3 conflicts as described in section 6, and use a new `PROMPT_VERSION`. That closes findings 1–7 and 13.
2. **Fix 8–10:**
   - classify subdomains like `dailynews.` as `news`
   - on startup, mark tasks stuck in a running status for over N minutes as `failed` with an error, and show this in the UI
   - stop starting new rounds or steps when the time budget is spent, and document that the budget is approximate
3. **Run the live smoke test**, then `uv run python evals/run_evals.py` (about 165–200 credits and 2 hours), then `uv run python evals/compare.py --set-baseline`.
4. **Answer the reviewer feedback (section 8):**
   - run the same 15 questions through a plain date-filtered search baseline and compare quality, not only credits
   - add "figures, image tables and supplements are not read" to the README use limits
   - optionally extract per-item design details (sample size, comparator, primary endpoint, follow-up), with a new `PROMPT_VERSION`
5. **Add a GitHub Actions workflow** that runs `uv run pytest`.
6. **Pre-run the demo questions** into `research.db`, then deploy to a Nebius CPU VM with `docker compose up -d --build` (see README "Deploy to the cloud").

## 10. Commands

```bash
uv sync
uv run python scripts/list_models.py            # confirm model IDs (needs NEBIUS_API_KEY)
uv run pytest                                   # 102 mocked tests
uv run pytest -m live -s                        # live smoke test (credits)
uv run research "question" --yes                # CLI; --show ID, --export ris|csv
uv run streamlit run app/streamlit_app.py       # UI on :8501
uv run python evals/run_evals.py                # live evals
uv run python evals/compare.py --set-baseline   # accept a baseline; later runs gate on it
uv run python evals/credit_comparison.py        # credits vs traditional search for the latest task
docker compose up -d --build                    # container (needs APP_PASSWORD in .env)
```
