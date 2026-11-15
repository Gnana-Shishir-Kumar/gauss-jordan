# Research Agent

Turns a question into a **sourced, verifiable answer**. Every claim in the answer cites specific evidence excerpts from live web sources, and a second model checks each claim against each cited excerpt. When the evidence doesn't hold up, the agent says so instead of guessing.

- **LLM inference:** [Nebius Token Factory](https://tokenfactory.nebius.com) (OpenAI-compatible API) for planning, writing and verification
- **Live web search:** [Tavily](https://tavily.com) `search` to find sources, `extract` to pull query-relevant page content

## Who it's for

People who need an answer they can check, not just one that sounds right: analysts, journalists, students, and engineers doing due diligence. Each claim in the answer shows:

- whether it is a **fact**, an **inference**, or **uncertain**
- which source excerpts it relies on (with clickable links)
- whether a verifier found each excerpt **supported**, **partial**, **unsupported**, or **source_unavailable**

The full trace (queries, sources, dropped items and why, latency, tokens) is stored in SQLite.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # then fill in NEBIUS_API_KEY and TAVILY_API_KEY
```

Confirm the configured model IDs exist on your Token Factory account:

```bash
uv run python scripts/list_models.py
```

It lists all available models and marks each configured model `OK` or `NOT FOUND`, with similar IDs suggested. If one is missing, change it in `src/research_agent/config.py` or override it in `.env` (`RESEARCH_PLANNER_MODEL`, `RESEARCH_WRITER_MODEL`, `RESEARCH_VERIFIER_MODEL`).

## Run

```bash
uv run research "How did global EV sales change in 2025, and which markets drove it?"
```

The CLI prints live progress on stderr, then the answer, citations with verdicts, and the trace. Options:

- `--show TASK_ID` reprints a stored run
- `--no-trace` omits the trace
- `--quiet` hides live progress
- `--db PATH` uses another SQLite file

Exit code: 0 = done, 3 = insufficient evidence, 1 = failed.

Web UI:

```bash
uv run streamlit run app/streamlit_app.py
```

It has a question box, live step-by-step progress, claims with verdict badges, clickable sources, an expandable trace (steps, sub-queries, evidence including dropped items) and recent runs in the sidebar.

Tests:

```bash
uv run pytest            # mocked unit tests, no network
uv run pytest -m live -s # optional live smoke test (skipped without keys)
```

## How it works

```
question
   │
   ▼
1. PLAN ─────────── planner model → 3–5 independent sub-queries (JSON, Pydantic-validated)
   │
   ▼
2. RESEARCH ─────── per sub-query, concurrently (asyncio.gather + semaphore):
   │                Tavily search (≤5 results) → Tavily extract (query-relevant chunks)
   │                failures recorded; extraction failure falls back to search snippets
   ▼
3. EVIDENCE ─────── normalize URL, pick best excerpt, score relevance,
   │                dedupe by URL and content hash; dropped items kept with drop_reason
   ▼
4. FOLLOW-UP ────── sub-queries with < 2 kept items get rewritten queries; one round max, then stop
   │
   ▼
5. WRITE ────────── writer model → claims tagged fact / inference / uncertain, each citing evidence IDs.
   │                Web text is inside <untrusted_source> blocks, marked as data, never instructions.
   │                No usable evidence → insufficient_evidence, no answer is invented.
   ▼
6. VERIFY ───────── verifier model, one call per claim → a verdict for every claim–citation pair
                    (supported | partial | unsupported | source_unavailable)
```

The task's status moves `planning → researching → writing → verifying → done`. It ends as `insufficient_evidence` when:

- no usable evidence is found, or
- the writer reports the evidence is insufficient, or
- no claim is supported, even partially, by its cited excerpt.

It ends as `failed` (with `error` set) only when a step that the run cannot continue without (planning or writing) fails.

### Reliability rules

- **Hard limits** live in `config.py`: max 5 sub-queries, max 1 follow-up round, max 5 results per search, per-request timeouts, and capped evidence sent to the writer.
- **Retries:** every network call goes through `retry.with_retry`, which uses exponential backoff with jitter for transient errors (timeouts, connection errors, 429, 5xx). Non-transient errors such as auth failures fail immediately.
- **Failed searches don't stop the run.** A failed search or extraction is recorded in `trace_events` and on the sub-query, and the run continues.
- **No silent unsupported output:**
  - A citation to evidence that was dropped or never shown to the writer is marked `source_unavailable`.
  - A citation to a nonexistent ID is removed, and a fact left with no valid citation is downgraded to `uncertain`.
  - A citation the verifier skipped, or one whose verification call failed, is marked `unsupported` with a note saying why.
- **Malformed LLM output:** if output fails Pydantic validation, the model gets one repair attempt with the validation errors. If a model rejects JSON mode, the wrapper falls back to prompt-only JSON.
- **Prompt-injection guard:** any `<untrusted_source>` tags inside retrieved text are removed before wrapping, so a page can't close its block and inject instructions.
- **Crash-safe trace:** SQLite is written after every step (WAL mode), so a crash leaves a full trace.

### Code layout

```
src/research_agent/
  config.py       models, limits, timeouts, retry policy; stdlib .env loader
  schemas.py      Pydantic models for every LLM output
  llm.py          Token Factory wrapper (AsyncOpenAI): JSON mode, validation + repair, usage
  retry.py        backoff + transient-error classification
  search/base.py  SearchBackend protocol, the only thing the engine knows about search
  search/tavily.py Tavily implementation (AsyncTavilyClient search + extract)
  evidence.py     pure functions: URL normalization, hashing, excerpt choice, scoring, dedupe
  prompts.py      planner / writer / verifier prompts, untrusted-data wrapping
  planner.py, researcher.py, writer.py, verifier.py   one module per pipeline step
  context.py      per-run context: DB handle, clients, trace + event emission
  pipeline.py     run_research(): orchestration, plain async, no UI code
  storage.py      SQLite schema and queries (stdlib sqlite3)
  render.py       text rendering for CLI / stored final answer
  cli.py          `research` entry point
app/streamlit_app.py   UI only; calls run_research with an on_event callback
scripts/list_models.py lists Token Factory models and checks configured IDs
```

The engine is plain async Python. `run_research(question, llm=..., search=..., conn=..., settings=..., on_event=...)` takes its dependencies as arguments, so a FastAPI endpoint can call it unchanged. To add another search source, implement `SearchBackend` (`search`, `extract`) and pass it in; the planner and the other steps don't change.

### Storage

There are seven tables:

| Table | Holds |
|---|---|
| `tasks` | the question, status, final answer, error |
| `sub_queries` | each query with its round and status (`done`, `weak`, `empty`, `failed`) |
| `sources` | unique by normalized URL, with content hash |
| `evidence` | excerpt, relevance score, `kept`, `drop_reason` |
| `claims` | claim text and claim type |
| `claim_citations` | a verdict and verifier note per claim–evidence pair |
| `trace_events` | step, detail JSON, latency, input/output tokens |

Inspect a run with any SQLite client, or `uv run research --show <id>`.

## Models and integrations

| Role | Default model | Why |
|---|---|---|
| Planner | `deepseek-ai/DeepSeek-V4-Flash` | fast, cheap, good at structured JSON |
| Writer | `NousResearch/Hermes-4-405B` | strong long-context synthesis over many excerpts |
| Verifier | `deepseek-ai/DeepSeek-V4-Flash` | many small checks, run concurrently |

- **Token Factory:** reached with the `openai` SDK (`AsyncOpenAI(base_url="https://api.tokenfactory.nebius.com/v1")`). The SDK's own retries are disabled so `retry.py` is the single retry policy. Token usage per call is recorded in `trace_events`.
- **Tavily:**
  - `search` uses `search_depth="advanced"` and `max_results=5`.
  - `extract` is called once per sub-query with the sub-query as `query` and `chunks_per_source=3`, so it returns the chunks of each page most relevant to that sub-query. The agent then picks one bounded excerpt (≤1500 chars) per page and that excerpt is what gets cited and verified.

Keys are read only from `NEBIUS_API_KEY` and `TAVILY_API_KEY` (environment or `.env`). They are never logged, printed, or included in `repr(Settings)`.

## Sample research trace

> Mocked run, shown to illustrate the format. Replace with a live run: `uv run research "<question>"`.

```
Question: What are the topic facts?
Status:   done

ANSWER
------
1. [fact] Claim from E1  (E1: supported)
2. [fact] Claim from E2  (E2: supported)
3. [fact] Claim from E3  (E3: supported)

CITATIONS
---------
Claim 1 [fact] -> supported
  E1 supported          https://alpha0.example.com/page
      stated
...

TRACE
-----
  [start    ] Task 1: What are the topic facts?
  [plan     ] Planned 3 sub-queries  (5ms, 100->20 tok)
  [search   ] 2 results for 'alpha topic facts'  (0ms)
  [extract  ] Extracted 2/2 pages for 'alpha topic facts'  (0ms)
  [search   ] 2 results for 'beta topic facts'  (0ms)
  [extract  ] Extracted 2/2 pages for 'beta topic facts'  (0ms)
  [search   ] 0 results for 'gamma topic facts'  (0ms)
  [evidence ] Kept 2/2 evidence items for 'alpha topic facts'
  [evidence ] Kept 2/2 evidence items for 'beta topic facts'
  [follow_up] Round 2: 1 follow-up queries for 1 weak sub-queries  (5ms, 100->20 tok)
  [search   ] 2 results for 'gamma topic facts alternative'  (0ms)
  [extract  ] Extracted 2/2 pages for 'gamma topic facts alternative'  (0ms)
  [evidence ] Kept 2/2 evidence items for 'gamma topic facts alternative'
  [write    ] Drafted 3 claims (sufficient=True)  (5ms, 100->20 tok)
  [verify   ] Claim 1: 1 supported  (5ms, 100->20 tok)
  [verify   ] Claim 2: 1 supported  (5ms, 100->20 tok)
  [verify   ] Claim 3: 1 supported  (5ms, 100->20 tok)
  [done     ] Finished: done
  totals: 4 sub-queries, 6/6 evidence kept, 600 input / 120 output tokens
```

In this run the `gamma` sub-query found nothing, so one follow-up round rewrote it, found sources, and then stopped.
