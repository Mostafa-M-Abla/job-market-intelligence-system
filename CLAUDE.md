# CLAUDE.md — Job Market Intelligence System v2

## Project Purpose
A portfolio-quality agentic AI system for job market analysis and resume gap identification.
Built with LangGraph to demonstrate: multi-task planning, dynamic intent routing, persistent caching,
human-in-the-loop (HITL), structured LLM outputs, and full-stack deployment.

## Tech Stack
- **LangGraph** — stateful graph orchestration (`StateGraph`, `interrupt()`, `AsyncSqliteSaver`)
- **LangChain / langchain-openai** — LLM abstractions, tool base classes
- **OpenAI** — `gpt-4o-mini` for classification/extraction, `gpt-4o` for analysis/generation
- **SerpAPI** — Google Jobs data collection
- **SQLite** — market analysis cache + conversation history
- **FastAPI** — backend API with SSE streaming; deployed to fly.io
- **sse-starlette** — Server-Sent Events support for FastAPI
- **aiosqlite** — async SQLite driver for `AsyncSqliteSaver` in the API layer
- **GitHub Pages** — frontend (personal_website/)

## Project Structure
```
graph/
  state.py              # JobMarketState TypedDict — shared state for all nodes
  graph.py              # StateGraph assembly + compile() with SqliteSaver
  routing.py            # All conditional edge routing functions (pure functions)
  session_store.py      # In-memory resume store (keyed by session_id)
  nodes/
    planner.py          # LLM structured output: decomposes message into 1–4 tasks
    task_dispatcher.py  # Pops next task from queue, sets intent/job_titles/country on state
    check_resume.py     # Deterministic: is PDF in session store?
    resume_parser.py    # PDF bytes -> text via pypdf
    cache_lookup.py     # SHA256 key -> SQLite cache check
    hitl.py             # HITL interrupt() nodes: confirm_search_params, confirm_report_format
    job_collector.py    # SerpAPI call via GoogleJobsCollectorTool; early-exits with message if 0 results
    requirements_extractor.py  # LLM batch extraction per job posting (gpt-4o-mini)
    market_analyzer.py  # LLM aggregation + writes to cache DB (gpt-4o)
    skill_gap_analyzer.py      # Resume vs. market comparison (gpt-4o)
    answer_general.py   # LLM direct answer for general_question intent (streaming)
    answer_focused.py   # Narrow question answering from market data
    html_report_generator.py   # Full HTML generation + save
    respond.py          # Combines accumulated_responses into final reply

api/
  main.py               # FastAPI app: CORS, static report serving, router registration
  dependencies.py       # Graph singleton (AsyncSqliteSaver); startup/shutdown lifecycle
  routers/
    chat.py             # POST /api/chat, POST /api/chat/{id}/reply — SSE streaming + HITL
    resume.py           # POST /api/chat/{id}/resume — PDF upload (max 10 MB)
    history.py          # GET /api/chat/{id}/history — conversation history

tools/
  google_jobs_tool.py   # LangChain BaseTool, SerpAPI, tenacity retries, async
  resume_pdf_tool.py    # Accepts pdf_bytes: bytes, uses pypdf
  html_report_saver.py  # Saves to outputs/{session_id}/, returns URL path
  market_cache_tool.py  # MarketCacheReadTool + MarketCacheWriteTool
  conversation_store.py # ConversationSaveTool + ConversationGetTool

db/
  connection.py         # sqlite3 connection (sync, for run_tests.py / script mode)
  models.py             # Table definitions
  migrations/
    001_initial_schema.sql

personal_website/       # GitHub Pages portfolio site (separate git repo)
  job-market-chat.html  # Live demo chat UI connecting to the fly.io API

tests/
  test_graph_routing.py # Pure routing unit tests (no LLM, no DB, no SerpAPI)

evaluation/
  eval_final_answer.py  # Eval 1: end-to-end answer quality (LLM-as-a-judge, 5 questions)
  eval_planner.py       # Eval 2: planner node isolation (8 examples, zero SerpAPI)
  eval_trajectory.py    # Eval 3: node sequence verification (3 questions)
  run_all.py            # Runs all 3 evals in sequence (cheapest first)
  evaluate_html_report.py  # Legacy: single HTML report quality check (@traceable)
  shared/
    graph_runner.py     # HITL-aware graph runner + trajectory capture
    dataset_utils.py    # Idempotent LangSmith dataset create/upsert
    llm_judge.py        # LLM-as-a-judge evaluators (relevance/correctness/completeness)

run_tests.py            # Non-interactive integration test runner (all 4 intents)
test_google_jobs_tool.py  # Interactive debug runner for GoogleJobsCollectorTool
                          # Saves all intermediates to debug_output/ for inspection
```

## The 4 Intents
| Intent | Description | HITL interrupts |
|---|---|---|
| `general_question` | Direct LLM answer, no tools | None |
| `focused_question` | Narrow market data question (cloud platforms, DBs, etc.) | confirm_search_params (cache miss only) |
| `full_market_analysis` | Comprehensive analysis, optional HTML report | confirm_search_params + confirm_report_format |
| `resume_analysis` | Resume gap vs. market, skill recommendations | confirm_search_params + confirm_report_format |

## Multi-Task Planner (Phase 3 addition)
The `planner` node replaces the old single-intent `intent_resolver`. It decomposes a user message
into an ordered list of 1–4 tasks. Each task maps to one of the 4 intents above.

**Single-task queries** behave identically to before — one task in the queue, same execution path.

**Compound queries** execute each task sequentially, collect answers in `accumulated_responses`,
and combine them in `respond` (one final LLM merge call when multiple answers exist).

Example:
```
"What skills do AI engineers need in Germany and explain what LangGraph is?"
→ planner → task_queue = [
    {intent: focused_question, job_titles: [AI Engineer], country: Germany},
    {intent: general_question, query: "explain LangGraph"}
  ]
→ task 1 executes full market pipeline (HITL confirm → SerpAPI → extract → analyze → answer_focused)
→ task 2 executes general LLM call (answer_general)
→ respond combines both answers
```

HITL gates are fully preserved — `confirm_search_params` still fires before any SerpAPI call.

**Anti-over-decomposition**: The planner prompt instructs the LLM:
- "What skills do AI and ML engineers need in Germany?" → 1 task (same market)
- "Analyse AI in Germany AND Data engineers in France" → 2 tasks (different markets)
- If in doubt, use 1 task

## Graph Topology
```
START → planner → task_dispatcher → [route_after_intent]
                        ↑                    │
                        └────────────────────┘
                     (loop if task_queue not empty)

route_after_intent:
  general_question     → answer_general  → [route_after_task_complete]
  resume_analysis      → check_resume    → resume_parser → cache_lookup
  focused/market       → cache_lookup    → confirm_search_params (HITL)
                                         → job_collector → [route_after_job_collector]
                                             ├─ 0 results → respond (or task_dispatcher if more tasks)
                                             └─ results   → requirements_extractor
                                         → market_analyzer → answer_focused / skill_gap_analyzer
                                         → confirm_report_format (HITL)
                                         → html_report_generator / answer_focused
                                         → [route_after_task_complete]

route_after_task_complete:
  task_queue not empty → task_dispatcher (loop)
  task_queue empty     → respond → END
```

## API Endpoints (Phase 2)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Start a new turn; returns SSE stream |
| `POST` | `/api/chat/{session_id}/reply` | Resume after a HITL interrupt; returns SSE stream |
| `POST` | `/api/chat/{session_id}/resume` | Upload a PDF resume (PDF only, ≤10 MB) |
| `GET` | `/api/chat/{session_id}/history` | Fetch past messages (`?limit=50`) |
| `GET` | `/api/reports/{session_id}/{file}` | Serve generated HTML reports (static mount) |

### SSE Event Types
| Event | Payload | When |
|---|---|---|
| `node_start` | `{"node": "<name>"}` | Each graph node begins |
| `intent_selected` | `{"intent": "<intent>"}` | After task_dispatcher completes; used to highlight the chosen intent node in the frontend diagram |
| `token` | `{"content": "<text>"}` | LLM streaming token (answer_general/respond/analyzer/focused nodes) |
| `interrupt` | `{"prompt": "...", "session_id": "..."}` | Graph paused at HITL |
| `done` | `{"session_id", "final_text_response", "html_report_path", "intent", "cache_hit"}` | Graph completed |
| `error` | `{"detail": "<message>"}` | Unhandled exception |

## HITL Pattern
```python
# In a node: pause and surface prompt to caller
user_reply: str = interrupt("Confirm search params: ...")

# API layer: resume after user responds
graph.invoke(Command(resume=user_reply), config)

# Detect if paused:
snapshot = graph.get_state(config)
is_interrupted = bool(snapshot.next)
```

### HITL Frontend UI (job-market-chat.html)
The `showInterrupt(prompt)` function detects which UI to render based on the prompt text:
- `prompt.includes('**confirm**')` → shows green "Confirm" button + text input for corrections (`#hitl-confirm-wrap`)
- `prompt.includes('Reply A or B')` → shows A/B buttons for report format choice (`#hitl-ab-wrap`)
- Otherwise → shows plain text input (`#hitl-text-wrap`)

## Cache Key
```python
SHA256(sorted(job_titles) + country.lower() + today's date)
```
Same job titles + country + same day = cache hit. Cache TTL = 7 days (configurable via `CACHE_TTL_DAYS`).
Cache is disabled in dev via `DISABLE_CACHE = True` in `config.py`.

## Running Tests
```bash
# Unit tests (no API keys needed)
python -m pytest tests/test_graph_routing.py -v

# Integration tests (needs OPENAI_API_KEY + SERPAPI_API_KEY in .env)
python run_tests.py general       # LLM only, cheapest
python run_tests.py focused       # SerpAPI + HITL
python run_tests.py market        # Full pipeline + 2x HITL
python run_tests.py report        # Full pipeline + HTML report generation
python run_tests.py cache         # Verify cache hit (run after market or focused)
python run_tests.py resume_missing
python run_tests.py resume        # Needs Resume.pdf in project root
python run_tests.py all           # All tests

# Debug GoogleJobsCollectorTool interactively (saves all intermediates to debug_output/)
python test_google_jobs_tool.py
python test_google_jobs_tool.py --titles "AI Engineer" --country Germany --limit 5
```

## LangSmith Evaluations (Phase 4)

Three evaluation types, all stored as datasets in LangSmith and viewable in the UI.

```bash
# Run all 3 evals (cheapest first — fails fast if planner is broken)
python evaluation/run_all.py

# Run individually
python evaluation/eval_planner.py       # Eval 2: planner node, OpenAI only, fastest
python evaluation/eval_trajectory.py    # Eval 3: node sequence, 1 example needs SerpAPI
python evaluation/eval_final_answer.py  # Eval 1: answer quality, 2 examples need SerpAPI
```

Requires `LANGSMITH_API_KEY` in `.env` with write access to the `job-market-intelligence-system` project.

### Eval 1 — Final Answer Quality (`eval-final-answer-v1`, 5 examples)
Runs the full graph end-to-end. LLM-as-a-judge (gpt-4o-mini) scores each response on:
- `relevance` — does the answer address the question?
- `correctness` — are facts and claims accurate?
- `completeness` — does it cover all key aspects?

3 `general_question` examples (no SerpAPI), 2 `focused_question` (SerpAPI + auto-HITL).

### Eval 2 — Planner Node Isolation (`eval-planner-v1`, 8 examples)
Calls `planner(state)` directly — no graph, no SerpAPI, no DB. Deterministic evaluators:
- `task_count_correct` — exact match on number of tasks produced
- `intent_correct` — all tasks have the correct intent (order-insensitive)
- `no_over_decomposition` — single-market queries must produce exactly 1 task
- `job_titles_extracted` — job titles present when expected
- `country_extracted` — country field correctly identified

Covers all 4 intents, anti-over-decomposition cases, and multi-task queries (2 distinct markets, general+focused).

### Eval 3 — Trajectory (`eval-trajectory-v1`, 3 examples)
Captures node execution order via `graph.stream(stream_mode="updates")`. Evaluators:
- `trajectory_exact_match` — full sequence must match exactly (1.0 or 0.0)
- `trajectory_prefix_match` — greedy subsequence score (partial credit)

3 trajectories: simple `general_question` (4 nodes), `focused_question` with HITL (9 nodes), compound `general+general` with task loop (6 nodes, `task_dispatcher` fires twice).

**HITL in evals:** All interrupts are auto-confirmed — `confirm_search_params` → replies `"confirm"`, `confirm_report_format` → replies `"B"` (text summary, avoids file generation). Each eval example uses an isolated in-memory SQLite checkpointer to prevent state bleed.

**Trajectory + HITL note:** `confirm_search_params` executes fully before `interrupt()` pauses the stream, so it appears in the trajectory from the first `graph.stream()` call. Post-resume nodes (`job_collector` onward) appear in the second call.

## Environment Variables
```
OPENAI_API_KEY=          # Required
SERPAPI_API_KEY=         # Required for market data tests
LANGSMITH_API_KEY=       # Required for LangSmith evaluations (Phase 4)
LANGSMITH_PROJECT=       # LangSmith project name (default: job-market-intelligence-system)
LANGSMITH_TRACING=       # true/false — enable LangSmith tracing
LANGCHAIN_TRACING_V2=    # true/false — legacy tracing flag (keep false if using LANGSMITH_TRACING)
LANGCHAIN_PROJECT=       # LangSmith project name (legacy alias)
CACHE_TTL_DAYS=7         # Market analysis cache TTL
DEFAULT_TOTAL_POSTS=5    # Default job postings to collect per run (low for dev; use 30 for prod)
DB_PATH=job_market.db    # SQLite file path (dev)
CORS_ORIGINS=*           # Comma-separated allowed origins (default * for local dev; restrict in prod)
OUTPUTS_DIR=outputs      # Directory where HTML reports are saved and served from
```

## Frontend (personal_website/job-market-chat.html)

### Page Layout (top to bottom)
1. Navbar
2. Header card — title, description, feature badges (4 Intents / HITL / Real-time Streaming)
3. Chat panel — messages, status bar, HITL card, input
4. Architecture diagram — LangGraph SVG visualisation
5. Back to Projects button
6. Footer

### Status Bar
Sits between the messages area and the input section. Shows the currently executing pipeline step
as `"Step N: <label>..."`. A `▾` button expands a history list of completed steps (with `✓` marks).
- Cleared (hidden text) when a new message is sent (`resetPipeline()`)
- Updated on each `node_start` SSE event
- Spinner dims on `interrupt` (pipeline waiting for user)
- Clears on `done` or `error`

### SSE Parsing — Important
sse-starlette sends events with `\r\n` line endings and `\r\n\r\n` event separators.
The frontend splits on `/\r?\n\r?\n/` (not `'\n\n'`) to correctly parse individual events.
Using `'\n\n'` causes all events in a TCP chunk to merge into one, with only the last
`event:` / `data:` pair surviving.

### Streaming Markdown
Tokens are rendered through `marked.parse()` on every `token` event (`innerHTML = renderMd(raw)`),
not appended as plain text. This prevents raw markdown syntax from being visible during streaming.

### Architecture Diagram
SVG LangGraph visualisation with live node highlighting as a query runs:
- Processing nodes: blue when active, green when done
- Intent nodes (`gnode-intent`): purple by default, green when active/done
- HITL nodes: amber
- The chosen intent node is highlighted via the `intent_selected` SSE event (emitted by the
  backend on `task_dispatcher` `on_chain_end`); `general_question` uses the `answer_general`
  node directly (no separate label node needed)
- To change `total_posts` sent by the frontend: edit `total_posts: 5` in the `fetch` call
  inside `handleSend()` in `job-market-chat.html`

## Implementation Phases
| Phase | Status | Description |
|---|---|---|
| 1 | **Complete** | Core LangGraph graph + tools, SQLite cache, HITL via terminal |
| 2 | **Complete** | FastAPI backend, SSE streaming, HITL over HTTP, fly.io deployment |
| 3 | **Complete** | GitHub Pages frontend, resume upload, multi-task planner, bug fixes |
| 3.5 | **Complete** | Frontend UX polish: step indicator, streaming markdown, 0-results handling, intent highlighting |
| 4 | **Complete** | LangSmith evaluation suite: final answer quality, planner node isolation, trajectory verification |

## Key Design Decisions
- **Bounded multi-task planner** — `planner` decomposes messages into 1–4 tasks using only the 4 known intents; it cannot invent new actions, keeping the system predictable and debuggable
- **task_dispatcher resets per-task state** — clears `raw_job_postings`, `extracted_requirements`, `market_analysis_markdown`, etc. before each task so one task's data never bleeds into the next
- **Sync nodes throughout** — all graph node functions are synchronous (`def`, not `async def`) using `llm.invoke()`. LangGraph's `astream_events()` still emits token-level streaming events because `streaming=True` on the LLM is handled at the LangChain level. This keeps nodes compatible with both `graph.invoke()` (run_tests.py) and `graph.astream_events()` (FastAPI)
- **Short job IDs to LLM** — `requirements_extractor` sends `job_1, job_2, ...` instead of the full SerpAPI base64 job_id. The real IDs are very long and caused the LLM to truncate them with `...`, producing invalid JSON that silently dropped all extracted skills
- **Markdown fence stripping in requirements_extractor** — gpt-4o-mini often wraps JSON in ` ```json ``` ` fences despite instructions not to; the extractor now strips fences before `json.loads()`
- **Lazy tool instantiation** — tools are initialized on first use, not at import time
- **Global market cache** — cache is shared across all sessions (market data is not user-specific); only conversation history and resumes are per-session
- **`gpt-4o-mini` for classification/extraction** — cheaper; `gpt-4o` only for analysis + HTML generation where quality matters
- **WAL mode on SQLite** — enabled via `PRAGMA journal_mode=WAL` for better concurrent read performance
- **AsyncSqliteSaver in API, SqliteSaver in tests** — `api/dependencies.py` uses `AsyncSqliteSaver` (aiosqlite) so `astream_events`/`aget_state` work inside FastAPI's async loop; `run_tests.py` uses the sync `SqliteSaver` via `graph.invoke()`
- **Conversation history saving skipped in respond.py** — `ConversationSaveTool` opens a sync sqlite3 connection via `asyncio.to_thread()`, which conflicts with `AsyncSqliteSaver`'s aiosqlite connection and causes `aget_state()` to hang; history persistence will be addressed separately
- **DISABLE_CACHE flag** — set in `config.py`; when `True`, every request runs the full collection pipeline. Useful during debugging
- **0-results short-circuit** — `job_collector` returns early with a user-facing message if SerpAPI finds no postings; `route_after_job_collector` skips the analysis pipeline and routes directly to `respond` (or `task_dispatcher` if more tasks remain)
- **intent_selected SSE event** — emitted on `task_dispatcher` `on_chain_end` (not `on_chain_start`) because the intent is only set in the node's output, not its input; the frontend uses this to highlight the correct intent label node in the architecture diagram
