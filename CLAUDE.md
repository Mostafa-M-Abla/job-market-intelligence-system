# CLAUDE.md — Job Market Intelligence System v2

## Project Purpose
A portfolio-quality agentic AI system for job market analysis and resume gap identification.
Built with LangGraph to demonstrate: dynamic intent routing, persistent caching, human-in-the-loop (HITL),
structured LLM outputs, and full-stack deployment.

## Tech Stack
- **LangGraph** — stateful graph orchestration (`StateGraph`, `interrupt()`, `AsyncSqliteSaver`)
- **LangChain / langchain-openai** — LLM abstractions, tool base classes
- **OpenAI** — `gpt-4o-mini` for classification/extraction, `gpt-4o` for analysis/generation
- **SerpAPI** — Google Jobs data collection
- **SQLite** (dev) / **PostgreSQL** (prod) — market analysis cache + conversation history
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
    intent_resolver.py  # LLM structured output: classifies into 4 intents
    check_resume.py     # Deterministic: is PDF in session store?
    resume_parser.py    # PDF bytes -> text via pypdf
    cache_lookup.py     # SHA256 key -> SQLite cache check
    hitl.py             # HITL interrupt() nodes: confirm_search_params, confirm_report_format
    job_collector.py    # SerpAPI call via GoogleJobsCollectorTool
    requirements_extractor.py  # LLM batch extraction per job posting (gpt-4o-mini)
    market_analyzer.py  # LLM aggregation + writes to cache DB (gpt-4o)
    skill_gap_analyzer.py      # Resume vs. market comparison (gpt-4o)
    answer_focused.py   # Narrow question answering from market data
    html_report_generator.py   # Full HTML generation + save
    respond.py          # Final AIMessage composition (history saving currently skipped)

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
  test_graph_routing.py # 21 pure routing unit tests (no LLM, no DB, no SerpAPI)

run_tests.py            # Non-interactive integration test runner (all 4 intents)
```

## The 4 Intents
| Intent | Description | HITL interrupts |
|---|---|---|
| `general_question` | Direct LLM answer, no tools | None |
| `focused_question` | Narrow market data question (cloud platforms, DBs, etc.) | confirm_search_params (cache miss only) |
| `full_market_analysis` | Comprehensive analysis, optional HTML report | confirm_search_params + confirm_report_format |
| `resume_analysis` | Resume gap vs. market, skill recommendations | confirm_search_params + confirm_report_format |

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
| `token` | `{"content": "<text>"}` | LLM streaming token (respond/analyzer/focused nodes only) |
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

## Cache Key
```python
SHA256(sorted(job_titles) + country.lower() + today's date)
```
Same job titles + country + same day = cache hit. Cache TTL = 7 days (configurable via `CACHE_TTL_DAYS`).

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
```

## Environment Variables
```
OPENAI_API_KEY=          # Required
SERPAPI_API_KEY=         # Required for market data tests
LANGCHAIN_TRACING_V2=    # true/false — LangSmith tracing (Phase 4)
LANGCHAIN_API_KEY=       # LangSmith key (Phase 4)
LANGCHAIN_PROJECT=       # LangSmith project name
DATABASE_URL=            # Postgres URI for production; SQLite if blank
CACHE_TTL_DAYS=7         # Market analysis cache TTL
DEFAULT_TOTAL_POSTS=30   # Default job postings to collect per run
DB_PATH=job_market.db    # SQLite file path (dev)
CORS_ORIGINS=*           # Comma-separated allowed origins (default * for local dev; restrict in prod)
OUTPUTS_DIR=outputs      # Directory where HTML reports are saved and served from
```

## Implementation Phases
| Phase | Status | Description |
|---|---|---|
| 1 | **Complete** | Core LangGraph graph + tools, SQLite cache, HITL via terminal |
| 2 | **Complete** | FastAPI backend, SSE streaming, HITL over HTTP, fly.io deployment |
| 3 | **In Progress** | GitHub Pages frontend (`personal_website/`), resume upload, inline report viewer |
| 4 | Pending | LangSmith evaluation, README polish |

## Key Design Decisions
- **Lazy tool instantiation** — tools are initialized on first use, not at import time, so the graph can be imported without API keys present
- **Global market cache** — cache is shared across all sessions (market data is not user-specific); only conversation history and resumes are per-session
- **`total_posts=5` in tests** — keeps API credit usage low during testing; default 30 for real use
- **`gpt-4o-mini` for classification/extraction** — cheaper; `gpt-4o` only for analysis + HTML generation where quality matters
- **WAL mode on SQLite** — enabled via `PRAGMA journal_mode=WAL` for better concurrent read performance
- **AsyncSqliteSaver in API, SqliteSaver in tests** — `api/dependencies.py` uses `AsyncSqliteSaver` (aiosqlite) so `astream_events`/`aget_state` work inside FastAPI's async loop; `run_tests.py` uses the sync `SqliteSaver` via `graph.invoke()`
- **Conversation history saving skipped in respond.py** — `ConversationSaveTool` opens a sync sqlite3 connection via `asyncio.to_thread()`, which conflicts with `AsyncSqliteSaver`'s aiosqlite connection and causes `aget_state()` to hang; history persistence will be addressed separately
