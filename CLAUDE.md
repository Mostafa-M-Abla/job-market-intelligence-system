# CLAUDE.md — Job Market Intelligence System v2

## Project Purpose
A portfolio-quality agentic AI system for job market analysis and resume gap identification.
Built with LangGraph to demonstrate: dynamic intent routing, persistent caching, human-in-the-loop (HITL),
structured LLM outputs, and full-stack deployment.

## Tech Stack
- **LangGraph** — stateful graph orchestration (`StateGraph`, `interrupt()`, `SqliteSaver`)
- **LangChain / langchain-openai** — LLM abstractions, tool base classes
- **OpenAI** — `gpt-4o-mini` for classification/extraction, `gpt-4o` for analysis/generation
- **SerpAPI** — Google Jobs data collection
- **SQLite** (dev) / **PostgreSQL** (prod) — market analysis cache + conversation history
- **FastAPI** — backend API (Phase 2)
- **GitHub Pages** — frontend (Phase 3)

## Project Structure
```
graph/
  state.py              # JobMarketState TypedDict — shared state for all nodes
  graph.py              # StateGraph assembly + compile() with SqliteSaver
  routing.py            # All conditional edge routing functions (pure functions)
  session_store.py      # In-memory resume store (Phase 1); replaced by DB in Phase 2
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
    respond.py          # Final AIMessage composition + conversation history write

tools/
  google_jobs_tool.py   # LangChain BaseTool, SerpAPI, tenacity retries, async
  resume_pdf_tool.py    # Accepts pdf_bytes: bytes, uses pypdf
  html_report_saver.py  # Saves to outputs/{session_id}/, returns URL path
  market_cache_tool.py  # MarketCacheReadTool + MarketCacheWriteTool
  conversation_store.py # ConversationSaveTool + ConversationGetTool

db/
  connection.py         # sqlite3 connection (Phase 1); SQLAlchemy async in Phase 2
  models.py             # Table definitions
  migrations/
    001_initial_schema.sql

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
DATABASE_URL=            # Postgres URI for production (Phase 2); SQLite if blank
CACHE_TTL_DAYS=7         # Market analysis cache TTL
DEFAULT_TOTAL_POSTS=30   # Default job postings to collect per run
DB_PATH=job_market.db    # SQLite file path (dev)
```

## Implementation Phases
| Phase | Status | Description |
|---|---|---|
| 1 | **Complete** | Core LangGraph graph + tools, SQLite cache, HITL via terminal |
| 2 | Pending | FastAPI backend, SSE streaming, fly.io deployment |
| 3 | Pending | GitHub Pages frontend, resume upload, inline report viewer |
| 4 | Pending | LangSmith evaluation, README polish |

## Key Design Decisions
- **Lazy tool instantiation** — tools are initialized on first use, not at import time, so the graph can be imported without API keys present
- **Global market cache** — cache is shared across all sessions (market data is not user-specific); only conversation history and resumes are per-session
- **`total_posts=5` in tests** — keeps API credit usage low during testing; default 30 for real use
- **`gpt-4o-mini` for classification/extraction** — cheaper; `gpt-4o` only for analysis + HTML generation where quality matters
- **WAL mode on SQLite** — enabled via `PRAGMA journal_mode=WAL` for better concurrent read performance
