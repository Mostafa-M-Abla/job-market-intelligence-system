"""
graph/graph.py — Assembles and compiles the LangGraph StateGraph.

This is the "wiring" file — it does not contain any business logic.
Its only job is to:
  1. Create a StateGraph and register every node.
  2. Connect nodes with edges (fixed and conditional).
  3. Attach a checkpointer so state is persisted to SQLite between turns.
  4. Compile and return the runnable graph.

How to use:
    from graph.graph import build_graph

    graph = build_graph()
    config = {"configurable": {"thread_id": "my-session-id"}}

    # First turn
    result = graph.invoke({"messages": [HumanMessage("...")], ...}, config)

    # Resume after a HITL interrupt
    from langgraph.types import Command
    result = graph.invoke(Command(resume="confirm"), config)
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

# ── Node imports ──────────────────────────────────────────────────────────────
# Each node is a plain Python function defined in its own file under graph/nodes/.
# A "node" in LangGraph is simply a function that receives the current state
# and returns a dict of fields to update.
from graph.nodes.answer_focused import answer_focused
from graph.nodes.cache_lookup import cache_lookup
from graph.nodes.check_resume import check_resume
from graph.nodes.hitl import confirm_report_format, confirm_search_params
from graph.nodes.html_report_generator import html_report_generator
from graph.nodes.intent_resolver import intent_resolver
from graph.nodes.job_collector import job_collector
from graph.nodes.market_analyzer import market_analyzer
from graph.nodes.requirements_extractor import requirements_extractor
from graph.nodes.respond import respond
from graph.nodes.resume_parser import resume_parser
from graph.nodes.skill_gap_analyzer import skill_gap_analyzer

# ── Routing function imports ───────────────────────────────────────────────────
# Routing functions are pure functions that look at the current state and
# return a string (the name of the next node to run).
from graph.routing import (
    route_after_cache_lookup,
    route_after_check_resume,
    route_after_confirm_report_format,
    route_after_confirm_search_params,
    route_after_intent,
    route_after_market_analyzer,
    route_after_resume_parser,
    route_after_skill_gap,
)
from graph.state import JobMarketState

# Path to the SQLite database file (overridable via DB_PATH env var).
DB_PATH = os.getenv("DB_PATH", "job_market.db")


def build_graph(checkpointer=None):
    """
    Build and compile the full Job Market Intelligence StateGraph.

    The graph is compiled once at startup and reused for every user request.
    State isolation between users is handled by the `thread_id` field in the
    LangGraph config dict — each session ID maps to its own independent state.

    Args:
        checkpointer: A LangGraph checkpointer that saves state between graph
                      invocations.  If None, a SqliteSaver backed by DB_PATH
                      is created automatically.

                      The checkpointer is what enables:
                        - Pausing at HITL interrupts and resuming later.
                        - Surviving server restarts mid-conversation.
                        - Per-user state isolation via thread_id.

    Returns:
        A compiled LangGraph graph ready to call with .invoke() or .astream_events().
    """
    # StateGraph is LangGraph's main graph type.  We pass in our state schema
    # so LangGraph knows which fields exist and how to merge updates.
    builder = StateGraph(JobMarketState)

    # ── Register nodes ────────────────────────────────────────────────────────
    # Each call says "when this node name is reached, call this function".
    # The function signature is always: (state: JobMarketState) -> dict
    builder.add_node("intent_resolver", intent_resolver)           # Classify user intent
    builder.add_node("check_resume", check_resume)                 # Is a resume uploaded?
    builder.add_node("resume_parser", resume_parser)               # PDF -> text
    builder.add_node("cache_lookup", cache_lookup)                 # Check market data cache
    builder.add_node("confirm_search_params", confirm_search_params)  # HITL: confirm job search params
    builder.add_node("job_collector", job_collector)               # Fetch jobs from SerpAPI
    builder.add_node("requirements_extractor", requirements_extractor)  # LLM: extract skills per job
    builder.add_node("market_analyzer", market_analyzer)           # LLM: aggregate + write cache
    builder.add_node("skill_gap_analyzer", skill_gap_analyzer)     # LLM: resume vs. market
    builder.add_node("answer_focused", answer_focused)             # LLM: answer specific question
    builder.add_node("confirm_report_format", confirm_report_format)  # HITL: HTML or text?
    builder.add_node("html_report_generator", html_report_generator)  # LLM: generate HTML report
    builder.add_node("respond", respond)                           # Compose final reply + save history

    # ── Entry point ───────────────────────────────────────────────────────────
    # Every conversation starts at intent_resolver — no exceptions.
    builder.add_edge(START, "intent_resolver")

    # ── Conditional edges ─────────────────────────────────────────────────────
    # add_conditional_edges(source_node, routing_function, path_map)
    # The routing_function returns a string key; the path_map maps that key
    # to the actual node name to run next.

    # After classifying intent: route to the appropriate major path
    builder.add_conditional_edges(
        "intent_resolver",
        route_after_intent,
        {
            "respond": "respond",           # general_question — straight to answer
            "check_resume": "check_resume", # resume_analysis — check PDF exists first
            "cache_lookup": "cache_lookup", # full_market / focused — check cache first
        },
    )

    # After checking for a resume: proceed or ask user to upload
    builder.add_conditional_edges(
        "check_resume",
        route_after_check_resume,
        {
            "resume_parser": "resume_parser",  # Resume found — extract text
            "respond": "respond",              # Resume missing — prompt user to upload
        },
    )

    # After parsing the resume: proceed to cache check, or bail on error
    builder.add_conditional_edges(
        "resume_parser",
        route_after_resume_parser,
        {
            "cache_lookup": "cache_lookup",  # Parsed OK — check if market data is cached
            "respond": "respond",            # Parse failed — surface error message
        },
    )

    # After cache lookup: use cached data or start a fresh search
    builder.add_conditional_edges(
        "cache_lookup",
        route_after_cache_lookup,
        {
            "answer_focused": "answer_focused",             # Cache hit + focused_question
            "skill_gap_analyzer": "skill_gap_analyzer",    # Cache hit + resume_analysis
            "confirm_report_format": "confirm_report_format",  # Cache hit + full_market
            "confirm_search_params": "confirm_search_params",  # Cache miss — ask user first
        },
    )

    # After HITL search param confirmation: run the search or re-classify
    builder.add_conditional_edges(
        "confirm_search_params",
        route_after_confirm_search_params,
        {
            "job_collector": "job_collector",       # User confirmed — start collecting jobs
            "intent_resolver": "intent_resolver",   # User edited params — re-parse
        },
    )

    # The job collection pipeline is always sequential — no branching here
    builder.add_edge("job_collector", "requirements_extractor")
    builder.add_edge("requirements_extractor", "market_analyzer")

    # After market analysis: fan out to the correct next step by intent
    builder.add_conditional_edges(
        "market_analyzer",
        route_after_market_analyzer,
        {
            "answer_focused": "answer_focused",             # focused_question
            "skill_gap_analyzer": "skill_gap_analyzer",    # resume_analysis
            "confirm_report_format": "confirm_report_format",  # full_market_analysis
        },
    )

    # After skill gap analysis: always ask the user about output format
    builder.add_conditional_edges(
        "skill_gap_analyzer",
        route_after_skill_gap,
        {"confirm_report_format": "confirm_report_format"},
    )

    # After HITL report format choice: generate HTML or write a text answer
    builder.add_conditional_edges(
        "confirm_report_format",
        route_after_confirm_report_format,
        {
            "html_report_generator": "html_report_generator",  # User chose A
            "answer_focused": "answer_focused",                # User chose B
        },
    )

    # All execution paths converge at respond, then the graph ends
    builder.add_edge("answer_focused", "respond")
    builder.add_edge("html_report_generator", "respond")
    builder.add_edge("respond", END)

    # ── Attach checkpointer and compile ───────────────────────────────────────
    # The checkpointer serialises the full state to SQLite after every node.
    # This is what allows the graph to pause at interrupt() calls and be
    # resumed in a future HTTP request with the exact same state.
    if checkpointer is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)
