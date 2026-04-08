"""
graph/graph.py — Assembles and compiles the LangGraph StateGraph.

This is the "wiring" file — it does not contain any business logic.
Its only job is to:
  1. Create a StateGraph and register every node.
  2. Connect nodes with edges (fixed and conditional).
  3. Attach a checkpointer so state is persisted to SQLite between turns.
  4. Compile and return the runnable graph.

Graph topology (high level):

  START
    │
    ▼
  planner          ← decomposes message into 1–4 tasks
    │
    ▼
  task_dispatcher  ← pops next task, sets intent/job_titles/country
    │
    ▼ (route_after_intent)
  ┌─────────────────────────────────────────────────────┐
  │ answer_general  │  check_resume  │  cache_lookup     │
  └─────────────────────────────────────────────────────┘
                             ... (existing pipelines) ...
                             │
                             ▼ (route_after_task_complete)
                    ┌────────────────────┐
                    │ more tasks?        │
                    │  YES → task_dispatcher (loop)
                    │  NO  → respond
                    └────────────────────┘
                             │
                             ▼
                           respond
                             │
                             ▼
                            END
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

# ── Node imports ──────────────────────────────────────────────────────────────
from graph.nodes.answer_focused import answer_focused
from graph.nodes.answer_general import answer_general
from graph.nodes.cache_lookup import cache_lookup
from graph.nodes.check_resume import check_resume
from graph.nodes.hitl import confirm_report_format, confirm_search_params
from graph.nodes.html_report_generator import html_report_generator
from graph.nodes.job_collector import job_collector
from graph.nodes.market_analyzer import market_analyzer
from graph.nodes.planner import planner
from graph.nodes.requirements_extractor import requirements_extractor
from graph.nodes.respond import respond
from graph.nodes.resume_parser import resume_parser
from graph.nodes.skill_gap_analyzer import skill_gap_analyzer
from graph.nodes.task_dispatcher import task_dispatcher

# ── Routing function imports ───────────────────────────────────────────────────
from graph.routing import (
    route_after_cache_lookup,
    route_after_check_resume,
    route_after_confirm_report_format,
    route_after_confirm_search_params,
    route_after_intent,
    route_after_job_collector,
    route_after_market_analyzer,
    route_after_resume_parser,
    route_after_skill_gap,
    route_after_task_complete,
)
from graph.state import JobMarketState

DB_PATH = os.getenv("DB_PATH", "job_market.db")


def build_graph(checkpointer=None):
    """
    Build and compile the full Job Market Intelligence StateGraph.

    Args:
        checkpointer: A LangGraph checkpointer. If None, a SqliteSaver backed
                      by DB_PATH is created automatically.

    Returns:
        A compiled LangGraph graph ready for .invoke() or .astream_events().
    """
    builder = StateGraph(JobMarketState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("planner", planner)                             # Decompose into 1–4 tasks
    builder.add_node("task_dispatcher", task_dispatcher)             # Pop next task, set state
    builder.add_node("answer_general", answer_general)               # LLM answer (general questions)
    builder.add_node("check_resume", check_resume)                   # Is a resume uploaded?
    builder.add_node("resume_parser", resume_parser)                 # PDF → text
    builder.add_node("cache_lookup", cache_lookup)                   # Check market data cache
    builder.add_node("confirm_search_params", confirm_search_params) # HITL: confirm job search params
    builder.add_node("job_collector", job_collector)                 # Fetch jobs from SerpAPI
    builder.add_node("requirements_extractor", requirements_extractor)  # LLM: extract skills per job
    builder.add_node("market_analyzer", market_analyzer)             # LLM: aggregate + write cache
    builder.add_node("skill_gap_analyzer", skill_gap_analyzer)       # LLM: resume vs. market
    builder.add_node("answer_focused", answer_focused)               # LLM: answer specific question
    builder.add_node("confirm_report_format", confirm_report_format) # HITL: HTML or text?
    builder.add_node("html_report_generator", html_report_generator) # LLM: generate HTML report
    builder.add_node("respond", respond)                             # Combine answers + finalise

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "task_dispatcher")

    # ── Route by intent (after task_dispatcher sets the intent field) ─────────
    builder.add_conditional_edges(
        "task_dispatcher",
        route_after_intent,
        {
            "answer_general": "answer_general",  # general_question — LLM only
            "check_resume": "check_resume",       # resume_analysis — check PDF first
            "cache_lookup": "cache_lookup",       # market intents — check cache first
        },
    )

    # ── Resume check path ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "check_resume",
        route_after_check_resume,
        {
            "resume_parser": "resume_parser",
            "respond": "respond",
        },
    )

    builder.add_conditional_edges(
        "resume_parser",
        route_after_resume_parser,
        {
            "cache_lookup": "cache_lookup",
            "respond": "respond",
        },
    )

    # ── Cache lookup path ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "cache_lookup",
        route_after_cache_lookup,
        {
            "answer_focused": "answer_focused",
            "skill_gap_analyzer": "skill_gap_analyzer",
            "confirm_report_format": "confirm_report_format",
            "confirm_search_params": "confirm_search_params",
        },
    )

    # ── HITL: search param confirmation ───────────────────────────────────────
    builder.add_conditional_edges(
        "confirm_search_params",
        route_after_confirm_search_params,
        {
            "job_collector": "job_collector",  # User confirmed — start collecting
            "planner": "planner",              # User edited — re-plan from updated message
        },
    )

    # ── After job collection: skip pipeline if no postings were found ─────────
    builder.add_conditional_edges(
        "job_collector",
        route_after_job_collector,
        {
            "requirements_extractor": "requirements_extractor",
            "task_dispatcher": "task_dispatcher",
            "respond": "respond",
        },
    )
    builder.add_edge("requirements_extractor", "market_analyzer")

    # ── After market analysis: fan out by intent ──────────────────────────────
    builder.add_conditional_edges(
        "market_analyzer",
        route_after_market_analyzer,
        {
            "answer_focused": "answer_focused",
            "skill_gap_analyzer": "skill_gap_analyzer",
            "confirm_report_format": "confirm_report_format",
        },
    )

    # ── Skill gap always leads to report format choice ────────────────────────
    builder.add_conditional_edges(
        "skill_gap_analyzer",
        route_after_skill_gap,
        {"confirm_report_format": "confirm_report_format"},
    )

    # ── HITL: report format choice ────────────────────────────────────────────
    builder.add_conditional_edges(
        "confirm_report_format",
        route_after_confirm_report_format,
        {
            "html_report_generator": "html_report_generator",
            "answer_focused": "answer_focused",
        },
    )

    # ── After each task-terminal node: loop or finish ─────────────────────────
    # These three nodes produce one answer per task.  After each one, check
    # if more tasks are queued; if so, loop back to task_dispatcher.
    for terminal_node in ("answer_general", "answer_focused", "html_report_generator"):
        builder.add_conditional_edges(
            terminal_node,
            route_after_task_complete,
            {
                "task_dispatcher": "task_dispatcher",  # More tasks remain
                "respond": "respond",                  # All tasks done
            },
        )

    # ── Final convergence ─────────────────────────────────────────────────────
    builder.add_edge("respond", END)

    # ── Attach checkpointer and compile ───────────────────────────────────────
    if checkpointer is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)
