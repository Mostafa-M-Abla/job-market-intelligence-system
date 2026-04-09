"""
evaluation/shared/graph_runner.py

HITL-aware graph runner and trajectory capture utilities for evaluations.

Two public functions:
  run_graph_with_hitl()         — drives graph end-to-end, auto-confirms all interrupts.
  run_graph_capture_trajectory() — same but captures the ordered list of node names that fired.
"""

import sqlite3
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

load_dotenv()


def _default_hitl_strategy(interrupt_value: str) -> str:
    """
    Auto-confirm strategy for evaluations.

    confirm_search_params prompt contains "**confirm**" and "proceed"
    confirm_report_format prompt contains "Reply A or B"

    Always confirm search params; always choose B (text summary) for report
    format to avoid html_report_generator writing files during evals.
    """
    lower = interrupt_value.lower()
    if "reply a or b" in lower:
        return "B"
    # Default: confirm everything (covers confirm_search_params and any unknown interrupt)
    return "confirm"


def _get_interrupt_value(snapshot) -> Optional[str]:
    """Extract interrupt message from a graph snapshot, or None if not paused."""
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            return task.interrupts[0].value
    return None


def _make_graph_and_config():
    """
    Build a fresh graph with an in-memory SQLite checkpointer.

    Each evaluation example gets its own isolated graph instance to prevent
    state bleed between examples. Returns (graph, config, session_id).
    """
    from db.connection import init_db
    from graph.graph import build_graph

    init_db()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer=checkpointer)
    session_id = f"eval-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}
    return graph, config, session_id


def run_graph_with_hitl(
    message: str,
    total_posts: int = 3,
    max_interrupts: int = 5,
) -> dict:
    """
    Run the graph end-to-end, auto-handling all HITL interrupts.

    Args:
        message:        The user's input message.
        total_posts:    Number of SerpAPI posts to request (keep low for evals).
        max_interrupts: Safety ceiling on interrupt rounds.

    Returns:
        Final graph state dict.
    """
    graph, config, session_id = _make_graph_and_config()

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "total_posts": total_posts,
        "params_confirmed": False,
        "report_confirmed": False,
        "cache_hit": False,
    }

    result = graph.invoke(initial_state, config)

    for _ in range(max_interrupts):
        snapshot = graph.get_state(config)
        interrupt_val = _get_interrupt_value(snapshot)
        if not interrupt_val:
            break
        reply = _default_hitl_strategy(interrupt_val)
        result = graph.invoke(Command(resume=reply), config)

    return result


def run_graph_capture_trajectory(
    message: str,
    total_posts: int = 3,
    max_interrupts: int = 5,
) -> tuple[dict, list[str]]:
    """
    Run the graph, capturing the ordered sequence of node names that fire.

    Uses graph.stream(stream_mode="updates") which yields {node_name: state_delta}
    after every node execution — one entry per node, not per token.

    HITL interrupts cause the stream to exit cleanly. The loop resumes with a
    new stream() call; those node names are appended to the same trajectory list.

    Note: confirm_search_params fires (and appears in the trajectory) before
    interrupt() pauses execution, so it appears in the first stream call's output.
    Post-resume nodes (job_collector onward) appear in subsequent stream calls.

    Args:
        message:        The user's input message.
        total_posts:    Number of SerpAPI posts to request.
        max_interrupts: Safety ceiling on interrupt rounds.

    Returns:
        (final_state_dict, trajectory)
        trajectory is list[str] of node names in execution order.
    """
    graph, config, session_id = _make_graph_and_config()

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "total_posts": total_posts,
        "params_confirmed": False,
        "report_confirmed": False,
        "cache_hit": False,
    }

    trajectory: list[str] = []
    final_state: dict = {}

    def _stream_and_collect(input_or_command):
        nonlocal final_state
        for chunk in graph.stream(input_or_command, config, stream_mode="updates"):
            for node_name in chunk:
                if node_name != "__end__":
                    trajectory.append(node_name)
        final_state = graph.get_state(config).values

    _stream_and_collect(initial_state)

    for _ in range(max_interrupts):
        snapshot = graph.get_state(config)
        interrupt_val = _get_interrupt_value(snapshot)
        if not interrupt_val:
            break
        reply = _default_hitl_strategy(interrupt_val)
        _stream_and_collect(Command(resume=reply))

    return final_state, trajectory
