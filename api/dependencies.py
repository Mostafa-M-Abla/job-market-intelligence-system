"""
api/dependencies.py — Shared FastAPI dependencies.

Holds the graph singleton so it is built once at startup and reused
across all requests, avoiding repeated checkpointer initialisation.

Uses AsyncSqliteSaver (backed by aiosqlite) instead of the synchronous
SqliteSaver so that graph.astream_events() and graph.aget_state() work
correctly inside FastAPI's async event loop.

The synchronous SqliteSaver is kept in graph/graph.py as the default for
run_tests.py and other script-mode callers that use graph.invoke().
"""

import os

import aiosqlite
from fastapi import Depends
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from graph.graph import build_graph

DB_PATH = os.getenv("DB_PATH", "job_market.db")

_graph = None


def get_graph():
    """FastAPI dependency that returns the compiled LangGraph instance."""
    return _graph


async def startup() -> None:
    """Called from the FastAPI lifespan context manager on app startup."""
    global _graph
    conn = await aiosqlite.connect(DB_PATH)
    await conn.execute("PRAGMA journal_mode=WAL")
    checkpointer = AsyncSqliteSaver(conn)
    _graph = build_graph(checkpointer=checkpointer)


async def shutdown() -> None:
    """Called from the FastAPI lifespan context manager on app shutdown."""
    pass
