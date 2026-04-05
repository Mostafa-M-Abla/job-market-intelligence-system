"""
tools/conversation_store.py — Persist and retrieve per-session conversation history.

WHY WE STORE CONVERSATIONS
---------------------------
The LangGraph checkpointer (SqliteSaver) already stores the full graph state
between turns — including the messages list.  So why add a separate conversation
history table?

  1. The checkpointer format is an internal LangGraph implementation detail.
     It is not suitable for returning clean message history to the frontend.

  2. The Phase 2 GET /api/chat/history/{session_id} endpoint needs a simple,
     well-structured query result to return to the frontend.

  3. The metadata column lets us attach useful context to each message:
     which intent was classified, whether a cache hit occurred, which report
     was generated, etc.  This is useful for analytics and debugging.

TWO TOOLS
---------
  ConversationSaveTool : called by the respond node at the end of every turn
  ConversationGetTool  : called by the Phase 2 API history endpoint

SESSION AUTO-CREATION
---------------------
ConversationSaveTool uses INSERT OR IGNORE on the sessions table before writing
the message, so sessions are created on first use without a separate setup step.
"""

import asyncio
import json
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from db.connection import get_connection


# ── Input schemas ─────────────────────────────────────────────────────────────

class SaveMessageInput(BaseModel):
    """Input for ConversationSaveTool."""
    session_id: str = Field(description="UUID identifying this user's session")
    role: str = Field(description="Who sent this message: 'human', 'ai', or 'tool'")
    content: str = Field(description="The message text to save")
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional extra context: intent, cache_key, report_path, etc."
    )


class GetHistoryInput(BaseModel):
    """Input for ConversationGetTool."""
    session_id: str = Field(description="UUID identifying the session to retrieve history for")
    limit: int = Field(default=50, description="Maximum number of messages to return (most recent first)")


# ── Tools ─────────────────────────────────────────────────────────────────────

class ConversationSaveTool(BaseTool):
    """
    Persist a single message (human or AI) to the conversation_history table.

    Also ensures the session row exists in the sessions table and updates the
    last_active timestamp so we can track when sessions were last used.

    Called by the respond node at the end of every conversation turn.
    """

    name: str = "conversation_save"
    description: str = "Persist a message (human/ai/tool) to conversation history for a session."
    args_schema: type[BaseModel] = SaveMessageInput

    def _run(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> str:
        """
        Write a message to the conversation_history table.

        Creates the session row if it doesn't exist yet (upsert pattern).
        Serialises the metadata dict to JSON for storage.

        Args:
            session_id: The user's session UUID.
            role:       "human", "ai", or "tool".
            content:    The message text.
            metadata:   Optional dict of extra context (intent, cache_key, etc.).

        Returns:
            "saved" on success.
        """
        with get_connection() as conn:
            # Ensure the session row exists before inserting the message,
            # to satisfy the foreign key constraint.
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, last_active) VALUES (?, datetime('now'))",
                (session_id,),
            )
            # Always update last_active so we know when the session was last used.
            conn.execute(
                "UPDATE sessions SET last_active = datetime('now') WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "INSERT INTO conversation_history (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(metadata) if metadata else None),
            )
        return "saved"

    async def _arun(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> str:
        """Async wrapper — runs the synchronous DB write in a thread pool."""
        return await asyncio.to_thread(self._run, session_id, role, content, metadata)


class ConversationGetTool(BaseTool):
    """
    Retrieve the conversation history for a session.

    Returns messages in chronological order (oldest first), up to the
    specified limit.  Used by the Phase 2 GET /api/chat/history endpoint.
    """

    name: str = "conversation_get"
    description: str = "Retrieve prior conversation messages for a session in chronological order."
    args_schema: type[BaseModel] = GetHistoryInput

    def _run(self, session_id: str, limit: int = 50) -> list[dict]:
        """
        Fetch the most recent messages for a session.

        Queries in descending order (newest first) then reverses so the
        result is chronological (oldest first) — the natural reading order.

        Args:
            session_id: The user's session UUID.
            limit:      Maximum number of messages to return.

        Returns:
            List of message dicts: {role, content, metadata, created_at}
            in chronological order.
        """
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, created_at FROM conversation_history "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()

        # Reverse so the result reads oldest-to-newest (chronological order).
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]

    async def _arun(self, session_id: str, limit: int = 50) -> list[dict]:
        """Async wrapper — runs the synchronous DB query in a thread pool."""
        return await asyncio.to_thread(self._run, session_id, limit)
