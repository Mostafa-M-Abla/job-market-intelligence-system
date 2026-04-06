"""
api/routers/history.py — Conversation history endpoint.

Endpoint:
  GET /api/chat/{session_id}/history — Fetch past messages for a session.

Uses the existing ConversationGetTool which queries the conversation_history
table ordered by created_at and returns them in chronological order.
"""

from fastapi import APIRouter

from tools.conversation_store import ConversationGetTool

router = APIRouter()


@router.get("/chat/{session_id}/history")
async def get_history(session_id: str, limit: int = 50):
    """
    Return the conversation history for a session.

    Response shape:
      {
        "session_id": "...",
        "messages": [
          {"role": "human", "content": "...", "metadata": null, "created_at": "..."},
          {"role": "ai",    "content": "...", "metadata": {...}, "created_at": "..."}
        ]
      }
    """
    tool = ConversationGetTool()
    messages = await tool._arun(session_id=session_id, limit=limit)
    return {"session_id": session_id, "messages": messages}
