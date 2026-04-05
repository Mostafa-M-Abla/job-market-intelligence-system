"""
graph/nodes/hitl.py — Human-in-the-Loop (HITL) nodes.

WHAT IS HITL?
-------------
HITL means pausing the AI pipeline and asking a real person to review or
confirm something before the system continues.  This is important for:
  - Avoiding unnecessary API costs (e.g. don't hit SerpAPI without user consent)
  - Giving users control over what the system does on their behalf
  - Choosing output format (HTML report vs. quick text summary)

HOW LANGGRAPH IMPLEMENTS HITL
------------------------------
LangGraph provides a special `interrupt()` function.  When a node calls
`interrupt(message)`, the graph:
  1. Saves the complete state to the checkpointer (SQLite/Postgres).
  2. Raises a special internal exception that pauses graph execution.
  3. Returns control to the caller (the API layer) with the interrupt message.

The graph stays "frozen" in this state until the caller resumes it by calling:
    graph.invoke(Command(resume="user's reply"), config)

This module contains two HITL nodes:
  - confirm_search_params  : shown before any SerpAPI call (cache miss only)
  - confirm_report_format  : shown after analysis — ask: HTML report or text?
"""

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from graph.state import JobMarketState


def confirm_search_params(state: JobMarketState) -> dict:
    """
    HITL node: pause before making any SerpAPI calls and show the user
    the search parameters that were extracted from their message.

    The user can either:
      - Reply "confirm" (or "yes", "ok", etc.) to proceed as-is.
      - Type a correction (e.g. "use 20 posts instead") to update the params.
        In that case, their reply is added to the message history and the graph
        loops back through intent_resolver to re-extract the updated parameters.

    This node only runs on a cache MISS — if we already have the market data,
    there is nothing to confirm.

    Args:
        state: Current graph state. Reads job_titles, country, total_posts.

    Returns:
        {"params_confirmed": True}  — if user confirmed
        {"params_confirmed": False, "messages": [HumanMessage(correction)]}
                                    — if user wants to change something
    """
    job_titles = state.get("job_titles") or []
    country = state.get("country") or "not specified"
    total_posts = state.get("total_posts", 30)

    # Build a clear, readable confirmation prompt for the user.
    prompt = (
        f"I'm about to search for job postings with the following parameters:\n\n"
        f"  Job titles : {', '.join(job_titles) if job_titles else 'not specified'}\n"
        f"  Country    : {country}\n"
        f"  Posts      : {total_posts}\n\n"
        f"Reply **confirm** to proceed, or tell me what to change."
    )

    # --- GRAPH PAUSES HERE ---
    # interrupt() serialises the state and pauses execution.
    # The value returned is whatever the user types in response.
    user_reply: str = interrupt(prompt)
    # --- GRAPH RESUMES HERE when graph.invoke(Command(resume=...), config) is called ---

    # Accept several common confirmation phrases case-insensitively.
    if user_reply.strip().lower() in ("confirm", "yes", "proceed", "ok", "go ahead"):
        return {"params_confirmed": True}

    # The user wants to change something.
    # We add their correction as a new HumanMessage so that when the graph
    # routes back to intent_resolver, it can parse the updated parameters.
    return {
        "params_confirmed": False,
        "messages": [HumanMessage(content=user_reply)],
    }


def confirm_report_format(state: JobMarketState) -> dict:
    """
    HITL node: after analysis is complete, ask the user how they want the
    results delivered — as a full HTML report or as a text summary in chat.

    This runs for full_market_analysis and resume_analysis intents.
    It is deliberately skipped for focused_question because a quick text
    answer is always the right format for a narrow question.

    Args:
        state: Current graph state. No specific fields are read beyond existence.

    Returns:
        {"report_confirmed": True}  — user wants an HTML report (choice A)
        {"report_confirmed": False} — user wants a text summary (choice B)
    """
    prompt = (
        "Analysis complete! How would you like the results?\n\n"
        "  **A** — Full HTML report (detailed, downloadable)\n"
        "  **B** — Text summary here in the chat\n\n"
        "Reply A or B."
    )

    # --- GRAPH PAUSES HERE ---
    user_reply: str = interrupt(prompt)
    # --- GRAPH RESUMES HERE ---

    # Any reply starting with "A" (case-insensitive) means HTML report.
    # Anything else (B, blank, unclear) defaults to text summary.
    wants_report = user_reply.strip().upper().startswith("A")
    return {"report_confirmed": wants_report}
