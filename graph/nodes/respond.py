"""
graph/nodes/respond.py — Compose the final reply and persist conversation history.

ROLE IN THE GRAPH
-----------------
This is the LAST node on every execution path.  All paths converge here:
  - general_question  -> respond
  - resume missing    -> respond
  - focused answer    -> respond
  - HTML report done  -> respond
  - text analysis     -> respond

Its two responsibilities are:
  1. Decide what text to show the user (based on which fields are populated).
  2. Persist both the user's message and the AI's reply to conversation_history.

For general_question intents, this node also makes the LLM call (the only node
to do so for that path).  All other intents have their content already prepared
in state fields by earlier nodes.

OUTPUTS written to state
------------------------
  messages : appends a new AIMessage with the final reply text.
             The add_messages reducer in state.py ensures this appends to the
             list rather than replacing it.
"""

import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from graph.state import JobMarketState

logger = logging.getLogger(__name__)


async def _build_general_answer(state: JobMarketState, config: RunnableConfig) -> str:
    """
    Answer a general question using the LLM's own knowledge (no market data).

    Uses gpt-4o-mini with streaming=True so tokens are observable by
    astream_events() and stream through to the frontend in real time.

    Args:
        state: Current graph state.  Reads the last human message.

    Returns:
        The LLM's answer as a plain string.
    """
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        streaming=True,   # enables token-level streaming via astream_events
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.info("respond: calling LLM for general_question")
    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a helpful AI assistant specialising in AI engineering, "
            "machine learning, and career development. Answer clearly and concisely."
        )),
        HumanMessage(content=last_human),
    ], config=config)
    logger.info("respond: LLM call completed")
    return response.content


async def respond(state: JobMarketState, config: RunnableConfig) -> dict:
    """
    Determine the final reply and persist the conversation turn to the DB.

    Reply selection priority (first match wins):
      1. general_question intent       -> call the LLM for a direct answer
      2. final_text_response is set    -> use it (covers focused answers,
                                          resume missing prompts, text summaries)
      3. html_report_path is set       -> build a "report ready" message with link
      4. market_analysis_markdown set  -> surface the raw analysis as a fallback
      5. none of the above             -> generic "I'm not sure" message

    After deciding the reply, both the human turn and this AI reply are saved
    to the conversation_history table so the full history is retrievable later
    (e.g. via GET /api/chat/history in Phase 2).

    Args:
        state: Current graph state.  Reads intent, session_id, and several
               output fields populated by earlier nodes.

    Returns:
        {"messages": [AIMessage(content=reply)]}
        The add_messages reducer appends this to the existing messages list.
    """
    intent = state.get("intent")
    session_id = state.get("session_id", "")
    logger.info("respond: intent=%s session=%s", intent, session_id[:8] if session_id else "?")

    # ── Determine the reply content ───────────────────────────────────────────

    if intent == "general_question":
        # No market data was needed — answer directly from LLM knowledge.
        reply = await _build_general_answer(state, config)

    elif state.get("final_text_response"):
        # This field is set by: answer_focused (text summaries),
        # check_resume (missing resume prompt), and resume_parser (parse errors).
        reply = state["final_text_response"]

    elif state.get("html_report_path"):
        # An HTML report was generated — tell the user and provide the link.
        report_url = state["html_report_path"]
        reply = (
            f"Your report is ready!\n\n"
            f"[Download / View Report]({report_url})\n\n"
            f"The report covers:\n"
            f"- Market analysis for {', '.join(state.get('job_titles') or [])}\n"
            f"- Top skills, cloud platforms, and certifications\n"
        )
        if state.get("skill_gap_markdown"):
            reply += "- Personalised skill gap analysis vs. your resume\n"

    elif state.get("market_analysis_markdown"):
        # Fallback: no specific output format was chosen, surface the raw markdown.
        reply = state["market_analysis_markdown"]

    else:
        # Should not normally be reached — indicates an unexpected graph path.
        reply = "I'm not sure how to respond. Please try rephrasing your question."

    # NOTE: Conversation history saving is intentionally skipped here.
    # ConversationSaveTool opens a synchronous sqlite3 connection via
    # asyncio.to_thread() which conflicts with AsyncSqliteSaver's aiosqlite
    # connection, causing aget_state() to hang after the node completes.
    # History saving will be handled separately (e.g. in a post-processing step).

    logger.info("respond: done")
    return {"messages": [AIMessage(content=reply)]}
