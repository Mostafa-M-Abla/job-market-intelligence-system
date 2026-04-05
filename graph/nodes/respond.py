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

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import JobMarketState
from tools.conversation_store import ConversationSaveTool

# Lazy initialisation — ConversationSaveTool connects to the DB.
_save_tool = None


def _get_save_tool() -> ConversationSaveTool:
    """Return the shared ConversationSaveTool instance, creating it on first call."""
    global _save_tool
    if _save_tool is None:
        _save_tool = ConversationSaveTool()
    return _save_tool


def _build_general_answer(state: JobMarketState) -> str:
    """
    Answer a general question using the LLM's own knowledge (no market data).

    This is only called for the general_question intent — questions like
    "What is LangGraph?" or "How do I improve my resume?".

    Uses gpt-4o-mini (fast and cheap) with a slightly higher temperature (0.3)
    to allow for more natural, conversational answers.

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
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful AI assistant specialising in AI engineering, "
            "machine learning, and career development. Answer clearly and concisely."
        )),
        HumanMessage(content=last_human),
    ])
    return response.content


def respond(state: JobMarketState) -> dict:
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

    # ── Determine the reply content ───────────────────────────────────────────

    if intent == "general_question":
        # No market data was needed — answer directly from LLM knowledge.
        reply = _build_general_answer(state)

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

    # ── Persist both turns to conversation history ────────────────────────────
    # We save here (at the very end) so that the history reflects the completed
    # turn, not a partial one.  In Phase 2 this feeds the GET /api/chat/history
    # endpoint so users can review past conversations.
    if session_id:
        # Save the user's message.
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        if last_human:
            _get_save_tool().run({"session_id": session_id, "role": "human", "content": last_human})

        # Save the AI's reply with metadata useful for debugging / analytics.
        _get_save_tool().run({
            "session_id": session_id,
            "role": "ai",
            "content": reply,
            "metadata": {
                "intent": intent,
                "cache_key": state.get("cache_key"),
                "report_path": state.get("html_report_path"),
                "cache_hit": state.get("cache_hit", False),
            },
        })

    # Append the AIMessage to the conversation history in the graph state.
    # The add_messages reducer (defined in state.py) ensures this APPENDS
    # rather than replacing the existing messages list.
    return {"messages": [AIMessage(content=reply)]}
