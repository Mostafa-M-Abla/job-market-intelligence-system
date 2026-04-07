"""
graph/nodes/respond.py — Combine all sub-task answers into the final reply.

ROLE IN THE GRAPH
-----------------
This is the LAST node on every execution path.  All paths converge here once
all tasks in task_queue have been completed.

With the multi-task planner, answer_general, answer_focused, and
html_report_generator each append their result to accumulated_responses.
This node reads that list and:

  - Single entry (the common case): use it directly — no extra LLM call needed.
  - Multiple entries: make one gpt-4o-mini call to merge them into a single
    coherent reply with a clear heading per section.

BACKWARDS COMPATIBILITY
-----------------------
If accumulated_responses is empty (edge case: a path that sets
final_text_response directly without going through answer_focused, such as
the "resume missing" or "PDF parse error" short-circuits), the node falls
back to the old priority chain:
  1. final_text_response
  2. html_report_path  (builds a "report ready" message)
  3. market_analysis_markdown  (raw fallback)
  4. generic "I'm not sure" message

OUTPUTS written to state
------------------------
  messages : appends a new AIMessage with the final combined reply.
             The add_messages reducer ensures this appends to history.
"""

import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

from graph.state import JobMarketState

_COMBINE_SYSTEM_PROMPT = """You are a helpful assistant combining multiple answers into one response.

You will receive several answers, each for a different part of a compound user question.
Combine them into a single, well-structured reply:
- Give each part a clear Markdown heading (##)
- Keep each part's content exactly as provided — do not summarise or omit details
- Add a single connecting sentence at the top if the parts are related, otherwise just present them sequentially
- Do not add a preamble like "Here are your answers:" — go straight to the first heading"""


def _combine_responses(responses: list[str], config: RunnableConfig) -> str:
    """Call gpt-4o-mini to merge multiple sub-task answers into one reply."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    numbered = "\n\n---\n\n".join(
        f"### Part {i+1}\n\n{r}" for i, r in enumerate(responses)
    )
    logger.info("respond: combining %d responses via LLM", len(responses))
    result = llm.invoke([
        SystemMessage(content=_COMBINE_SYSTEM_PROMPT),
        HumanMessage(content=f"Combine these answers:\n\n{numbered}"),
    ], config=config)
    return result.content


def respond(state: JobMarketState, config: RunnableConfig) -> dict:
    """
    Produce the final reply from all accumulated sub-task answers.

    Args:
        state: Current graph state.
        config: LangGraph RunnableConfig for streaming token observability.

    Returns:
        {"messages": [AIMessage(content=reply)]}
    """
    session_id = state.get("session_id", "")
    logger.info("respond: finalising reply (session=%s)", session_id[:8] if session_id else "?")

    accumulated = list(state.get("accumulated_responses") or [])

    if len(accumulated) > 1:
        # Multiple sub-task answers — combine into one structured reply
        reply = _combine_responses(accumulated, config)

    elif len(accumulated) == 1:
        # Single answer — use directly, no extra LLM call
        reply = accumulated[0]

    else:
        # Fallback: accumulated_responses is empty — use legacy priority chain
        # (handles short-circuit paths like "resume missing" errors)
        final_text = state.get("final_text_response")
        html_path = state.get("html_report_path")
        market_md = state.get("market_analysis_markdown")

        if final_text:
            reply = final_text
        elif html_path:
            titles_str = ", ".join(state.get("job_titles") or [])
            reply = (
                f"Your report is ready!\n\n"
                f"[Download / View Report]({html_path})\n\n"
                f"The report covers:\n"
                f"- Market analysis for {titles_str}\n"
                f"- Top skills, cloud platforms, and certifications\n"
            )
            if state.get("skill_gap_markdown"):
                reply += "- Personalised skill gap analysis vs. your resume\n"
        elif market_md:
            reply = market_md
        else:
            reply = "I'm not sure how to respond. Please try rephrasing your question."

    logger.info("respond: done — %d chars", len(reply))
    return {"messages": [AIMessage(content=reply)]}
