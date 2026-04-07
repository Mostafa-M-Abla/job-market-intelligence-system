"""
answer_focused node — answers a narrow, specific question from the market analysis data.
Used for focused_question intent (and as a fallback when user declines HTML report).
"""

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import JobMarketState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a job market intelligence assistant.

You have access to a market analysis report. Answer the user's specific question
concisely and precisely, citing frequencies and percentages from the data.

Guidelines:
- Focus only on what the user asked — don't dump the entire report
- Use numbers and percentages from the analysis
- Use a clear, readable format (bullet points or a small table if appropriate)
- Keep the answer under 400 words unless the question genuinely requires more
- If the question is about a specific topic (cloud platforms, databases, frameworks, etc.),
  only address that topic"""


def answer_focused(state: JobMarketState) -> dict:
    market_analysis = state.get("market_analysis_markdown") or ""
    skill_gap = state.get("skill_gap_markdown") or ""
    focused_topic = state.get("focused_topic") or ""
    intent = state.get("intent") or ""
    logger.info("answer_focused: generating focused answer (topic=%r, intent=%s)", focused_topic, intent)

    # Get the last human message
    from langchain_core.messages import HumanMessage as HM
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HM)),
        "Summarise the key findings.",
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    context_parts = []
    if market_analysis:
        context_parts.append(f"## Market Analysis\n\n{market_analysis}")
    if skill_gap and intent == "resume_analysis":
        context_parts.append(f"## Skill Gap Analysis\n\n{skill_gap}")

    context = "\n\n---\n\n".join(context_parts)

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Context (market data):\n\n{context}\n\n"
            f"---\n\nUser question: {last_human}"
            + (f"\n\nFocus specifically on: {focused_topic}" if focused_topic else "")
        )),
    ])

    logger.info("answer_focused: done — %d chars", len(response.content))
    return {"final_text_response": response.content}
