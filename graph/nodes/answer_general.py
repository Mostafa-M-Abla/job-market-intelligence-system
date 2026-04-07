"""
graph/nodes/answer_general.py — Answer a general question from LLM knowledge.

ROLE IN THE GRAPH
-----------------
This node handles the general_question intent.  It was previously handled
inside respond.py (_build_general_answer), but was moved here so that:

  1. The respond node becomes a pure "combine and finalise" node with no LLM calls.
  2. The loop edge (route_after_task_complete) can sit after this node — if more
     tasks remain in the queue, the graph loops back to task_dispatcher instead
     of going straight to respond.

WHY STREAMING=TRUE ON A SYNC FUNCTION?
---------------------------------------
streaming=True enables token-level emission when the graph is driven by
astream_events() (the FastAPI path).  LangGraph's event stream intercepts
each LLM chunk and emits on_chat_model_stream events regardless of whether
the node function itself is sync or async.  Using llm.invoke() (sync) keeps
this node compatible with both graph.invoke() (run_tests.py) and
graph.astream_events() (FastAPI).

OUTPUTS written to state
------------------------
  accumulated_responses : appends the LLM answer as a new entry
"""

import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from graph.state import JobMarketState

logger = logging.getLogger(__name__)


def answer_general(state: JobMarketState, config: RunnableConfig) -> dict:
    """
    Answer a general knowledge question using gpt-4o-mini.

    Args:
        state: Current graph state.  Reads the last human message.
        config: LangGraph RunnableConfig — passed to llm.invoke() so tokens
                are observable via astream_events() in the FastAPI path.

    Returns:
        {"accumulated_responses": [*existing, new_answer]}
    """
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        streaming=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    logger.info("answer_general: calling LLM for general_question")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful AI assistant specialising in AI engineering, "
            "machine learning, and career development. Answer clearly and concisely."
        )),
        HumanMessage(content=last_human),
    ], config=config)
    logger.info("answer_general: done — %d chars", len(response.content))

    existing = list(state.get("accumulated_responses") or [])
    return {"accumulated_responses": existing + [response.content]}
