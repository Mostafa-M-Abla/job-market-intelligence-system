"""
graph/nodes/intent_resolver.py — The entry node for every conversation turn.

ROLE IN THE GRAPH
-----------------
This is the first node that runs on every user message.  Its job is to figure
out what the user wants to do (their "intent") and extract any structured
parameters from their message (job titles, country, focused topic).

The output of this node drives all subsequent routing decisions.

HOW IT WORKS
------------
We use LangChain's `with_structured_output()` to make the LLM return a typed
Pydantic object (IntentResolution) rather than free-form text.  This guarantees
we always get valid JSON with the exact fields we need — no parsing required.

The LLM used is gpt-4o-mini (fast and cheap) with temperature=0 (deterministic,
no creativity needed for classification).

OUTPUTS written to state
------------------------
  intent        : one of the 4 intent strings
  job_titles    : list of job roles extracted from the message
  country       : country name (or None)
  focused_topic : narrow topic for focused_question (or None)
  total_posts   : number of job postings to collect (from state or env var)
  params_confirmed, report_confirmed, cache_hit : reset to False for each new turn
"""

import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import DEFAULT_TOTAL_POSTS
from graph.state import JobMarketState

# The system prompt tells the LLM exactly how to classify messages and what
# to extract.  Concrete examples for each intent help the model be consistent.
_SYSTEM_PROMPT = """You are an intent classifier for a Job Market Intelligence System.

Classify the user's latest message into exactly one of these intents:

1. full_market_analysis — User wants a comprehensive market analysis for given job titles and country.
   Examples: "Do a full market analysis for Data Engineers in Germany",
             "Analyze the job market for ML engineers in the US"

2. focused_question — User wants a narrow, specific answer about market trends.
   Examples: "What are the most in-demand cloud platforms for AI jobs?",
             "Which databases are most popular in backend job postings?",
             "What frameworks do DevOps jobs require most?"

3. resume_analysis — User wants their resume reviewed, skill gap identified, or personalized
   learning recommendations. They may or may not reference their resume explicitly.
   Examples: "What skills should I learn next?", "Compare my CV to the market",
             "What's missing from my resume for ML engineer roles?"

4. general_question — A general question answerable from AI knowledge, not requiring job market data.
   Examples: "What is LangGraph?", "How do I write a good resume?",
             "Explain the difference between supervised and unsupervised learning"

Also extract:
- job_titles: list of job titles mentioned or implied (e.g. ["Data Engineer", "Analytics Engineer"])
  Leave empty [] for general_question or if no titles mentioned.
- country: country name if mentioned, else null
- focused_topic: for focused_question only — the specific topic (e.g. "cloud platforms", "databases",
  "frameworks"). null for other intents.

Always return valid JSON matching the schema."""


class IntentResolution(BaseModel):
    """
    Structured output schema for the intent classifier.

    LangChain's `with_structured_output()` will force the LLM to return JSON
    that matches this exact shape.  If the LLM produces invalid JSON, LangChain
    automatically retries before raising an error.
    """
    intent: str = Field(
        description="One of: full_market_analysis, focused_question, resume_analysis, general_question"
    )
    job_titles: list[str] = Field(
        default_factory=list,
        description="Job roles extracted from the message, e.g. ['AI Engineer', 'ML Engineer']"
    )
    country: Optional[str] = Field(
        default=None,
        description="Country name if mentioned in the message, otherwise null"
    )
    focused_topic: Optional[str] = Field(
        default=None,
        description="For focused_question only: the specific topic being asked about"
    )


def intent_resolver(state: JobMarketState) -> dict:
    """
    Classify the user's latest message and extract structured parameters.

    This node runs first on every turn.  It reads only the most recent human
    message (not the full history) so that each turn is classified independently.

    Args:
        state: Current graph state.  Only `messages` is read here.

    Returns:
        A dict updating: intent, job_titles, country, focused_topic, total_posts,
        and resetting HITL flags (params_confirmed, report_confirmed, cache_hit)
        so they don't carry over from a previous turn.
    """
    # Build the LLM with structured output.
    # `with_structured_output(IntentResolution)` wraps the LLM so it always
    # returns an IntentResolution object instead of a raw string.
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,          # No randomness — classification should be deterministic
        api_key=os.getenv("OPENAI_API_KEY"),
    ).with_structured_output(IntentResolution)

    # Find the most recent human message in the conversation history.
    # We reverse the list and take the first HumanMessage we find.
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    # Ask the LLM to classify and extract.
    result: IntentResolution = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=last_human),
    ])

    return {
        "intent": result.intent,
        # Prefer newly extracted titles; fall back to whatever was in state already
        # (useful when user confirms params without repeating the job titles).
        "job_titles": result.job_titles or state.get("job_titles") or [],
        "country": result.country or state.get("country"),
        "focused_topic": result.focused_topic,
        # Use the value from state (set at graph invocation) or the env-var default.
        "total_posts": state.get("total_posts") or DEFAULT_TOTAL_POSTS,
        # Reset HITL flags at the start of every new turn so they don't bleed
        # over from a previous conversation turn.
        "params_confirmed": False,
        "report_confirmed": False,
        "cache_hit": False,
    }
